"""Make the mHC hyper-connection mix a set of real submodules.

Not a hardware adaptation — the arithmetic here is bit-for-bit the vendor's. It is a
*structural* patch, and it exists because `hc_mixes`, `hc_pre` and `hc_post` are methods on
`Block` rather than `nn.Module`s, with their coefficients held as `nn.Parameter`s on the
block itself. A hook-based trace has nothing to attach to a method, so that arithmetic
belonged to no partition module: the plan's residual stream was rank 3 where the model
carries rank 4, `verify_chain` could carry 3 of 27 boundaries, and the 258 `hc_*`
parameters sat in `metadata.layer_owned_params` owned by nothing.

What this changes, and nothing else:

- `Block` gains four children — `hc_attn_in`, `hc_attn_out`, `hc_ffn_in`, `hc_ffn_out` —
  and its `forward` calls them instead of its own methods. The `*_in` pair takes the
  layer's `hc_*_fn/base/scale` with it, so exactly one thing owns each parameter; the
  spec's `checkpoint.rename` rules put the checkpoint's names onto the new paths.
- `Transformer` gains `hc_expand` (the `unsqueeze(2).repeat` after `embed`, plus the
  identity `pre_mix`) and `hc_collapse` (the final `hc_pre` before `norm`), which are the
  two ends of the rank-4 stream and were likewise inside a method.

Every formula below is copied verbatim from `Block.hc_mixes` / `hc_pre` / `hc_post` and
`Transformer.forward`. `tests/test_partition_compat.py` runs a stand-in carrying those
methods' text before and after patching and asserts the logits are bit-identical, because
a "structural" patch that changed a number would silently move the reference every later
stage is measured against.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from kernel import hc_split_sinkhorn
# Defined in the modeling file rather than in `kernel`, and imported from there so the
# slice `extract` takes of this file keeps a working import.
from model import make_identity_pre_mix


class HyperConnectSublayerIn(nn.Module):
    """The coefficients a sublayer derives, and the one stream its sublayer sees.

    `hc_mixes` and `hc_pre` together: the projection's only consumer is the Sinkhorn, and
    splitting them apart would materialise a `[b, s, (2 + hc) * hc]` fp32 tensor for
    nothing. Returns `(x_in, post, comb, pre)` — `x_in` for this sublayer, `post` and
    `comb` for the `Out` half, and `pre` for the *next* sublayer, which is the vendor's
    "the coefficients a sublayer computes are used by the next one".

    Built the way any module of the model is: from config fields, named as `ModelArgs`
    names them, with the checkpoint's tensors loaded onto the parameters afterwards. A
    constructor that took the parameters themselves could only ever be called from inside
    `Block.__init__`, so nothing that rebuilds one module from its config and a state
    dict — which is how every stage after the trace runs it — could construct it at all.
    """

    def __init__(self, dim: int, hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float,
                 norm_eps: float) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps
        mix_hc = (2 + hc_mult) * hc_mult
        # float32 inside a bfloat16 model, as `Block.__init__` builds them under
        # `set_dtype(torch.float32)`.
        self.fn = nn.Parameter(torch.empty(mix_hc, hc_mult * dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(self, x: torch.Tensor, pre_mix: torch.Tensor):
        # Block.hc_mixes, verbatim: normalized over the whole flattened hc*d stream, one
        # statistic per token.
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        # Block.hc_pre, verbatim: collapse the hc copies into one, weighted by pre_mix.
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype), post, comb, pre


class HyperConnectSublayerOut(nn.Module):
    """`hc_post`: expand the sublayer output back to hc copies, residual mixed in."""

    def forward(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                comb: torch.Tensor):
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        return y.type_as(x)


class HyperConnectExpand(nn.Module):
    """Where the rank-4 stream begins: one copy per hyper-connection, and a one-hot mix.

    `make_identity_pre_mix` is why the first layer's `hc_pre` returns copy 0 unchanged,
    which is the coincidence that made `embed -> layers.0.attention` look like the one
    working edge in a plan where no edge worked.
    """

    def __init__(self, hc_mult: int) -> None:
        super().__init__()
        self.hc_mult = hc_mult

    def forward(self, h: torch.Tensor):
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return h, make_identity_pre_mix(h, self.hc_mult)


class HyperConnectCollapse(nn.Module):
    """Where it ends: the final `hc_pre` into the output norm."""

    def forward(self, x: torch.Tensor, pre_mix: torch.Tensor):
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)


def _install_block(block) -> None:
    """Give one block its four mHC children, moving the coefficients onto them."""
    for which in ("attn", "ffn"):
        fn = block._parameters.pop(f"hc_{which}_fn", None)
        base = block._parameters.pop(f"hc_{which}_base", None)
        scale = block._parameters.pop(f"hc_{which}_scale", None)
        if fn is None or base is None or scale is None:
            # Already moved: the coefficients live on the submodule now.
            continue
        # On meta, because its own freshly allocated coefficients are replaced at once by
        # the block's: the very tensors the loader has already placed or will stream into.
        with torch.device("meta"):
            mix_in = HyperConnectSublayerIn(
                fn.shape[1] // block.hc_mult, block.hc_mult, block.hc_sinkhorn_iters,
                block.hc_eps, block.norm_eps)
        mix_in.fn, mix_in.base, mix_in.scale = fn, base, scale
        block.add_module(f"hc_{which}_in", mix_in)
        block.add_module(f"hc_{which}_out", HyperConnectSublayerOut())


def _block_forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
    """`Block.forward` with the mix, the collapse and the expansion called as submodules.

    Identical in arithmetic and in order to the original. The only difference is that each
    step is now a call a hook can see, which is what gives it a traced reference and a
    module of its own.
    """
    residual = x
    x, attn_post, attn_comb, attn_pre = self.hc_attn_in(x, pre_mix)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos, *attn_args)
    x = self.hc_attn_out(x, residual, attn_post, attn_comb)

    residual = x
    x, ffn_post, ffn_comb, ffn_pre = self.hc_ffn_in(x, attn_pre)
    x = self.ffn_norm(x)
    x = self.ffn(x, image_mask)
    x = self.hc_ffn_out(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre


def _make_transformer_forward(vendor):
    """`Transformer.forward` with the stream's two ends called as submodules.

    Copied line for line from the vendor's, with exactly two substitutions: the
    `unsqueeze(2).repeat` plus `make_identity_pre_mix` pair becomes `self.hc_expand(h)`,
    and the trailing `layer.hc_pre(h, pre_mix)` becomes `self.hc_collapse(h, pre_mix)`.
    """

    @torch.inference_mode()
    def forward(self, input_ids, start_pos: int = 0, images=None, token_types=None):
        image_mask = None if token_types is None else token_types >= 0  # TEXT is -1
        engram_mask = None if image_mask is None else ~image_mask
        engram_hashes = (self.engram_hash(input_ids, start_pos, engram_mask)
                         if self.engram_hash is not None else None)
        h = self.embed(input_ids)
        if images is not None:
            assert start_pos == 0, "image spans must be prefilled in a single chunk"
            self.merge_image_embeddings(images, h)
        # Expand to hc_mult copies for Hyper-Connections, and start from a one-hot mix.
        h, pre_mix = self.hc_expand(h)
        main_hiddens = []
        for i, layer in enumerate(self.layers):
            if layer.engram is not None:
                h = layer.engram(
                    h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
            # the MTP head reads the attention input of its target layers, not their output
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
        h = self.hc_collapse(h, pre_mix)
        logits = self.head(self.norm(h))
        output_ids = vendor.sample(logits, self.temperature)
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return output_ids, logits, main_hidden

    return forward


#: Marks the classes as already patched. A patch that *rebinds* a name is idempotent for
#: free; this one wraps `__init__`, so applying it twice would wrap the wrapper and try to
#: move the coefficients a second time — `KeyError: 'hc_attn_fn'` from inside construction.
#: A run builds the model more than once (once on meta for structure, once for real), and
#: patches are applied each time.
PATCH_MARKER = "_hyper_connections_patched"

REPLACED = ["Block.hc_mixes", "Block.hc_pre", "Block.hc_post",
            "Transformer.hc_expand", "Transformer.hc_collapse"]


def apply(vendor, device):
    """Replace the mHC methods with submodules. Returns what was replaced."""
    del device  # structural: the same on every card

    block_cls = vendor.Block
    if getattr(block_cls, PATCH_MARKER, False):
        return REPLACED
    setattr(block_cls, PATCH_MARKER, True)
    original_init = block_cls.__init__

    # `functools.wraps` is load-bearing, not tidiness. The loader decides what to hand the
    # model factory by inspecting its signature — a `tokenizer` parameter is how DeepSeek's
    # engram layout gets the vocabulary it builds its compressed token map from. A
    # replacement `__init__` declaring only `*args, **kwargs` advertises no such parameter,
    # so the tokenizer is silently not passed and construction dies inside `engram.py` on
    # `None.backend_tokenizer`. Copying `__wrapped__` over is what keeps
    # `inspect.signature` telling the truth about the constructor.
    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _install_block(self)

    block_cls.__init__ = __init__
    block_cls.forward = _block_forward

    transformer_cls = vendor.Transformer
    original_transformer_init = transformer_cls.__init__

    @functools.wraps(original_transformer_init)
    def transformer_init(self, *args, **kwargs):
        original_transformer_init(self, *args, **kwargs)
        self.add_module("hc_expand", HyperConnectExpand(self.hc_mult))
        self.add_module("hc_collapse", HyperConnectCollapse())

    transformer_cls.__init__ = transformer_init
    transformer_cls.forward = _make_transformer_forward(vendor)
    return REPLACED
