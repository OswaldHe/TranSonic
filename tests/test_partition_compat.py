# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for porting a model's own code to the GPU in front of us.

A vendor's kernels are written for the vendor's hardware. When this card refuses one,
the model cannot run, there is no trace and nothing to partition — so the loop treats
that as work rather than as a wall: the agent writes a compatibility patch under
``compat/`` and the trace is taken again with it.
"""

from pathlib import Path

import pytest

from model_partition.runtime.compat import apply_patches, is_hardware_limit, patch_paths


def test_a_kernel_this_card_refuses_is_told_apart_from_a_bug():
    """Who gets the failure depends on this: a porter, or whoever owns the arithmetic."""
    assert is_hardware_limit(RuntimeError(
        "Failed to set the allowed dynamic shared memory size to 141312"))
    assert is_hardware_limit(RuntimeError("CUDA error: no kernel image is available"))
    assert is_hardware_limit(RuntimeError("kernel input device_type mismatch, expected cuda"))
    # Not hardware: these belong to whoever wrote the code.
    assert not is_hardware_limit(RuntimeError("shapes cannot be multiplied (8x4 and 5x2)"))
    assert not is_hardware_limit(ValueError("expected a positive integer"))
    assert not is_hardware_limit(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))


def test_patches_are_discovered_in_order(tmp_path):
    (tmp_path / "compat").mkdir()
    for name in ("20-second.py", "10-first.py", "_helper.py"):
        (tmp_path / "compat" / name).write_text("def apply(vendor, device):\n    return []\n")
    assert [p.name for p in patch_paths(tmp_path)] == ["10-first.py", "20-second.py"]
    assert patch_paths(tmp_path / "absent") == []


def test_a_patch_replaces_what_will_not_run(tmp_path):
    """The point: the model's own module now calls the ported version."""
    import types

    vendor = types.SimpleNamespace(sparse_attn=lambda *a: "hopper kernel", other=1)
    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "sparse_attn.py").write_text(
        '"""sparse_attn wants 141312 bytes of shared memory; this card allows 101376."""\n'
        "\n"
        "def _ported(*args):\n"
        "    return 'torch fallback'\n"
        "\n"
        "def apply(vendor, device):\n"
        "    vendor.sparse_attn = _ported\n"
        "    return ['sparse_attn']\n"
    )
    report = apply_patches(vendor, patch_paths(tmp_path), device=None)
    assert report.applied
    assert report.replaced == ["sparse_attn"]
    assert vendor.sparse_attn() == "torch fallback"
    assert vendor.other == 1
    assert "replaced 1" in report.summary()


def test_a_patch_is_told_what_it_is_adapting_to(tmp_path):
    """So it can branch on what the card supports instead of on its name."""
    import types

    from model_partition.hardware import GPUInfo

    vendor = types.SimpleNamespace()
    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "p.py").write_text(
        "def apply(vendor, device):\n"
        "    vendor.shared = device.shared_memory_per_block\n"
        "    vendor.fp8 = device.supports_fp8()\n"
        "    return ['shared', 'fp8']\n"
    )
    card = GPUInfo(index=0, name="NVIDIA L40S", total_bytes=1, free_bytes=1,
                   capability=(8, 9), shared_memory_per_block=101376)
    apply_patches(vendor, patch_paths(tmp_path), device=card)
    assert vendor.shared == 101376 and vendor.fp8 is True


def test_a_broken_patch_is_reported_not_ignored(tmp_path):
    import types

    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "bad.py").write_text("raise RuntimeError('boom')\n")
    (tmp_path / "compat" / "empty.py").write_text("VALUE = 1\n")
    report = apply_patches(types.SimpleNamespace(), patch_paths(tmp_path), None)
    assert not report.applied
    assert set(report.errors) == {"bad.py", "empty.py"}
    assert "failed" in report.summary()


def test_the_loader_applies_a_patch_and_refuses_a_broken_one(tmp_path):
    """The patch runs at import, before anything is constructed or traced."""
    from model_partition.ingest import ingest
    from model_partition.loaders import LoaderError, build_loader
    from model_partition.spec import parse_spec
    from tests.fixtures.tiny_llm import TinyConfig, write_tiny_repo

    pytest.importorskip("torch")
    repo = write_tiny_repo(tmp_path / "repo", TinyConfig())
    spec = parse_spec({"source": str(repo), "name": "tiny", "dtype": "float32",
                       "trust_remote_code": True})
    result = ingest(spec)

    directory = tmp_path / "compat"
    directory.mkdir()
    (directory / "marker.py").write_text(
        "def apply(vendor, device):\n"
        "    vendor.PORTED_HERE = True\n"
        "    return ['PORTED_HERE']\n"
    )
    loader = build_loader(result, compat_paths=patch_paths(tmp_path))
    loaded = loader.build_meta()
    assert loaded.model is not None
    assert loader.compat_report.replaced == ["PORTED_HERE"]

    (directory / "marker.py").write_text("def apply(vendor, device):\n    raise KeyError('x')\n")
    with pytest.raises(LoaderError, match="compatibility patch"):
        build_loader(result, compat_paths=patch_paths(tmp_path)).build_meta()


def _context(tiny_run, tmp_path):
    """A real loop context for the toy model, up to the point of tracing."""
    from model_partition.loop.driver import PartitionLoop
    from model_partition.loop.stages import LoopOptions, stage_ingest, stage_plan

    options = LoopOptions(artifact_root=str(tmp_path / "artifacts"), device="cpu",
                          trace_device="cpu", judge_kind="stub", max_iterations=1,
                          use_agent_planner=False, gpu_memory_gib=1.0)
    ctx = PartitionLoop(spec=tiny_run.spec, options=options,
                        report=lambda _m: None).build_context()
    stage_ingest(ctx)
    stage_plan(ctx)
    return ctx


def test_a_hardware_failure_in_the_trace_goes_to_the_porting_surface(tiny_run, tmp_path,
                                                                    monkeypatch):
    """Not the plan and not the arithmetic: this one is a porting job."""
    from model_partition.loop.stages import stage_trace
    from model_partition.trace import Tracer

    ctx = _context(tiny_run, tmp_path)

    def refuse(self, *args, **kwargs):
        raise RuntimeError("Failed to set the allowed dynamic shared memory size to 141312")

    monkeypatch.setattr(Tracer, "trace_sample", refuse)
    result = stage_trace(ctx)
    assert not result.ok and result.repairable
    assert result.repair_surface == "compat"
    assert "does not run on this GPU" in result.detail


def test_an_ordinary_trace_failure_is_not_a_porting_job(tiny_run, tmp_path, monkeypatch):
    """A wrong shape is a bug in the code, and porting it would fix nothing."""
    from model_partition.loop.stages import stage_trace
    from model_partition.trace import Tracer

    ctx = _context(tiny_run, tmp_path)

    def broken(self, *args, **kwargs):
        raise RuntimeError("shapes cannot be multiplied (8x4 and 5x2)")

    monkeypatch.setattr(Tracer, "trace_sample", broken)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        stage_trace(ctx)


def test_importing_a_patch_leaves_no_bytecode_beside_it(tmp_path):
    """A run directory is a deliverable someone reads and copies."""
    import types

    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "p.py").write_text("def apply(vendor, device):\n    return []\n")
    apply_patches(types.SimpleNamespace(), patch_paths(tmp_path), None)
    assert not (tmp_path / "compat" / "__pycache__").exists()


def test_a_streamed_model_is_not_moved_after_building(tiny_run, tmp_path, monkeypatch):
    """Its weights are placeholders until each module reads its own, and `.to()` on a
    placeholder is an error rather than a move."""
    from model_partition.hardware import move_to_device
    from model_partition.loop.stages import stage_trace

    ctx = _context(tiny_run, tmp_path)
    build = ctx.build_model

    def streamed(placed=False):
        model = build(placed=placed)
        ctx.last_placement = "streamed"
        return model

    ctx.build_model = streamed
    moved = []
    monkeypatch.setattr("model_partition.loop.stages.move_to_device",
                        lambda model, device: moved.append(device) or (model, device))
    stage_trace(ctx)
    assert not moved, "a streamed model must be left where the loader placed it"
    del move_to_device


# -- DeepSeek V4.1's hyper-connection patch ------------------------------------------


def _hyper_connections_path():
    import model_partition

    return (Path(model_partition.__file__).resolve().parents[1] / "compat"
            / "deepseek-v4.1-flash" / "hyper_connections.py")


def _sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """`kernel.hc_split_sinkhorn` in torch: the vendor's is TileLang and needs a GPU."""
    import torch

    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _identity_pre_mix(x, hc_mult):
    """`model.make_identity_pre_mix`, verbatim."""
    import torch

    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def _hyper_connections(monkeypatch):
    """The patch, imported as the loader imports it: beside the vendor's own modules."""
    import importlib.util
    import sys
    import types

    kernel = types.ModuleType("kernel")
    kernel.hc_split_sinkhorn = _sinkhorn
    model = types.ModuleType("model")
    model.make_identity_pre_mix = _identity_pre_mix
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    monkeypatch.setitem(sys.modules, "model", model)
    spec = importlib.util.spec_from_file_location("_hyper_connections_under_test",
                                                  _hyper_connections_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stand_in_vendor():
    """DeepSeek V4.1's `Block` and `Transformer`, cut down to what the patch touches.

    `hc_mixes`, `hc_pre`, `hc_post`, both `forward`s and `sample` are the vendor's text.
    Attention, the FFN, the norms and the embedding are stand-ins: the patch rewires
    around them and never looks inside. Built fresh per test, because the patch edits the
    classes it is given.
    """
    import types
    from dataclasses import dataclass

    import torch
    import torch.nn.functional as F
    from torch import nn

    hc_split_sinkhorn, make_identity_pre_mix = _sinkhorn, _identity_pre_mix

    @dataclass
    class ModelArgs:
        dim: int = 8
        vocab_size: int = 32
        n_layers: int = 3
        norm_eps: float = 1e-20
        hc_mult: int = 4
        hc_sinkhorn_iters: int = 20
        hc_eps: float = 1e-6

    class Sublayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x, *_):
            return torch.tanh(self.proj(x))

    def sample(logits, temperature: float = 1.0):
        if temperature == 0:
            return logits.argmax(dim=-1)
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

    class Block(nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            self.layer_id = layer_id
            self.norm_eps = args.norm_eps
            self.attn = Sublayer(args.dim)
            self.ffn = Sublayer(args.dim)
            self.engram = None
            self.attn_norm = nn.LayerNorm(args.dim)
            self.ffn_norm = nn.LayerNorm(args.dim)
            self.hc_mult = hc_mult = args.hc_mult
            self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
            self.hc_eps = args.hc_eps
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * args.dim
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
            x = x.flatten(2).float()
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            mixes = F.linear(x, hc_fn) * rsqrt
            return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult,
                                     self.hc_sinkhorn_iters, self.hc_eps)

        def hc_pre(self, x, pre_mix):
            y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
            return y.to(x.dtype)

        def hc_post(self, x, residual, post, comb):
            y = (post.unsqueeze(-1) * x.unsqueeze(-2)
                 + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
            return y.type_as(x)

        def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
            residual = x
            attn_pre, attn_post, attn_comb = self.hc_mixes(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            x = self.hc_pre(x, pre_mix)
            x = self.attn_norm(x)
            x = self.attn(x, start_pos, *attn_args)
            x = self.hc_post(x, residual, attn_post, attn_comb)

            residual = x
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
                x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
            x = self.hc_pre(x, attn_pre)
            x = self.ffn_norm(x)
            x = self.ffn(x, image_mask)
            x = self.hc_post(x, residual, ffn_post, ffn_comb)
            return x, ffn_pre

    class Transformer(nn.Module):
        def __init__(self, args, tokenizer=None):
            super().__init__()
            self.embed = nn.Embedding(args.vocab_size, args.dim)
            self.layers = nn.ModuleList(Block(i, args) for i in range(args.n_layers))
            self.norm = nn.LayerNorm(args.dim)
            self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
            self.hc_mult = args.hc_mult
            self.engram_hash = None
            self.target_layer_ids = [1]
            self.temperature = 0.0

        @torch.inference_mode()
        def forward(self, input_ids, start_pos=0, images=None, token_types=None):
            image_mask = None if token_types is None else token_types >= 0
            engram_mask = None if image_mask is None else ~image_mask
            engram_hashes = (self.engram_hash(input_ids, start_pos, engram_mask)
                             if self.engram_hash is not None else None)
            h = self.embed(input_ids)
            if images is not None:
                assert start_pos == 0, "image spans must be prefilled in a single chunk"
                self.merge_image_embeddings(images, h)
            h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
            main_hiddens = []
            pre_mix = make_identity_pre_mix(h, self.hc_mult)
            for i, layer in enumerate(self.layers):
                if layer.engram is not None:
                    h = layer.engram(
                        h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
                if i in self.target_layer_ids:
                    main_hiddens.append(h.mean(dim=2))
                h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
            h = layer.hc_pre(h, pre_mix)
            logits = self.head(self.norm(h))
            output_ids = sample(logits, self.temperature)
            main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
            return output_ids, logits, main_hidden

    return types.SimpleNamespace(ModelArgs=ModelArgs, Block=Block, Transformer=Transformer,
                                 sample=sample)


def test_the_mhc_patch_moves_the_mix_into_modules_and_changes_no_number(monkeypatch):
    """Every stage after the trace is measured against the patched model, so a patch that
    moved one number would move the reference for all of them. The checkpoint's names,
    renamed by the spec's own rules, load onto it strictly: nothing left over, nothing
    owned twice."""
    import torch

    from model_partition.cli import resolve_spec
    from model_partition.loaders.streamed import rename

    patch = _hyper_connections(monkeypatch)
    vendor = _stand_in_vendor()
    args = vendor.ModelArgs()
    torch.manual_seed(0)
    unpatched = vendor.Transformer(args)
    for parameter in unpatched.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    tokens = torch.randint(0, args.vocab_size, (2, 5))
    expected = unpatched(tokens)
    checkpoint = unpatched.state_dict()

    assert patch.apply(vendor, None) == patch.REPLACED
    patched = vendor.Transformer(args)
    rules = resolve_spec("deepseek-v4.1-flash").checkpoint.rename
    patched.load_state_dict({rename(key, rules): value for key, value in checkpoint.items()})

    for actual, reference in zip(patched(tokens), expected):
        assert torch.equal(actual, reference)
    block = patched.layers[0]
    assert isinstance(block.hc_attn_in, patch.HyperConnectSublayerIn)
    assert isinstance(block.hc_ffn_out, patch.HyperConnectSublayerOut)
    assert not [name for name, _ in block.named_parameters(recurse=False)]


def test_the_mhc_patch_applies_once_and_keeps_the_constructors_signatures(monkeypatch):
    """A run builds the model more than once and patches each time. Wrapping a wrapped
    `__init__` moved the coefficients twice, and a wrapper advertising only `*args` hid
    the `tokenizer` parameter the loader looks for."""
    import inspect

    patch = _hyper_connections(monkeypatch)
    vendor = _stand_in_vendor()
    patch.apply(vendor, None)
    patch.apply(vendor, None)

    vendor.Transformer(vendor.ModelArgs())
    assert "tokenizer" in inspect.signature(vendor.Transformer).parameters
    assert list(inspect.signature(vendor.Block).parameters) == ["layer_id", "args"]


def test_every_mhc_module_builds_from_its_config_and_weights_alone(monkeypatch):
    """Which is how every stage after the trace runs one. The first version took its
    parameters as constructor arguments, and the modules that own none could not be
    built from nothing: 522 of 822 module checks failed without running."""
    import torch

    from model_partition.runtime.launcher import _construct, load_weights

    patch = _hyper_connections(monkeypatch)
    config = {"dim": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
              "norm_eps": 1e-20}
    recorded = {"layers.0.hc_attn_in.fn": torch.randn(24, 32),
                "layers.0.hc_attn_in.base": torch.randn(24),
                "layers.0.hc_attn_in.scale": torch.randn(3)}

    mix_in = _construct(patch.HyperConnectSublayerIn, config, 0, recorded, config)
    assert (mix_in.hc_mult, mix_in.hc_sinkhorn_iters, mix_in.hc_eps, mix_in.norm_eps) \
        == (4, 20, 1e-6, 1e-20)
    loaded, missing = load_weights(mix_in, recorded)
    assert loaded == 3 and not missing
    assert mix_in.fn.dtype is torch.float32
    assert torch.equal(mix_in.fn, recorded["layers.0.hc_attn_in.fn"])

    for cls in (patch.HyperConnectSublayerOut, patch.HyperConnectCollapse):
        assert isinstance(_construct(cls, config, 0, {}, config), cls)
    assert _construct(patch.HyperConnectExpand, config, None, {}, config).hc_mult == 4
