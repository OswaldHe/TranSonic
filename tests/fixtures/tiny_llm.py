# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tiny but real causal LM used as a local model repo in tests.

Deliberately mirrors the vendor-code layout the loop prefers: the generated
directory holds ``config.json``, ``model.safetensors`` and an
``inference/model.py`` exposing ``build_model(config, state_dict)``. Parameter
names follow HuggingFace conventions so layer and expert regexes apply.

``n_experts > 0`` adds a MoE block to odd layers, giving tests a hybrid stack
with two distinct layer signatures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

#: Written into each generated repo as ``inference/model.py``. ``__REPO_ROOT__``
#: is substituted with an absolute path so the generated repo works from any
#: working directory. (Plain replacement, not str.format — the body has braces.)
VENDOR_CODE = '''\
"""Reference implementation for the tiny test model."""

import sys

sys.path.insert(0, "__REPO_ROOT__")

from tests.fixtures.tiny_llm import TinyCausalLM, TinyConfig


def build_model(config, state_dict=None):
    model = TinyCausalLM(TinyConfig(**{
        k: v for k, v in config.items() if k in TinyConfig.__dataclass_fields__
    }))
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.eval()
'''


@dataclass
class TinyConfig:
    """Config for :class:`TinyCausalLM`, shaped like an HF config."""

    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rms_norm_eps: float = 1e-6
    n_experts: int = 0
    num_experts_per_tok: int = 2
    tie_word_embeddings: bool = False
    layer_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_type": "tiny_llm",
            "architectures": ["TinyCausalLM"],
            "dtype": "float32",
            **{f: getattr(self, f) for f in self.__dataclass_fields__},
        }


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class TinyAttention(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyMoE(nn.Module):
    """Top-k routed experts, shaped like a real MoE block."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyMLP(config.hidden_size, config.intermediate_size) for _ in range(config.n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for expert_id, expert in enumerate(self.experts):
                mask = indices[..., slot] == expert_id
                if mask.any():
                    out[mask] += expert(x[mask]) * weights[..., slot][mask].unsqueeze(-1)
        return out

    def routing(self, x: torch.Tensor) -> torch.Tensor:
        """Expert ids selected per token, for trace records."""
        return torch.topk(self.gate(x), self.top_k, dim=-1).indices


class TinyDecoderLayer(nn.Module):
    def __init__(self, config: TinyConfig, use_moe: bool):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TinyAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TinyMoE(config) if use_moe else TinyMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyDecoderLayer(config, use_moe=bool(config.n_experts) and i % 2 == 1)
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class TinyCausalLM(nn.Module):
    """Whole model: ``model.*`` plus ``lm_head``, as HF checkpoints are laid out."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        self.model = TinyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids))


def build_tiny_model(config: TinyConfig | None = None, seed: int = 0) -> TinyCausalLM:
    """Deterministically initialized model in eval mode."""
    torch.manual_seed(seed)
    model = TinyCausalLM(config or TinyConfig())
    for param in model.parameters():
        nn.init.normal_(param, std=0.05)
    return model.eval()


def write_tiny_repo(
    directory: str | Path,
    config: TinyConfig | None = None,
    seed: int = 0,
    with_vendor_code: bool = True,
) -> Path:
    """Materialize a local model repo and return its path."""
    from safetensors.torch import save_file

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    config = config or TinyConfig()
    model = build_tiny_model(config, seed=seed)

    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    if with_vendor_code:
        inference = root / "inference"
        inference.mkdir(exist_ok=True)
        project_root = Path(__file__).resolve().parents[2]
        (inference / "model.py").write_text(
            VENDOR_CODE.replace("__REPO_ROOT__", str(project_root))
        )
    return root


def sample_inputs(n: int = 3, seq_len: int = 8, vocab_size: int = 128, seed: int = 1) -> torch.Tensor:
    """Deterministic token-id batch for tracing."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n, seq_len), generator=generator)
