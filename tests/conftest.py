# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the model_partition tests.

``tiny_run`` builds a complete partition run over the toy model — plan, trace,
weight dumps — so verification, emulation, extraction and retention tests all
work against real artifacts rather than mocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass
class TinyRun:
    """A toy model with a plan and a populated trace bundle."""

    root: Path
    repo: Path
    spec: Any
    result: Any
    graph: Any
    bundle: Any
    inventory: Any
    build_model: Any
    sample_ids: list[str]
    layout: Any


def _build(tmp_path: Path, n_experts: int, n_samples: int, seq_len: int, n_layers: int,
           one_layer_per_module: bool = False) -> TinyRun:
    from safetensors.torch import load_file

    from model_partition.ingest import ingest
    from model_partition.layout import RunLayout
    from model_partition.loaders import build_loader
    from model_partition.planner.auto import PlanOptions, plan
    from model_partition.runtime.module_runner import TraceBundle
    from model_partition.sizing import ModelInventory
    from model_partition.spec import parse_spec
    from model_partition.tensorstore import TensorStore
    from model_partition.trace import Tracer
    from model_partition.weights_index import WeightIndex
    from tests.fixtures.tiny_llm import TinyConfig, sample_inputs, write_tiny_repo

    repo = write_tiny_repo(
        tmp_path / "repo",
        TinyConfig(n_experts=n_experts, num_hidden_layers=n_layers),
    )
    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(
        json.dumps({"id": f"s{i}", "prompt": "alpha beta gamma delta " * 3,
                    "role": "raw", "target_tokens": seq_len})
        for i in range(n_samples)
    ))
    spec = parse_spec({
        "source": str(repo), "name": "tiny", "entry": "inference/model.py",
        "loader": "repo_code", "code_paths": ["inference"],
        "trust_remote_code": True, "dtype": "float32",
        "inputs": {"short": str(short)},
    })
    result = ingest(spec)
    config = json.loads((repo / "config.json").read_text())
    inventory = ModelInventory.build(WeightIndex.from_local(repo), config)
    graph = plan(
        inventory, 10 ** 9,
        PlanOptions(seq_len=seq_len, one_layer_per_module=one_layer_per_module),
        model_name="tiny",
    )

    state_dict = load_file(str(repo / "model.safetensors"))
    loader = build_loader(result)

    def build_model():
        return loader.build(dict(state_dict)).model

    layout = RunLayout.create("tiny", tmp_path / "artifacts").ensure()
    graph.save(layout.graph_path)

    store = TensorStore(layout.trace_dir)
    tracer = Tracer(build_model(), graph, store)
    weights = tracer.dump_weights()
    ids = sample_inputs(n_samples, seq_len)
    sample_ids = []
    for index in range(n_samples):
        sample_id = f"s{index}"
        tracer.trace_sample(sample_id, ids[index:index + 1])
        sample_ids.append(sample_id)

    bundle = TraceBundle(store=store, records=tracer.records, weights=weights,
                         metadata={"model": "tiny"})
    bundle.save()
    layout.write_run({"spec": spec.to_dict(), "revision": None, "loader": result.loader})

    return TinyRun(root=tmp_path, repo=repo, spec=spec, result=result, graph=graph,
                   bundle=TraceBundle.load(layout.trace_dir), inventory=inventory,
                   build_model=build_model, sample_ids=sample_ids, layout=layout)


@pytest.fixture
def tiny_run(tmp_path):
    """Dense 4-layer toy model, 2 samples of 8 tokens."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    return _build(tmp_path, n_experts=0, n_samples=2, seq_len=8, n_layers=4)


@pytest.fixture
def tiny_moe_run(tmp_path):
    """Toy model with MoE on odd layers: two distinct layer signatures."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    return _build(tmp_path, n_experts=4, n_samples=1, seq_len=8, n_layers=4)


@pytest.fixture
def tiny_deep_run(tmp_path):
    """12-layer toy model, one module per layer.

    Retention prunes at module granularity, so pruning is only observable when
    layers occupy separate modules.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    return _build(tmp_path, n_experts=0, n_samples=1, seq_len=8, n_layers=12,
                  one_layer_per_module=True)
