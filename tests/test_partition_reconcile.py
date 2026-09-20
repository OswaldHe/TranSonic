# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reconciling plan submodule names against the real module tree."""

import pytest

from model_partition.planner.graph import ModuleNode, PartitionGraph, TensorRef
from model_partition.planner.reconcile import (
    build_suffix_index,
    reconcile_submodules,
    resolve_name,
)

torch = pytest.importorskip("torch")


def names_of(model):
    return [name for name, _ in model.named_modules() if name]


class FakeModel:
    """Minimal stand-in exposing named_modules()."""

    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return [(name, object()) for name in self._names]


def graph_with(submodules: list[str]) -> PartitionGraph:
    return PartitionGraph(
        model="m",
        modules=[ModuleNode(id="m0", kind="decoder_layers", inputs=["a"], outputs=["b"],
                            submodules=list(submodules))],
        tensors={"a": TensorRef(name="a"), "b": TensorRef(name="b")},
        entry_tensors=["a"], output_tensors=["b"],
    )


# -- suffix resolution -------------------------------------------------------


def test_exact_match_wins():
    known = {"model.layers.0"}
    index = build_suffix_index(list(known))
    assert resolve_name("model.layers.0", known, index) == ("model.layers.0", [])


def test_checkpoint_prefix_is_stripped_by_suffix_match():
    """Regression: checkpoints name model.language_model.layers.N while the
    instantiated model exposes model.layers.N."""
    names = ["model.layers.0", "model.layers.1", "lm_head"]
    resolved, ambiguous = resolve_name("model.language_model.layers.0", set(names),
                                       build_suffix_index(names))
    assert resolved == "model.layers.0"
    assert ambiguous == []


def test_extra_prefix_in_the_model_also_resolves():
    names = ["model.language_model.layers.0", "model.visual.blocks.0"]
    resolved, _ = resolve_name("model.layers.0", set(names), build_suffix_index(names))
    assert resolved == "model.language_model.layers.0"


def test_ambiguity_is_broken_toward_the_shared_prefix_then_the_shorter_name():
    """Regression: model.language_model.norm resolved to nothing because many
    modules end in `.norm`."""
    names = [
        "model.norm",
        "model.layers.0.linear_attn.norm",
        "model.layers.1.linear_attn.norm",
    ]
    resolved, ambiguous = resolve_name("model.language_model.norm", set(names),
                                       build_suffix_index(names))
    assert resolved == "model.norm"
    assert len(ambiguous) == 3


def test_truly_absent_name_is_unresolved():
    names = ["model.layers.0"]
    resolved, ambiguous = resolve_name("mtp.pre_fc_norm_hidden", set(names),
                                       build_suffix_index(names))
    assert resolved is None and ambiguous == []


def test_suffix_index_covers_every_suffix():
    index = build_suffix_index(["a.b.c"])
    assert set(index) == {"a.b.c", "b.c", "c"}
    assert index["c"] == ["a.b.c"]


# -- graph reconciliation ----------------------------------------------------


def test_reconcile_rewrites_the_graph_in_place():
    graph = graph_with(["model.language_model.layers.0", "model.language_model.layers.1"])
    model = FakeModel(["model", "model.layers", "model.layers.0", "model.layers.1"])
    report = reconcile_submodules(graph, model)
    assert graph.modules[0].submodules == ["model.layers.0", "model.layers.1"]
    assert report.changed
    assert report.unresolved == []
    assert "2 submodule name(s) remapped" in report.summary()


def test_reconcile_is_idempotent():
    graph = graph_with(["model.language_model.layers.0"])
    model = FakeModel(["model.layers.0"])
    reconcile_submodules(graph, model)
    second = reconcile_submodules(graph, model)
    assert graph.modules[0].submodules == ["model.layers.0"]
    assert not second.changed


def test_unresolved_names_are_kept_and_reported():
    graph = graph_with(["model.layers.0", "mtp.norm"])
    model = FakeModel(["model.layers.0"])
    report = reconcile_submodules(graph, model)
    assert report.unresolved == ["mtp.norm"]
    assert "mtp.norm" in graph.modules[0].submodules
    assert "1 unresolved" in report.summary()


def test_duplicates_introduced_by_remapping_are_collapsed():
    """Two checkpoint paths can map onto one module; the plan must not list it twice."""
    graph = graph_with(["a.model.layers.0", "b.model.layers.0"])
    model = FakeModel(["model.layers.0"])
    reconcile_submodules(graph, model)
    assert graph.modules[0].submodules == ["model.layers.0"]


def test_order_is_preserved():
    graph = graph_with(["x.layers.2", "x.layers.0", "x.layers.1"])
    model = FakeModel(["layers.0", "layers.1", "layers.2"])
    reconcile_submodules(graph, model)
    assert graph.modules[0].submodules == ["layers.2", "layers.0", "layers.1"]


def test_unpartitioned_modules_are_left_alone():
    graph = graph_with(["model.layers.0"])
    graph.modules.append(ModuleNode(id="vision", kind="vision",
                                    submodules=["model.visual.blocks.0"], partitioned=False))
    reconcile_submodules(graph, FakeModel(["model.layers.0"]))
    assert graph.by_id("vision").submodules == ["model.visual.blocks.0"]


def test_reconcile_against_a_real_model_resolves_everything(tiny_run):
    report = reconcile_submodules(tiny_run.graph, tiny_run.build_model())
    assert report.unresolved == []


def test_reconcile_recovers_a_corrupted_prefix(tiny_run):
    """A plan written against a differently-prefixed checkpoint still traces."""
    from model_partition.trace import Tracer

    graph = tiny_run.graph
    for module in graph.partitioned_modules:
        module.submodules = [f"wrapper.{name}" for name in module.submodules]
    model = tiny_run.build_model()
    assert Tracer(model, graph, tiny_run.bundle.store).unresolved_submodules()

    reconcile_submodules(graph, model)
    assert Tracer(model, graph, tiny_run.bundle.store).unresolved_submodules() == []
