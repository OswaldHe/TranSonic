# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-loop retention: keep representative layers, prune the rest."""

import pytest
import yaml

from model_partition.retention import (
    RetentionPolicy,
    apply_retention,
    plan_retention,
    write_retention_report,
)
from model_partition.sizing import ModelInventory
from model_partition.weights_index import TensorEntry, WeightIndex

pytest.importorskip("torch")


def inventory_for(n_layers: int, signature_of=lambda i: "attn") -> ModelInventory:
    entries = [TensorEntry(f"model.layers.{i}.self_attn.{signature_of(i)}.weight",
                           "bfloat16", (8, 8), 128, "s0") for i in range(n_layers)]
    return ModelInventory.build(WeightIndex(entries=entries),
                                {"hidden_size": 8, "num_hidden_layers": n_layers})


# -- layer selection ---------------------------------------------------------


def test_default_policy_keeps_first_1_5_mid_and_last():
    keep = RetentionPolicy().layers_to_keep(inventory_for(64))
    assert keep == [0, 1, 5, 32, 63]


def test_preferred_layers_are_configurable():
    keep = RetentionPolicy(preferred_layers=(3, 7)).layers_to_keep(inventory_for(32))
    assert {3, 7} <= set(keep)


def test_out_of_range_preferred_layers_are_ignored():
    keep = RetentionPolicy(preferred_layers=(1, 500)).layers_to_keep(inventory_for(8))
    assert 500 not in keep
    assert max(keep) == 7


def test_every_signature_keeps_a_representative():
    """A hybrid stack must not lose the only copy of a kernel variant."""
    inventory = inventory_for(24, lambda i: "full" if i % 4 == 3 else "linear")
    keep = RetentionPolicy().layers_to_keep(inventory)
    kept_signatures = {inventory.layers[i].signature for i in keep}
    assert kept_signatures == set(inventory.signature_groups())


def test_signature_coverage_can_be_disabled():
    inventory = inventory_for(24, lambda i: f"variant{i}")
    with_coverage = RetentionPolicy().layers_to_keep(inventory)
    without = RetentionPolicy(keep_all_signatures=False).layers_to_keep(inventory)
    assert len(with_coverage) > len(without)


def test_empty_stack_keeps_nothing():
    assert RetentionPolicy().layers_to_keep(inventory_for(0)) == []


def test_single_layer_stack():
    assert RetentionPolicy().layers_to_keep(inventory_for(1)) == [0]


# -- planning ----------------------------------------------------------------


def test_plan_keeps_globals_and_representative_layers(tiny_deep_run):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    assert plan.kept_layers == [0, 1, 5, 6, 11]
    # embed, final_norm and lm_head have no layer index and always survive.
    assert {"embed", "final_norm", "lm_head"} <= set(plan.kept_modules)
    assert plan.dropped_modules
    assert plan.bytes_dropped > 0


def test_plan_accounts_for_every_byte(tiny_deep_run):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    total = sum(e.nbytes for e in run.bundle.store.entries)
    assert plan.bytes_kept + plan.bytes_dropped == total


def test_plan_never_empties_a_signature_group(tiny_deep_run):
    run = tiny_deep_run
    policy = RetentionPolicy(preferred_layers=(), keep_first=False,
                             keep_mid=False, keep_last=False)
    plan = plan_retention(run.graph, run.inventory, run.bundle, policy)
    for module_ids in run.graph.signature_groups().values():
        assert set(module_ids) & set(plan.kept_modules), module_ids


def test_small_model_keeps_everything(tiny_run):
    """A 4-layer stack is entirely representative, so nothing is dropped."""
    plan = plan_retention(tiny_run.graph, tiny_run.inventory, tiny_run.bundle)
    assert plan.dropped_modules == []
    assert plan.bytes_dropped == 0


def test_plan_renders_and_serializes(tiny_deep_run):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    text = plan.render()
    assert "kept layers" in text and "reduction" in text
    assert plan.to_dict()["kept_layers"] == plan.kept_layers


# -- applying ----------------------------------------------------------------


def test_dry_run_changes_nothing_on_disk(tiny_deep_run):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    before = sorted(p.name for p in run.bundle.root.rglob("*.bin"))
    result = apply_retention(run.bundle, plan, dry_run=True)
    after = sorted(p.name for p in run.bundle.root.rglob("*.bin"))
    assert before == after
    assert result.dry_run and result.removed_files > 0
    assert "would remove" in result.summary()


def test_apply_removes_dropped_artifacts(tiny_deep_run):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    result = apply_retention(run.bundle, plan)
    assert result.removed_files > 0
    assert result.bytes_freed > 0
    assert not result.dry_run
    for module_id in plan.dropped_modules:
        assert run.bundle.store.find(module_id=module_id) == []


def test_kept_modules_remain_verifiable_after_pruning(tiny_deep_run):
    """Pruning must not break the modules it keeps."""
    from model_partition.verify.modules import verify_modules

    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    apply_retention(run.bundle, plan)
    report = verify_modules(run.build_model, run.bundle, run.graph,
                            module_ids=plan.kept_modules)
    assert report.passed, report.render()


def test_manifest_and_records_are_rewritten(tiny_deep_run):
    from model_partition.runtime.module_runner import TraceBundle

    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    apply_retention(run.bundle, plan)
    reloaded = TraceBundle.load(run.bundle.root)
    assert set(reloaded.module_ids()) <= set(plan.kept_modules)
    assert set(reloaded.weights) <= set(plan.kept_modules)
    assert reloaded.metadata["retention"]["kept_layers"] == plan.kept_layers


def test_surviving_blobs_still_verify_after_pruning(tiny_deep_run):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    apply_retention(run.bundle, plan)
    for entry in run.bundle.store.entries:
        run.bundle.store.verify(entry)


def test_shared_blob_is_not_double_counted(tiny_deep_run):
    """Hardlinked blobs only free bytes when the last reference goes."""
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    result = apply_retention(run.bundle, plan)
    assert result.bytes_freed <= plan.bytes_dropped


def test_report_is_written(tiny_deep_run, tmp_path):
    run = tiny_deep_run
    plan = plan_retention(run.graph, run.inventory, run.bundle)
    result = apply_retention(run.bundle, plan, dry_run=True)
    path = write_retention_report(tmp_path / "retention.yaml", result)
    payload = yaml.safe_load(path.read_text())
    assert payload["dry_run"] is True
    assert payload["kept_layers"] == plan.kept_layers
    assert payload["removed_files"] == result.removed_files


def test_shared_blobs_are_counted_once(tmp_path):
    """Hardlinked duplicates release one blob's bytes, not one per name."""
    import numpy as np

    from model_partition.retention import RetentionPlan
    from model_partition.runtime.module_runner import TraceBundle
    from model_partition.tensorstore import TensorStore

    store = TensorStore(tmp_path)
    payload = np.zeros((64,), dtype=np.float32)
    for name in ("a", "b", "c"):
        store.write(name, payload, role="weight", module_id="layers.9")
    nbytes = store.entries[0].nbytes
    assert len({e.sha256 for e in store.entries}) == 1

    bundle = TraceBundle(store=store)
    bundle.save()
    plan = RetentionPlan(kept_layers=[], kept_modules=[], dropped_modules=["layers.9"])
    result = apply_retention(bundle, plan, dry_run=True)

    assert result.removed_files == 3
    assert result.bytes_freed == nbytes
