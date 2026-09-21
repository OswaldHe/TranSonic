# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the binary tensor dump format."""

import numpy as np
import pytest

from model_partition.tensorstore import (
    SliceInfo,
    TensorMeta,
    TensorStore,
    TensorStoreError,
    dtype_name,
)

torch = pytest.importorskip("torch")


def test_numpy_round_trip(tmp_path):
    store = TensorStore(tmp_path)
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    meta = store.write("x", array, role="weight", module_id="layers.0")
    assert meta.dtype == "float32"
    assert meta.shape == [2, 3, 4]
    assert meta.nbytes == 96
    np.testing.assert_array_equal(store.read_numpy(meta), array)
    store.verify(meta)


def test_torch_bfloat16_round_trip(tmp_path):
    """bfloat16 has no numpy dtype, so it must survive as raw bytes."""
    store = TensorStore(tmp_path)
    tensor = torch.randn(4, 8, dtype=torch.bfloat16)
    meta = store.write("h", tensor)
    assert meta.dtype == "bfloat16"
    assert meta.nbytes == 4 * 8 * 2
    assert torch.equal(store.read_torch(meta), tensor)


def test_torch_fp8_round_trip(tmp_path):
    store = TensorStore(tmp_path)
    tensor = torch.randn(16, 16).to(torch.float8_e4m3fn)
    meta = store.write("w_fp8", tensor, role="weight")
    assert meta.dtype == "float8_e4m3fn"
    restored = store.read_torch(meta)
    assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))


def test_non_contiguous_tensor_is_made_contiguous(tmp_path):
    store = TensorStore(tmp_path)
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    meta = store.write("t", tensor)
    assert meta.shape == [4, 3]
    assert torch.equal(store.read_torch(meta), tensor)


def test_sidecar_json_written_next_to_blob(tmp_path):
    store = TensorStore(tmp_path)
    meta = store.write("x", np.zeros(4, dtype=np.int32), subdir="layers.0")
    blob = store.blob_path(meta)
    assert blob.name == "x.bin"
    assert blob.parent.name == "layers.0"
    assert blob.with_suffix(".json").is_file()


def test_identical_blobs_are_deduplicated(tmp_path):
    """Tied embeddings and repeated layers must not double the footprint."""
    store = TensorStore(tmp_path, dedupe=True)
    array = np.ones(1024, dtype=np.float32)
    first = store.write("embed", array, subdir="a")
    second = store.write("lm_head", array, subdir="b")
    assert first.sha256 == second.sha256
    assert store.blob_path(first).stat().st_ino == store.blob_path(second).stat().st_ino


def test_distinct_blobs_are_not_linked(tmp_path):
    store = TensorStore(tmp_path)
    a = store.write("a", np.ones(8, dtype=np.float32))
    b = store.write("b", np.zeros(8, dtype=np.float32))
    assert a.sha256 != b.sha256
    assert store.blob_path(a).stat().st_ino != store.blob_path(b).stat().st_ino


def test_verify_detects_truncation(tmp_path):
    store = TensorStore(tmp_path)
    meta = store.write("x", np.arange(16, dtype=np.float32))
    store.blob_path(meta).write_bytes(b"\x00" * 8)
    with pytest.raises(TensorStoreError, match="expected 64 bytes, found 8"):
        store.verify(meta)


def test_verify_detects_bit_corruption(tmp_path):
    store = TensorStore(tmp_path)
    meta = store.write("x", np.arange(16, dtype=np.float32))
    raw = bytearray(store.blob_path(meta).read_bytes())
    raw[0] ^= 0xFF
    store.blob_path(meta).write_bytes(bytes(raw))
    with pytest.raises(TensorStoreError, match="sha256 mismatch"):
        store.verify(meta)


def test_verify_detects_missing_blob(tmp_path):
    store = TensorStore(tmp_path)
    meta = store.write("x", np.zeros(2, dtype=np.float32))
    store.blob_path(meta).unlink()
    with pytest.raises(TensorStoreError, match="Missing blob"):
        store.verify(meta)


def test_manifest_round_trip_and_find(tmp_path):
    store = TensorStore(tmp_path)
    store.write("in", np.zeros(4, dtype=np.float32), role="activation", module_id="m0", sample_id="s0")
    store.write("out", np.ones(4, dtype=np.float32), role="output", module_id="m0", sample_id="s0")
    store.write("w", np.ones(8, dtype=np.float32), role="weight", module_id="m0")
    store.save_manifest({"model": "toy"})

    reloaded = TensorStore.load(tmp_path)
    assert reloaded.metadata == {"model": "toy"}
    assert len(reloaded.entries) == 3
    assert [m.name for m in reloaded.find(role="weight")] == ["w"]
    assert [m.name for m in reloaded.find(module_id="m0", sample_id="s0")] == ["in", "out"]
    for meta in reloaded.entries:
        reloaded.verify(meta)


def test_manifest_records_dedup_savings(tmp_path):
    store = TensorStore(tmp_path)
    array = np.ones(256, dtype=np.float32)
    store.write("a", array, subdir="x")
    store.write("b", array, subdir="y")
    import yaml
    payload = yaml.safe_load(store.save_manifest().read_text())
    assert payload["unique_blobs"] == 1
    assert payload["total_bytes"] == 2048


def test_slice_info_is_recorded(tmp_path):
    """Long-context dumps must say they are partial."""
    store = TensorStore(tmp_path)
    meta = store.write(
        "h", np.zeros((256, 8), dtype=np.float32),
        slice_info=SliceInfo(axes=[0], head=128, tail=128, original_length=16384),
    )
    expected = {"axes": [0], "head": 128, "tail": 128, "original_length": 16384}
    assert meta.slice_info == expected
    store.save_manifest()
    assert TensorStore.load(tmp_path).entries[0].slice_info == expected


def test_load_without_manifest_raises(tmp_path):
    with pytest.raises(TensorStoreError, match="No manifest"):
        TensorStore.load(tmp_path)


def test_dtype_name_normalizes_torch_and_numpy():
    assert dtype_name(torch.bfloat16) == "bfloat16"
    assert dtype_name(torch.float32) == "float32"
    assert dtype_name(np.dtype("int64")) == "int64"
    with pytest.raises(TensorStoreError, match="Unsupported dtype"):
        dtype_name("complex128")


def test_element_count():
    meta = TensorMeta(name="x", dtype="float32", shape=[2, 3, 4], nbytes=96, sha256="", path="x.bin")
    assert meta.element_count == 24
