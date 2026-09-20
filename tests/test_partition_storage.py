# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dump policy, storage estimate, and preflight."""

import pytest

from model_partition.hardware import GIB, HostInfo, format_bytes
from model_partition.storage import (
    DumpPolicy,
    StoragePreflightError,
    TraceShape,
    estimate_storage,
    preflight,
)


def host(free_gib: float) -> HostInfo:
    from pathlib import Path
    return HostInfo(
        cpu_count=16, ram_total_bytes=124 * GIB, ram_available_bytes=120 * GIB,
        disk_total_bytes=484 * GIB, disk_free_bytes=int(free_gib * GIB),
        artifact_root=Path("/tmp/artifacts"),
    )


def qwen27b_trace() -> TraceShape:
    """Qwen3.8-27B shape: hidden 5120, bf16, 20 modules."""
    return TraceShape(
        n_modules=20,
        boundary_bytes_per_token=5120 * 2,
        short_seq_lens=[128, 128, 128, 128],
        long_seq_lens=[2048, 8192, 16384, 16384],
    )


def test_short_samples_are_never_sliced():
    policy = DumpPolicy()
    assert policy.kept_positions(128) == 128
    assert policy.kept_positions(1023) == 1023


def test_long_samples_are_sliced_to_head_plus_tail():
    policy = DumpPolicy()
    assert policy.kept_positions(16384) == 256
    assert policy.kept_positions(2048) == 256


def test_slice_never_exceeds_sequence_length():
    policy = DumpPolicy()
    assert policy.kept_positions(1100) == 256
    assert DumpPolicy(slice_head=1024, slice_tail=1024).kept_positions(1100) == 1100


def test_full_dumps_disable_slicing():
    assert DumpPolicy(full_dumps=True).kept_positions(16384) == 16384


def test_slicing_dominates_the_long_context_estimate():
    """The headline reason slicing exists: full 16k dumps are ~26x larger."""
    trace = qwen27b_trace()
    sliced = estimate_storage(
        checkpoint_bytes=0, module_weight_bytes=0, dequant_bytes=0,
        trace=trace, policy=DumpPolicy(),
    )
    full = estimate_storage(
        checkpoint_bytes=0, module_weight_bytes=0, dequant_bytes=0,
        trace=trace, policy=DumpPolicy(full_dumps=True),
    )
    assert full.total_bytes > 20 * sliced.total_bytes


def test_estimate_lines_follow_the_cache_flags():
    trace = qwen27b_trace()
    est = estimate_storage(
        checkpoint_bytes=56 * GIB, module_weight_bytes=56 * GIB, dequant_bytes=56 * GIB,
        trace=trace, policy=DumpPolicy(cache_weights=False, cache_dequant=False),
    )
    labels = [line.label for line in est.lines]
    assert "module weight dumps" not in labels
    assert "bf16 dequant mirror" not in labels
    assert "checkpoint" in labels


def test_cache_flags_enabled_add_both_lines():
    est = estimate_storage(
        checkpoint_bytes=GIB, module_weight_bytes=GIB, dequant_bytes=2 * GIB,
        trace=qwen27b_trace(), policy=DumpPolicy(),
    )
    labels = [line.label for line in est.lines]
    assert "module weight dumps" in labels
    assert "bf16 dequant mirror" in labels
    assert est.total_bytes >= 4 * GIB


def test_streamed_checkpoint_is_labelled_as_peak():
    est = estimate_storage(
        checkpoint_bytes=16 * GIB, module_weight_bytes=0, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(), stream_checkpoint=True,
    )
    assert any("streamed, peak" in line.label for line in est.lines)


def test_decode_steps_are_counted_once_not_per_sample():
    trace = qwen27b_trace()
    with_decode = estimate_storage(
        checkpoint_bytes=0, module_weight_bytes=0, dequant_bytes=0,
        trace=trace, policy=DumpPolicy(decode_steps=4),
    )
    without = estimate_storage(
        checkpoint_bytes=0, module_weight_bytes=0, dequant_bytes=0,
        trace=trace, policy=DumpPolicy(decode_steps=0),
    )
    delta = with_decode.total_bytes - without.total_bytes
    assert delta == 2 * trace.n_modules * 4 * trace.boundary_bytes_per_token


def test_preflight_passes_with_ample_disk():
    est = estimate_storage(
        checkpoint_bytes=GIB, module_weight_bytes=GIB, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(), host=host(400),
    )
    assert est.fits
    assert preflight(est) == []


def test_preflight_raises_when_projection_exceeds_disk():
    """The DeepSeek V4.1 case: 765 GB projected against 465 GB free."""
    est = estimate_storage(
        checkpoint_bytes=765 * GIB, module_weight_bytes=0, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(), host=host(465),
    )
    assert not est.fits
    with pytest.raises(StoragePreflightError) as excinfo:
        preflight(est)
    message = str(excinfo.value)
    assert "cache_dequant=false" in message
    assert "765.0 GiB" in message


def test_preflight_non_strict_returns_warning_instead():
    est = estimate_storage(
        checkpoint_bytes=765 * GIB, module_weight_bytes=0, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(), host=host(465),
    )
    warnings = preflight(est, strict=False)
    assert len(warnings) == 1 and "exceeds usable disk" in warnings[0]


def test_preflight_warns_near_the_limit():
    est = estimate_storage(
        checkpoint_bytes=90 * GIB, module_weight_bytes=0, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(), host=host(100),
    )
    warnings = preflight(est)
    assert any("over 80%" in w for w in warnings)


def test_preflight_skips_when_disk_is_unknown():
    est = estimate_storage(
        checkpoint_bytes=GIB, module_weight_bytes=0, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(),
    )
    assert "skipping storage preflight" in preflight(est)[0]


def test_render_includes_total_and_verdict():
    est = estimate_storage(
        checkpoint_bytes=GIB, module_weight_bytes=0, dequant_bytes=0,
        trace=qwen27b_trace(), policy=DumpPolicy(), host=host(400),
    )
    rendered = est.render()
    assert "TOTAL" in rendered and "disk free" in rendered and "(fits)" in rendered


def test_format_bytes_is_base_1024():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(55.6 * GIB).startswith("55.6 GiB")
