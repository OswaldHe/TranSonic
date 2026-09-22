# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage estimation, preflight, and the dump policy.

Feature maps dominate the footprint at long context, so short samples are dumped
in full and long ones sliced. :func:`preflight` runs before any work so a run
that cannot fit fails with arithmetic rather than ENOSPC mid-trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_partition.hardware import GIB, HostInfo, format_bytes

#: Positions kept at each end of a sliced long-sample feature map. Two 128-token
#: windows keep the prefix/suffix behaviour kernel bring-up needs (RoPE phase,
#: sliding-window edges) for a fraction of a full 16k dump.
DEFAULT_SLICE_HEAD = 128
DEFAULT_SLICE_TAIL = 128

#: A sample is "long" at or above this many tokens.
LONG_THRESHOLD_TOKENS = 1024


@dataclass
class DumpPolicy:
    """How much tensor data tracing is allowed to write."""

    #: Cut long-sample activations down to a head/tail window. Off by default,
    #: and deliberately: attention mixes every position, so a module's sliced
    #: output is *not* a function of its sliced input. Sliced records are kept for
    #: inspection and excluded from numeric verification, which is a real loss of
    #: coverage — worth it only when disk genuinely cannot hold the full maps.
    slice_long: bool = False
    slice_head: int = DEFAULT_SLICE_HEAD
    slice_tail: int = DEFAULT_SLICE_TAIL
    #: Persist per-module weight dumps. ``False`` keeps only the index of which
    #: parameters each module owns and reads their values from the checkpoint on
    #: demand, so verification still runs but nothing large is written.
    cache_weights: bool = True
    #: Persist the bf16 dequant mirror; ``False`` recomputes it per load.
    cache_dequant: bool = True
    #: Decode steps traced beyond prefill, first sample only.
    decode_steps: int = 4
    #: Hard ceiling; tracing aborts rather than truncating silently.
    max_total_bytes: int | None = None

    def is_long(self, seq_len: int) -> bool:
        return seq_len >= LONG_THRESHOLD_TOKENS

    def slices(self, seq_len: int) -> bool:
        """True when a sample of this length gets a windowed dump."""
        return self.slice_long and self.is_long(seq_len)

    def kept_positions(self, seq_len: int) -> int:
        """Sequence positions actually dumped for a tensor of length ``seq_len``."""
        if not self.slices(seq_len):
            return seq_len
        return min(seq_len, self.slice_head + self.slice_tail)


@dataclass
class EstimateLine:
    """One row of the storage estimate."""

    label: str
    bytes_: int
    detail: str = ""


@dataclass
class StorageEstimate:
    """Projected on-disk footprint of a full partition run."""

    lines: list[EstimateLine] = field(default_factory=list)
    disk_free_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(line.bytes_ for line in self.lines)

    @property
    def fits(self) -> bool:
        return self.disk_free_bytes <= 0 or self.total_bytes <= self.disk_free_bytes

    def add(self, label: str, nbytes: int, detail: str = "") -> None:
        if nbytes > 0:
            self.lines.append(EstimateLine(label, int(nbytes), detail))

    def render(self) -> str:
        """Format the estimate as a plain-text table for the loop preamble."""
        width = max([len(line.label) for line in self.lines] + [len("TOTAL")])
        rows = [f"  {line.label.ljust(width)}  {format_bytes(line.bytes_).rjust(10)}"
                f"{'  ' + line.detail if line.detail else ''}"
                for line in self.lines]
        rows.append(f"  {'-' * (width + 12)}")
        rows.append(f"  {'TOTAL'.ljust(width)}  {format_bytes(self.total_bytes).rjust(10)}")
        if self.disk_free_bytes:
            verdict = "fits" if self.fits else "DOES NOT FIT"
            rows.append(f"  {'disk free'.ljust(width)}  "
                        f"{format_bytes(self.disk_free_bytes).rjust(10)}  ({verdict})")
        return "\n".join(rows)


#: Sequence-length tensors dumped per module per sample. A module records its input
#: and its output, plus whatever its architecture passes alongside the hidden state —
#: rotary cos/sin, an attention mask, convolution state — and those are dumped too
#: because replay needs them. Measured at just under 7 for Qwen3.5's hybrid stack;
#: this is a preflight, so it rounds up rather than down.
DEFAULT_TENSORS_PER_MODULE = 7


@dataclass
class TraceShape:
    """The shape of the tracing workload, as far as storage is concerned."""

    n_modules: int
    #: Bytes of one module-boundary tensor per sequence position (hidden * dtype).
    boundary_bytes_per_token: int
    short_seq_lens: list[int] = field(default_factory=list)
    long_seq_lens: list[int] = field(default_factory=list)
    #: Extra per-module state dumped once per sample regardless of length
    #: (routing maps, per-layer scalars).
    aux_bytes_per_module: int = 0
    tensors_per_module: int = DEFAULT_TENSORS_PER_MODULE


def estimate_storage(
    *,
    checkpoint_bytes: int,
    module_weight_bytes: int,
    dequant_bytes: int,
    trace: TraceShape,
    policy: DumpPolicy,
    host: HostInfo | None = None,
    stream_checkpoint: bool = False,
) -> StorageEstimate:
    """Project the disk footprint of a run.

    Under ``stream_checkpoint`` only one shard is resident at a time; pass the
    largest shard size as ``checkpoint_bytes``.
    """
    est = StorageEstimate(disk_free_bytes=host.disk_free_bytes if host else 0)
    est.add(
        "checkpoint" + (" (streamed, peak)" if stream_checkpoint else ""),
        checkpoint_bytes,
    )
    if policy.cache_weights:
        est.add("module weight dumps", module_weight_bytes, "deduplicated across identical modules")
    if policy.cache_dequant:
        est.add("bf16 dequant mirror", dequant_bytes, "set cache_dequant=false to recompute")

    def _featmap_bytes(seq_lens: list[int]) -> int:
        total = 0
        for seq_len in seq_lens:
            kept = policy.kept_positions(seq_len)
            total += (trace.tensors_per_module * trace.n_modules
                      * kept * trace.boundary_bytes_per_token)
        return total

    short_bytes = _featmap_bytes(trace.short_seq_lens)
    long_bytes = _featmap_bytes(trace.long_seq_lens)
    est.add("feature maps (short)", short_bytes, f"{len(trace.short_seq_lens)} samples, full")
    slicing = (f"windowed to {policy.slice_head}+{policy.slice_tail}"
               if policy.slice_long else "full")
    est.add("feature maps (long)", long_bytes, f"{len(trace.long_seq_lens)} samples, {slicing}")

    if policy.decode_steps and trace.short_seq_lens:
        est.add(
            "feature maps (decode)",
            trace.tensors_per_module * trace.n_modules * policy.decode_steps
            * trace.boundary_bytes_per_token,
            f"{policy.decode_steps} steps, first sample only",
        )

    n_samples = len(trace.short_seq_lens) + len(trace.long_seq_lens)
    est.add("aux (routing maps, metadata)", trace.aux_bytes_per_module * trace.n_modules * n_samples)
    return est


class StoragePreflightError(RuntimeError):
    """Raised when a run cannot possibly fit on the available disk."""


def preflight(
    estimate: StorageEstimate,
    *,
    strict: bool = True,
    reserve_bytes: int = 8 * GIB,
) -> list[str]:
    """Check an estimate against free disk, returning warnings.

    Under ``strict``, raises when the projection exceeds free disk minus
    ``reserve_bytes``.
    """
    warnings: list[str] = []
    free = estimate.disk_free_bytes
    if not free:
        return ["Could not determine free disk space; skipping storage preflight."]

    budget = max(free - reserve_bytes, 0)
    if estimate.total_bytes > budget:
        message = (
            f"Projected footprint {format_bytes(estimate.total_bytes)} exceeds usable disk "
            f"{format_bytes(budget)} ({format_bytes(free)} free minus "
            f"{format_bytes(reserve_bytes)} reserve).\n{estimate.render()}\n"
            "Options: set cache_dequant=false, set cache_weights=false, reduce the input "
            "set, or point --artifact-root at a larger volume."
        )
        if strict:
            raise StoragePreflightError(message)
        warnings.append(message)
    elif estimate.total_bytes > budget * 0.8:
        warnings.append(
            f"Projected footprint {format_bytes(estimate.total_bytes)} uses over 80% of usable "
            f"disk ({format_bytes(budget)})."
        )
    return warnings
