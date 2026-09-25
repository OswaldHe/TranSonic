# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The floorplan: what goes where, split how, resident in which tier, in what order.

This is the schema for the only file the exploration loop may edit. Everything the agent
decides is expressed here and nothing else is writable, so this module is where the design
space is actually defined — a dimension absent from ``PARTITION_DIMS`` is a dimension the
loop cannot explore, whatever the prompt says.

Validation is deliberately strict and deliberately *structural*. It answers "is this a
well-formed plan for this hardware and this model", never "is this a good plan": no check
here consults a cost. A malformed plan has to fail before the simulator runs, because a
simulator fed a plan whose fractions sum to 0.9 will happily report a latency 10% too low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Dimensions a module may be partitioned along. Closed vocabulary: the simulator has to
#: know what a split *means* to cost it, so a new dimension is a code change here and in
#: ``sim/api.py``, not a string an agent can invent in the floorplan.
PARTITION_DIMS = frozenset({
    "head",     # attention heads / KV heads
    "hidden",   # the 5120 model dim, or an MLP's intermediate dim
    "expert",   # MoE routed experts
    "seq",      # sequence positions (context/sequence parallel)
    "batch",    # batch members (data parallel)
    "layer",    # whole layers, for a module that spans several (pipeline parallel)
    "vocab",    # the 129280-entry vocabulary, for embed and lm_head
    "ngram",    # Engram n-gram table rows
})

#: What rejoining a split costs. ``none`` is legal and means the split needs no
#: communication — true for a pure batch split, and for a split across the two physical
#: cores of one logical core, which share an address space.
COLLECTIVES = frozenset({
    "none", "allreduce", "allgather", "reduce_scatter", "all_to_all", "p2p",
})

#: Dimensions that partition the *weights*. Splitting along one of these divides the parameter
#: bytes each unit holds; splitting along any other dimension divides only the activations and
#: leaves every participant holding the module's full weights.
#:
#: The distinction is load-bearing for capacity. Dividing weight bytes by a batch or sequence
#: factor made a batch-64 placement account for 1/64 of its real weights, so a plan that
#: overflowed a bank passed the gate and data parallelism looked nearly free.
WEIGHT_PARTITION_DIMS = frozenset({"head", "hidden", "expert", "layer", "vocab", "ngram"})

#: Dimensions along which a split genuinely needs no communication to rejoin. Data parallelism
#: is the only one: each participant computes a complete result for its own batch members.
#:
#: ``collective: none`` is otherwise only legal when every participant shares one logical
#: NeuronCore, since the two physical cores at LNC=2 share an address space. Without this
#: check a plan could declare a head or expert split across the whole torus as `none`, delete
#: all of its communication, and win the metrics with a deployment that cannot run.
COMMUNICATION_FREE_DIMS = frozenset({"batch"})

#: Tiers a module's weights may be resident in. Must name a tier the system YAML declares
#: with ``inference_path`` unset or true.
WEIGHT_TIERS = frozenset({"hbm_bank", "device_hbm", "peer_hbm", "host_dram", "nvme"})

#: Which dimensions make sense for which module kind, keyed by the ``kind`` field of the
#: partition graph. A split along a dimension the module does not have is not a bad idea,
#: it is a meaningless one, and it would silently produce shards of size zero.
DIMS_BY_KIND: dict[str, frozenset[str]] = {
    "embed":     frozenset({"vocab", "hidden", "batch", "seq"}),
    "attention": frozenset({"head", "hidden", "seq", "batch", "layer"}),
    "mlp":       frozenset({"expert", "hidden", "batch", "seq", "layer"}),
    "norm":      frozenset({"hidden", "seq", "batch", "layer"}),
    "lm_head":   frozenset({"vocab", "hidden", "batch", "seq"}),
    "vision":    frozenset({"hidden", "batch", "seq", "layer"}),
    "other":     frozenset({"hidden", "batch", "seq", "layer"}),
}

#: Refinements to ``DIMS_BY_KIND`` for modules the graph files under ``other`` but which do
#: have a characteristic dimension. Matched as a substring of the module id, longest first.
#:
#: The keys are the id fragments this artifact actually uses — ``layers.1.engram``,
#: ``layers.0.hc_attn_in``, ``hc_expand`` — not the group names in ``modules/``. A hint that
#: matches nothing silently falls back to ``other``'s conservative set, which would forbid
#: the ``ngram`` split that is the whole question for Engram, so ``tests/`` asserts each
#: hint matches at least one real module id.
DIMS_BY_ID_HINT: dict[str, frozenset[str]] = {
    "engram": frozenset({"ngram", "hidden", "head", "batch", "seq", "layer"}),
    "dspark": frozenset({"expert", "hidden", "vocab", "batch", "seq", "layer"}),
    "hc_":    frozenset({"hidden", "batch", "seq", "layer"}),
}

_ADDRESS = re.compile(r"^d(\d+)\.l(\d+)(?:\.p(\d+))?$")


class FloorplanError(ValueError):
    """The floorplan is not a well-formed plan for this hardware and this model."""


@dataclass(frozen=True)
class Address:
    """One placement target: ``d3.l2``, or ``d3.l2.p1`` for a physical-core subdivision."""

    device: int
    logical_nc: int
    physical_nc: int | None = None

    @classmethod
    def parse(cls, text: str) -> "Address":
        match = _ADDRESS.match(text.strip())
        if not match:
            raise FloorplanError(
                f"'{text}' is not an address. Expected d<device>.l<logical_nc> "
                f"(optionally .p<physical_nc>), e.g. d0.l0 or d15.l3.p1"
            )
        device, logical, physical = match.groups()
        return cls(int(device), int(logical), None if physical is None else int(physical))

    def logical(self) -> "Address":
        """This address with any physical-core suffix dropped."""
        return Address(self.device, self.logical_nc)

    def __str__(self) -> str:
        tail = "" if self.physical_nc is None else f".p{self.physical_nc}"
        return f"d{self.device}.l{self.logical_nc}{tail}"


@dataclass(frozen=True)
class Split:
    """One factor of a partition: a dimension, how many ways, and what rejoining costs."""

    dim: str
    factor: int
    collective: str = "allreduce"

    def validate(self, where: str) -> None:
        if self.dim not in PARTITION_DIMS:
            raise FloorplanError(
                f"{where}: unknown partition dim '{self.dim}'. "
                f"Known: {', '.join(sorted(PARTITION_DIMS))}"
            )
        if self.factor < 1:
            raise FloorplanError(f"{where}: split factor must be >= 1, got {self.factor}")
        if self.collective not in COLLECTIVES:
            raise FloorplanError(
                f"{where}: unknown collective '{self.collective}'. "
                f"Known: {', '.join(sorted(COLLECTIVES))}"
            )


@dataclass(frozen=True)
class Residency:
    """Where a module's weights live, and how much of them is cached nearer."""

    tier: str = "hbm_bank"
    #: Fraction of the weight bytes held in ``cache_tier``. 1.0 means fully resident in
    #: ``tier`` with no tiering; below 1.0 the rest is fetched per use and charged.
    resident_fraction: float = 1.0
    cache_tier: str | None = None
    #: For a tiered lookup table, the fraction of accesses the cache is assumed to serve.
    #: An assumption, not a measurement — the report is required to say so.
    hit_rate: float | None = None
    #: Which device owns the weights, for ``tier: peer_hbm``. Optional but consequential: it is
    #: what lets a remote read be charged its actual torus distance instead of a flat cost, so
    #: that reaching a neighbour and reaching the far corner stop looking identical.
    backing_device: int | None = None

    def validate(self, where: str) -> None:
        if self.tier not in WEIGHT_TIERS:
            raise FloorplanError(
                f"{where}: unknown weight tier '{self.tier}'. "
                f"Known: {', '.join(sorted(WEIGHT_TIERS))}"
            )
        if not 0.0 < self.resident_fraction <= 1.0:
            raise FloorplanError(
                f"{where}: resident_fraction must be in (0, 1], got {self.resident_fraction}"
            )
        if self.resident_fraction < 1.0:
            if self.cache_tier is None:
                raise FloorplanError(
                    f"{where}: resident_fraction {self.resident_fraction} < 1 needs a "
                    f"cache_tier saying where the resident part is held"
                )
            if self.cache_tier not in WEIGHT_TIERS:
                raise FloorplanError(f"{where}: unknown cache_tier '{self.cache_tier}'")
            if self.hit_rate is None:
                raise FloorplanError(
                    f"{where}: a tiered module needs an explicit hit_rate — the cost of a "
                    f"miss is the whole point of tiering it, and it cannot be inferred"
                )
            if not 0.0 <= self.hit_rate <= 1.0:
                raise FloorplanError(f"{where}: hit_rate must be in [0, 1], got {self.hit_rate}")
        if self.backing_device is not None:
            if self.tier != "peer_hbm":
                raise FloorplanError(
                    f"{where}: backing_device only means something for tier 'peer_hbm', "
                    f"not '{self.tier}'"
                )
            if self.backing_device < 0:
                raise FloorplanError(
                    f"{where}: backing_device must be a device index, got {self.backing_device}"
                )


@dataclass
class Placement:
    """One module, or a fraction of one, on a set of units.

    ``units`` is ordered and its length must equal the product of the split factors, so
    shard *i* of the partition lands on ``units[i]`` with no ambiguity about which shard
    went where. That ordering is what the report's decomposition recipe reads back.
    """

    module: str
    units: list[Address]
    splits: list[Split] = field(default_factory=list)
    fraction: float = 1.0
    residency: Residency = field(default_factory=Residency)
    #: Schedule position. Modules sharing a stage may run concurrently if their data
    #: dependencies allow; a higher stage never starts before a lower one it depends on.
    stage: int = 0
    #: Credit communication/compute overlap for this placement's collectives.
    overlap_collectives: bool = False

    def shard_count(self) -> int:
        count = 1
        for split in self.splits:
            count *= split.factor
        return count

    def validate(self, where: str) -> None:
        for index, split in enumerate(self.splits):
            split.validate(f"{where}.splits[{index}]")
        self.residency.validate(where)
        if not 0.0 < self.fraction <= 1.0:
            raise FloorplanError(f"{where}: fraction must be in (0, 1], got {self.fraction}")
        if not self.units:
            raise FloorplanError(f"{where}: no units — a placement must name where it runs")
        expected = self.shard_count()
        if len(self.units) != expected:
            dims = " x ".join(f"{s.dim}:{s.factor}" for s in self.splits) or "no split"
            raise FloorplanError(
                f"{where}: {len(self.units)} unit(s) for {expected} shard(s) ({dims}). "
                f"units must be one per shard, in shard order"
            )
        if len(set(self.units)) != len(self.units):
            duplicated = sorted({str(u) for u in self.units if self.units.count(u) > 1})
            raise FloorplanError(f"{where}: repeats unit(s) {', '.join(duplicated)}")
        seen: set[str] = set()
        for split in self.splits:
            if split.dim in seen:
                raise FloorplanError(f"{where}: splits the '{split.dim}' dim twice")
            seen.add(split.dim)

        # `collective: none` is free, so it has to be earned. It is legal for a batch split,
        # and legal for any split whose participants all share one logical NeuronCore (the two
        # physical cores at LNC=2 share an address space). Anywhere else it would delete real
        # communication: a head or expert split declared `none` across the torus would cost
        # nothing to rejoin and win the metrics with a deployment that cannot run.
        shares_one_core = len({u.logical() for u in self.units}) <= 1
        for index, split in enumerate(self.splits):
            if split.collective != "none":
                continue
            if split.dim in COMMUNICATION_FREE_DIMS or shares_one_core:
                continue
            raise FloorplanError(
                f"{where}.splits[{index}]: 'collective: none' is only free for a "
                f"{'/'.join(sorted(COMMUNICATION_FREE_DIMS))} split, or when every unit shares "
                f"one logical NeuronCore. This splits '{split.dim}' across "
                f"{len({u.logical() for u in self.units})} logical core(s), which has to be "
                f"rejoined — name the collective that does it"
            )

        if self.stage < 0:
            raise FloorplanError(f"{where}: stage must be >= 0, got {self.stage}")


@dataclass
class Runtime:
    """Execution knobs that are not placements but do change the schedule."""

    prefill_chunk_tokens: int = 2048
    decode_micro_batch: int = 1
    #: Whether a pipeline stage may begin a chunk before the previous stage finishes the
    #: next one. Off means strict stage-at-a-time, which is simpler and usually slower.
    pipeline_chunks: bool = True

    def validate(self) -> None:
        if self.prefill_chunk_tokens < 1:
            raise FloorplanError(
                f"runtime.prefill_chunk_tokens must be >= 1, got {self.prefill_chunk_tokens}"
            )
        if self.decode_micro_batch < 1:
            raise FloorplanError(
                f"runtime.decode_micro_batch must be >= 1, got {self.decode_micro_batch}"
            )


@dataclass
class Floorplan:
    """A complete deployment scheme."""

    target: str
    placements: list[Placement]
    runtime: Runtime = field(default_factory=Runtime)
    version: int = 1
    notes: str = ""

    def modules(self) -> set[str]:
        return {p.module for p in self.placements}

    def units_used(self) -> set[Address]:
        return {u.logical() for p in self.placements for u in p.units}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Floorplan":
        if not isinstance(data, dict):
            raise FloorplanError("the floorplan did not parse as a mapping")
        unknown = set(data) - {"version", "target", "placements", "runtime", "notes"}
        if unknown:
            raise FloorplanError(
                f"unknown top-level key(s): {', '.join(sorted(unknown))}. "
                f"Known: version, target, placements, runtime, notes"
            )
        target = data.get("target")
        if not isinstance(target, str) or not target.strip():
            raise FloorplanError("'target' must name a system YAML, e.g. trn2-16device")

        raw_runtime = data.get("runtime") or {}
        if not isinstance(raw_runtime, dict):
            raise FloorplanError("'runtime' must be a mapping")
        runtime_unknown = set(raw_runtime) - {
            "prefill_chunk_tokens", "decode_micro_batch", "pipeline_chunks",
        }
        if runtime_unknown:
            raise FloorplanError(
                f"unknown runtime key(s): {', '.join(sorted(runtime_unknown))}"
            )
        runtime = Runtime(
            prefill_chunk_tokens=int(raw_runtime.get("prefill_chunk_tokens", 2048)),
            decode_micro_batch=int(raw_runtime.get("decode_micro_batch", 1)),
            pipeline_chunks=bool(raw_runtime.get("pipeline_chunks", True)),
        )

        raw_placements = data.get("placements")
        if not isinstance(raw_placements, list) or not raw_placements:
            raise FloorplanError("'placements' must be a non-empty list")

        placements: list[Placement] = []
        for index, entry in enumerate(raw_placements):
            where = f"placements[{index}]"
            if not isinstance(entry, dict):
                raise FloorplanError(f"{where}: must be a mapping")
            entry_unknown = set(entry) - {
                "module", "units", "splits", "fraction", "weights", "stage",
                "overlap_collectives",
            }
            if entry_unknown:
                raise FloorplanError(
                    f"{where}: unknown key(s) {', '.join(sorted(entry_unknown))}"
                )
            module = entry.get("module")
            if not isinstance(module, str) or not module.strip():
                raise FloorplanError(f"{where}: 'module' must be a module id")

            raw_units = entry.get("units")
            if isinstance(raw_units, str):
                raw_units = [raw_units]
            if not isinstance(raw_units, list):
                raise FloorplanError(f"{where}: 'units' must be an address or a list of them")
            units = [Address.parse(str(u)) for u in raw_units]

            splits: list[Split] = []
            for split_index, raw_split in enumerate(entry.get("splits") or []):
                if not isinstance(raw_split, dict):
                    raise FloorplanError(f"{where}.splits[{split_index}]: must be a mapping")
                split_unknown = set(raw_split) - {"dim", "factor", "collective"}
                if split_unknown:
                    raise FloorplanError(
                        f"{where}.splits[{split_index}]: unknown key(s) "
                        f"{', '.join(sorted(split_unknown))}"
                    )
                splits.append(Split(
                    dim=str(raw_split.get("dim", "")),
                    factor=int(raw_split.get("factor", 1)),
                    collective=str(raw_split.get("collective", "allreduce")),
                ))

            raw_weights = entry.get("weights") or {}
            if not isinstance(raw_weights, dict):
                raise FloorplanError(f"{where}: 'weights' must be a mapping")
            weights_unknown = set(raw_weights) - {
                "tier", "resident_fraction", "cache_tier", "hit_rate", "backing_device",
            }
            if weights_unknown:
                raise FloorplanError(
                    f"{where}.weights: unknown key(s) {', '.join(sorted(weights_unknown))}"
                )
            hit_rate = raw_weights.get("hit_rate")
            residency = Residency(
                tier=str(raw_weights.get("tier", "hbm_bank")),
                resident_fraction=float(raw_weights.get("resident_fraction", 1.0)),
                cache_tier=(None if raw_weights.get("cache_tier") is None
                            else str(raw_weights["cache_tier"])),
                hit_rate=None if hit_rate is None else float(hit_rate),
                backing_device=(None if raw_weights.get("backing_device") is None
                                else int(raw_weights["backing_device"])),
            )

            placements.append(Placement(
                module=module,
                units=units,
                splits=splits,
                fraction=float(entry.get("fraction", 1.0)),
                residency=residency,
                stage=int(entry.get("stage", 0)),
                overlap_collectives=bool(entry.get("overlap_collectives", False)),
            ))

        plan = cls(
            target=target.strip(),
            placements=placements,
            runtime=runtime,
            version=int(data.get("version", 1)),
            notes=str(data.get("notes", "")),
        )
        plan.validate()
        return plan

    @classmethod
    def load(cls, path: str | Path) -> "Floorplan":
        text = Path(path).read_text()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise FloorplanError(f"{path} is not valid YAML: {exc}") from exc
        return cls.from_dict(data or {})

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Structural validation, independent of hardware and model.

        ``validate_against`` adds the checks that need them. Split here so a malformed
        file reports as malformed even when the graph or the system YAML is unavailable.
        """
        if self.version != 1:
            raise FloorplanError(f"unsupported floorplan version {self.version} (expected 1)")
        self.runtime.validate()
        for index, placement in enumerate(self.placements):
            placement.validate(f"placements[{index}]")

        totals: dict[str, float] = {}
        for placement in self.placements:
            totals[placement.module] = totals.get(placement.module, 0.0) + placement.fraction
        for module, total in sorted(totals.items()):
            # Tolerance covers writing thirds as 0.333; anything looser would let a module
            # be quietly under-deployed, which is exactly what the coverage check exists
            # to catch.
            if abs(total - 1.0) > 1e-6:
                raise FloorplanError(
                    f"module '{module}': fractions sum to {total:g}, not 1. "
                    f"Every part of a module has to be deployed somewhere"
                )

    def validate_against(
        self,
        system: dict[str, Any],
        graph_modules: dict[str, dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Check the plan against the hardware it targets and the model it deploys.

        ``system`` is a resolved system YAML (see ``parser.load_system``); ``graph_modules``
        maps module id to its partition-graph entry; ``config`` is the model's ``config.json``,
        which is what makes a split factor checkable against the dimension's real size. All
        optional-by-argument rather than optional-in-effect: the checker always passes them,
        and the tests exercise the structural half alone.
        """
        devices = int(system["hierarchy"]["device"]["count"])
        logical_per_device = int(system["hierarchy"]["logical_nc"]["count"])
        physical_per_logical = int(system["hierarchy"]["physical_nc"]["count"])
        declared_tiers = {t["name"] for t in system.get("tiers", [])}
        host_only = {
            t["name"] for t in system.get("tiers", [])
            if t.get("inference_path") is False
        }

        for index, placement in enumerate(self.placements):
            where = f"placements[{index}] ({placement.module})"
            for unit in placement.units:
                if not 0 <= unit.device < devices:
                    raise FloorplanError(
                        f"{where}: {unit} names device {unit.device}, but '{self.target}' "
                        f"has {devices} (d0..d{devices - 1})"
                    )
                if not 0 <= unit.logical_nc < logical_per_device:
                    raise FloorplanError(
                        f"{where}: {unit} names logical NC {unit.logical_nc}, but a device "
                        f"has {logical_per_device} (l0..l{logical_per_device - 1})"
                    )
                if unit.physical_nc is not None and not 0 <= unit.physical_nc < physical_per_logical:
                    raise FloorplanError(
                        f"{where}: {unit} names physical NC {unit.physical_nc}, but a "
                        f"logical NC has {physical_per_logical}"
                    )

            tier = placement.residency.tier
            if tier not in declared_tiers:
                raise FloorplanError(
                    f"{where}: tier '{tier}' is not declared by target '{self.target}'. "
                    f"Declared: {', '.join(sorted(declared_tiers))}"
                )
            if tier in host_only:
                raise FloorplanError(
                    f"{where}: tier '{tier}' is load-time only and may not be read during "
                    f"a forward pass"
                )
            cache_tier = placement.residency.cache_tier
            if cache_tier is not None and cache_tier not in declared_tiers:
                raise FloorplanError(
                    f"{where}: cache_tier '{cache_tier}' is not declared by "
                    f"target '{self.target}'"
                )

            if graph_modules is not None:
                entry = graph_modules.get(placement.module)
                if entry is None:
                    raise FloorplanError(
                        f"{where}: no such module in the partition graph. "
                        f"Module ids come from plan/partition_graph.yaml"
                    )
                legal = legal_dims(placement.module, entry)
                for split in placement.splits:
                    if split.dim not in legal:
                        raise FloorplanError(
                            f"{where}: cannot split a '{entry.get('kind')}' module along "
                            f"'{split.dim}'. Legal here: {', '.join(sorted(legal))}"
                        )
                    limit = dimension_extent(split.dim, entry, config)
                    if limit is not None and split.factor > limit:
                        raise FloorplanError(
                            f"{where}: {split.factor}-way split along '{split.dim}', but this "
                            f"module has only {limit} of them. A split cannot have more shards "
                            f"than the dimension has elements — the surplus shards would hold "
                            f"nothing while the framework still divided the work by "
                            f"{split.factor}"
                        )

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable form. ``from_dict(plan.to_dict())`` must equal ``plan``."""
        out: dict[str, Any] = {"version": self.version, "target": self.target}
        if self.notes:
            out["notes"] = self.notes
        out["runtime"] = {
            "prefill_chunk_tokens": self.runtime.prefill_chunk_tokens,
            "decode_micro_batch": self.runtime.decode_micro_batch,
            "pipeline_chunks": self.runtime.pipeline_chunks,
        }
        placements: list[dict[str, Any]] = []
        for placement in self.placements:
            entry: dict[str, Any] = {
                "module": placement.module,
                "units": [str(u) for u in placement.units],
            }
            if placement.splits:
                entry["splits"] = [
                    {"dim": s.dim, "factor": s.factor, "collective": s.collective}
                    for s in placement.splits
                ]
            if placement.fraction != 1.0:
                entry["fraction"] = placement.fraction
            weights: dict[str, Any] = {"tier": placement.residency.tier}
            if placement.residency.resident_fraction != 1.0:
                weights["resident_fraction"] = placement.residency.resident_fraction
                weights["cache_tier"] = placement.residency.cache_tier
                weights["hit_rate"] = placement.residency.hit_rate
            if placement.residency.backing_device is not None:
                weights["backing_device"] = placement.residency.backing_device
            entry["weights"] = weights
            entry["stage"] = placement.stage
            if placement.overlap_collectives:
                entry["overlap_collectives"] = True
            placements.append(entry)
        out["placements"] = placements
        return out

    def dump(self, path: str | Path, header: str = "") -> None:
        body = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        text = f"{header.rstrip()}\n\n{body}" if header else body
        Path(path).write_text(text)


def dimension_extent(
    dim: str, entry: dict[str, Any], config: dict[str, Any] | None,
) -> int | None:
    """How many elements this module has along ``dim``, or None when it cannot be determined.

    The upper bound on a legal split factor. Without it a two-layer module could declare a
    64-way ``layer`` split, and the framework would dutifully present 64 shards and divide the
    parameters and the work by 64 while 62 of them had no layer to execute — a 64x speedup
    from arithmetic alone.

    ``batch`` is bounded by the workloads, which are fixed at batch 1 (see
    ``sim/runner.WORKLOADS``), so any batch split is invalid rather than merely useless. None
    is returned for dimensions whose extent this artifact does not state, which is honest: a
    bound that has to be guessed is worse than no bound.
    """
    if dim == "layer":
        return max(len(entry.get("layer_indices") or []), 1)
    if dim == "batch":
        return 1
    if config is None:
        return None
    if dim == "head":
        return _positive(config.get("engram_n_heads") if "engram" in str(entry.get("id", ""))
                         else config.get("n_heads"))
    if dim == "expert":
        return _positive(config.get("n_routed_experts"))
    if dim == "vocab":
        return _positive(config.get("vocab_size"))
    if dim == "hidden":
        return _positive(config.get("dim"))
    if dim == "seq":
        return _positive(config.get("max_seq_len"))
    if dim == "ngram":
        rows = config.get("engram_num_embeddings")
        if isinstance(rows, list):
            rows = min(rows) if rows else None
        return _positive(rows)
    return None


def _positive(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def legal_dims(module_id: str, entry: dict[str, Any]) -> frozenset[str]:
    """Which dimensions this module may be split along.

    Kind first, then an id hint, because the partition graph files Engram, the
    hyper-connection plumbing and the DSpark heads all under ``other`` while they have
    quite different dimensions — and ``other``'s conservative default would forbid the
    ``ngram`` split that is the whole question for Engram.
    """
    for hint in sorted(DIMS_BY_ID_HINT, key=len, reverse=True):
        if hint in module_id.lower():
            return DIMS_BY_ID_HINT[hint]
    return DIMS_BY_KIND.get(str(entry.get("kind", "other")), DIMS_BY_KIND["other"])
