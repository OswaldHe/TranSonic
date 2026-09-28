# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projecting a floorplan placement onto one device.

The ranked floorplan is written for a 16-device trn2.48xlarge. Development happens on a
one-device trn2.3xlarge, and the interesting modules do not fit: in `schemes/rank1.yaml` every
`.ffn` and every `.attention` is spread over eight logical NeuronCores spanning *two* devices,
and both Engram tables over sixteen spanning four. Only the hyper-connection plumbing, the
norms, `embed` and the small MTP heads are single-device placements.

Of the scheme's 271 placements, 182 fit one device and 89 do not — and those 89 are exactly the
ones worth optimizing: all 43 `.ffn`, all 43 `.attention`, `lm_head`, and both Engram tables. So a
literal "this module must fit in one device" check refuses every module anyone would bring here,
which is what the projection exists to avoid: reduce the split product to exactly the four logical
cores one device has, and record what was given up.

The rule, and why it is this rule:

- **Drop non-weight-partitioning factors first.** A `batch` split divides activations and
  leaves every participant holding the module's full weights, so dropping it costs no
  per-core capacity. Dropping a `head` or `expert` factor doubles what each core must hold.
  `attention: head x4 * batch x2` therefore projects to `head x4`, not `head x2 * batch x2`.
- **Then scale the weight-partitioning factor down** to the largest divisor of itself that
  brings the product to the target. `ffn: expert x8` projects to `expert x4`.
- **Never project up.** A placement already at or below four units is used unchanged; the point
  is to fit the device, not to fill it.

The projection is a *divergence from the ranked plan* and is reported as one. For `.ffn` the
divergence is precisely the trade the floorplan priced and rejected — its report chose 8-wide
tensor parallelism because it halves per-bank weight residency and decode is bank-bandwidth
bound — so an optimized kernel for `expert x4` is a kernel for the runner-up plan. That is a
legitimate thing to want (it is the fastest single-device MoE) and a misleading thing to forget,
which is why `describe()` puts it in the repo the agent reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from optimization import PROJECTION_TARGET_UNITS

#: `Factor.label()`'s shape, for reading a recorded projection back: `expertx8(all_to_all)`.
_LABEL = re.compile(r"(?P<dim>\w+)x(?P<factor>\d+)\((?P<collective>[\w.]+)\)")

#: Dimensions whose split divides the *weight* bytes each unit holds. Mirrors
#: `floorplan.schema.WEIGHT_PARTITION_DIMS`, duplicated rather than imported so a projection can
#: be computed from a scheme file alone, without the floorplan package or its probe data.
#: `tests/test_optimization_projection.py` asserts the two stay equal.
WEIGHT_PARTITION_DIMS = frozenset({"head", "hidden", "expert", "layer", "vocab", "ngram"})


class ProjectionError(ValueError):
    """The placement cannot be projected onto one device."""


@dataclass(frozen=True)
class Factor:
    """One split factor of a placement: a dimension, a width, and what rejoining costs."""

    dim: str
    factor: int
    collective: str = "allreduce"

    def partitions_weights(self) -> bool:
        return self.dim in WEIGHT_PARTITION_DIMS

    def label(self) -> str:
        return f"{self.dim}x{self.factor}({self.collective})"

    @classmethod
    def parse(cls, label: str) -> Factor:
        """The inverse of `label()`, so a recorded projection can be read back.

        `dropped` and `scaled` are serialized as labels rather than as objects, which is right for
        a human reading `projection.json` and means rehydrating one has to parse them.
        """
        match = _LABEL.fullmatch(label.strip())
        if not match:
            raise ProjectionError(f"cannot read '{label}' as a split factor")
        return cls(dim=match.group("dim"), factor=int(match.group("factor")),
                   collective=match.group("collective"))


@dataclass
class Projection:
    """What one module's placement becomes on a single device, and what that cost.

    Everything a downstream stage needs to state the cut without re-reading the scheme, and
    everything the report needs to say how far from the ranked plan this run has moved.
    """

    module: str
    planned: list[Factor]
    planned_units: int
    planned_devices: int
    projected: list[Factor]
    projected_units: int
    dropped: list[Factor] = field(default_factory=list)
    scaled: list[tuple[Factor, Factor]] = field(default_factory=list)
    target: str = ""

    @property
    def diverges(self) -> bool:
        """Whether the projection is a different plan, rather than the plan restated."""
        return bool(self.dropped or self.scaled)

    @property
    def shard_fraction(self) -> str:
        """The fraction of the module one projected rank holds, as a legible ratio."""
        return f"1/{self.projected_units}"

    def weight_residency_ratio(self) -> float:
        """How much more weight a projected core holds than a planned one.

        The number the floorplan's own argument turns on: for `.ffn` this is 2.0, and the report
        that ranked the plan chose 8-wide tensor parallelism specifically to avoid it.
        """
        planned = _product(f.factor for f in self.planned if f.partitions_weights()) or 1
        projected = _product(f.factor for f in self.projected if f.partitions_weights()) or 1
        return planned / projected

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "target": self.target,
            "planned": {
                "units": self.planned_units,
                "devices": self.planned_devices,
                "splits": [
                    {"dim": f.dim, "factor": f.factor, "collective": f.collective}
                    for f in self.planned
                ],
            },
            "projected": {
                "units": self.projected_units,
                "devices": 1,
                "splits": [
                    {"dim": f.dim, "factor": f.factor, "collective": f.collective}
                    for f in self.projected
                ],
            },
            "diverges": self.diverges,
            "dropped": [f.label() for f in self.dropped],
            "scaled": [f"{was.label()} -> {now.label()}" for was, now in self.scaled],
            "weight_residency_ratio": round(self.weight_residency_ratio(), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Projection:
        """Rehydrate a projection recorded by `init`, so later stages read it rather than redo it.

        Recomputing is what the pipeline used to do, and it means a scheme edited mid-run silently
        re-plans the module: `assemble` would build against a cut the submodule repo was never
        made for, and `report` would label finished measurements with a split that never ran.
        """
        def splits(side: str) -> list[Factor]:
            return [Factor(dim=str(s.get("dim")), factor=int(s.get("factor") or 1),
                           collective=str(s.get("collective") or "allreduce"))
                    for s in ((data.get(side) or {}).get("splits") or [])]

        planned, projected = data.get("planned") or {}, data.get("projected") or {}
        scaled: list[tuple[Factor, Factor]] = []
        for entry in data.get("scaled") or []:
            was, _, now = str(entry).partition("->")
            scaled.append((Factor.parse(was), Factor.parse(now)))
        return cls(
            module=str(data.get("module") or ""),
            planned=splits("planned"),
            planned_units=int(planned.get("units") or 0),
            planned_devices=int(planned.get("devices") or 0),
            projected=splits("projected"),
            projected_units=int(projected.get("units") or 0),
            dropped=[Factor.parse(d) for d in (data.get("dropped") or [])],
            scaled=scaled,
            target=str(data.get("target") or ""),
        )

    def differences(self, other: Projection) -> list[str]:
        """How this projection and another disagree, in the terms a reader would check.

        Only the fields a later stage builds on. The unit count alone is not enough: a scheme
        edited from `expert x4` to `head x4` keeps the width and changes the cut entirely, and a
        changed collective changes what rejoining the ranks costs.
        """
        found: list[str] = []
        if self.projected_units != other.projected_units:
            found.append(f"{self.projected_units} unit(s) -> {other.projected_units}")
        mine = " * ".join(f.label() for f in self.projected) or "no split"
        theirs = " * ".join(f.label() for f in other.projected) or "no split"
        if mine != theirs:
            found.append(f"{mine} -> {theirs}")
        if self.planned_units != other.planned_units:
            found.append(f"planned {self.planned_units} unit(s) -> {other.planned_units}")
        return found

    def describe(self) -> str:
        """The prose the submodule repo carries as `FLOORPLAN.md`.

        Written for the agent that has to cut the module, so it states the planned placement,
        the projection, and — when they differ — that the projection is a departure and what it
        costs. An agent told only "split the experts four ways" would have no way to know it was
        working against the runner-up plan.
        """
        planned = " * ".join(f.label() for f in self.planned) or "no split"
        projected = " * ".join(f.label() for f in self.projected) or "no split"
        lines = [
            f"# Planned placement of `{self.module}`",
            "",
            f"The ranked floorplan (`{self.target or 'the scheme this project was built from'}`)"
            f" places this module on **{self.planned_units} logical NeuronCore(s) spanning"
            f" {self.planned_devices} device(s)**:",
            "",
            f"    {planned}",
            "",
            "## Projected onto one device",
            "",
            f"This host is a one-device trn2.3xlarge with {PROJECTION_TARGET_UNITS} logical"
            f" NeuronCores, so the placement is projected to"
            f" **{self.projected_units} unit(s) on one device**:",
            "",
            f"    {projected}",
            "",
            f"One rank therefore holds **{self.shard_fraction}** of this module.",
        ]
        if not self.diverges:
            lines += [
                "",
                "The plan already fit one device, so nothing was given up: this is the ranked"
                " placement, not an approximation of it.",
            ]
            return "\n".join(lines) + "\n"

        lines += ["", "## What the projection gave up", ""]
        for f in self.dropped:
            lines.append(
                f"- **Dropped `{f.label()}`.** A `{f.dim}` split divides activations, not"
                f" weights, so dropping it costs no per-core capacity — it costs the"
                f" parallelism."
                if not f.partitions_weights() else
                f"- **Dropped `{f.label()}`.**"
            )
        for was, now in self.scaled:
            lines.append(f"- **Narrowed `{was.label()}` to `{now.label()}`.**")
        ratio = self.weight_residency_ratio()
        if ratio > 1.0:
            lines += [
                "",
                f"**A projected rank holds {ratio:g}x the weight bytes a planned one does.**"
                f" That is not a detail: the floorplan's own report chose the wider split"
                f" *because* it lowers per-bank weight residency and decode is bank-bandwidth"
                f" bound. Optimizing against this projection is optimizing the plan the search"
                f" ranked second. It is the right thing to do when the goal is the fastest"
                f" single-device kernel, and the wrong thing to forget when reading the result"
                f" as evidence about the 16-device deployment.",
            ]
        return "\n".join(lines) + "\n"


def _product(values) -> int:
    total = 1
    for v in values:
        total *= int(v)
    return total


def _divisors_descending(n: int) -> list[int]:
    return [d for d in range(n, 0, -1) if n % d == 0]


def project(
    module: str,
    splits: list[dict[str, Any]],
    units: list[str],
    target_units: int = PROJECTION_TARGET_UNITS,
    target: str = "",
) -> Projection:
    """Project one placement onto ``target_units`` logical cores of a single device.

    ``splits`` and ``units`` are the placement's fields as they appear in a floorplan scheme.
    Raises :class:`ProjectionError` only when no combination of dropping and narrowing reaches
    the target — which in practice means a single factor whose dimension has fewer elements than
    the target, and which no module in this artifact hits.
    """
    if target_units < 1:
        raise ProjectionError(f"target_units must be >= 1, got {target_units}")

    planned = [
        Factor(
            dim=str(s.get("dim", "")),
            factor=int(s.get("factor", 1)),
            collective=str(s.get("collective", "allreduce")),
        )
        for s in (splits or [])
    ]
    planned_units = len(units) or _product(f.factor for f in planned) or 1
    planned_devices = len({str(u).split(".")[0] for u in units}) if units else 1

    common = dict(
        module=module, planned=planned, planned_units=planned_units,
        planned_devices=planned_devices, target=target,
    )

    # Already single-device-sized: pass it through untouched. Projecting *up* to fill the device
    # would be inventing parallelism the plan did not ask for.
    if planned_units <= target_units:
        return Projection(projected=list(planned), projected_units=planned_units, **common)

    remaining = list(planned)
    dropped: list[Factor] = []
    scaled: list[tuple[Factor, Factor]] = []

    # Step 1: drop activation-only factors, widest first, while that is still not enough.
    # Widest first because dropping one wide factor loses less structure than several narrow
    # ones, and because it is more likely to reach the target in a single step.
    activation_only = sorted(
        (f for f in remaining if not f.partitions_weights()),
        key=lambda f: -f.factor,
    )
    for f in activation_only:
        if _product(x.factor for x in remaining) <= target_units:
            break
        candidate = [x for x in remaining if x is not f]
        if _product(x.factor for x in candidate) >= target_units or not candidate:
            remaining = candidate
            dropped.append(f)

    # Step 2: narrow the weight-partitioning factors until the product hits the target exactly.
    for index, f in enumerate(list(remaining)):
        product = _product(x.factor for x in remaining)
        if product <= target_units:
            break
        others = product // f.factor
        if others == 0:
            continue
        wanted = max(target_units // others, 1)
        narrowed = next((d for d in _divisors_descending(f.factor) if d <= wanted), 1)
        if narrowed != f.factor:
            replacement = Factor(f.dim, narrowed, f.collective)
            remaining[index] = replacement
            scaled.append((f, replacement))

    # Step 3: drop anything left at factor 1 — a 1-way split is not a split.
    kept: list[Factor] = []
    for f in remaining:
        if f.factor == 1:
            dropped.append(f)
        else:
            kept.append(f)
    remaining = kept

    projected_units = _product(f.factor for f in remaining) or 1
    if projected_units > target_units:
        raise ProjectionError(
            f"{module}: cannot project {planned_units} unit(s) "
            f"({' * '.join(f.label() for f in planned)}) onto {target_units} — "
            f"the narrowest legal combination is {projected_units}"
        )
    # A projection that collapses a parallel placement all the way to one unit is refused rather
    # than accepted. It would build a "submodule" that is the whole module, and the assembly stage
    # would then declare a 1-way cut where it runs `target_units` ranks — failing the declared-cut
    # check several hours later, for a reason decided here. Every factor being coprime with the
    # target is the only way to reach this, and no module in this artifact does.
    if projected_units == 1 and target_units > 1 and planned_units > 1:
        widths = ", ".join(f"{f.dim}x{f.factor}" for f in planned)
        raise ProjectionError(
            f"{module}: projecting {planned_units} unit(s) ({widths}) onto {target_units} would "
            f"discard all of its parallelism — no factor has a divisor above 1 that fits. A "
            f"single-rank 'submodule' is the whole module, so there would be nothing to assemble"
        )
    return Projection(
        projected=remaining, projected_units=projected_units,
        dropped=dropped, scaled=scaled, **common,
    )


def load_scheme(path: str | Path) -> dict[str, Any]:
    """Read a floorplan scheme (`schemes/rank1.yaml`) as a plain mapping.

    Deliberately *not* `floorplan.schema.Floorplan.load`: that validates against the hardware
    model and the partition graph, which would make this pipeline depend on a floorplan project
    being present and probed. Here the scheme is an input document, and the only thing needed
    from it is one module's placement.
    """
    path = Path(path)
    if not path.is_file():
        raise ProjectionError(f"no floorplan scheme at {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ProjectionError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("placements"), list):
        raise ProjectionError(f"{path} has no 'placements' list — is it a floorplan scheme?")
    return data


def project_module(
    scheme_path: str | Path,
    module: str,
    target_units: int = PROJECTION_TARGET_UNITS,
) -> Projection:
    """Project one named module out of a floorplan scheme file.

    Sums the placements when a module appears more than once (a `fraction`-split module), since
    the question here is how wide the module is in total, not how the plan divided the bookkeeping.
    """
    data = load_scheme(scheme_path)
    entries = [p for p in data["placements"] if p.get("module") == module]
    if not entries:
        available = sorted({str(p.get("module")) for p in data["placements"]})
        hint = ", ".join(available[:6])
        raise ProjectionError(
            f"'{module}' is not placed in {scheme_path}. "
            f"{len(available)} module(s) are, e.g. {hint}"
        )
    if len(entries) > 1:
        units = [u for e in entries for u in (e.get("units") or [])]
        # Fractional placements of one module are alternative homes for parts of it, so the
        # widest single entry is the shape one rank sees; the others add units, not width.
        widest = max(entries, key=lambda e: len(e.get("units") or []))
        return project(
            module, widest.get("splits") or [], [str(u) for u in units],
            target_units=target_units, target=str(data.get("target", "")),
        )
    entry = entries[0]
    return project(
        module, entry.get("splits") or [], [str(u) for u in (entry.get("units") or [])],
        target_units=target_units, target=str(data.get("target", "")),
    )
