# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The discrete-event kernel: ops in, a timeline out.

The division of labour in this simulator is worth stating plainly, because it is what
keeps an agent-written cost model reviewable. A cost model says *what work exists* — "this
attention shard is a 3.2 ms tensor-engine op, then a 0.4 ms vector op, then an allreduce of
10 MiB". This module decides *when it runs*: which engine it occupies, what it waits for,
and what it contends with. A cost model that wants to make something look fast has to
misstate a duration, which a reviewer can check against arithmetic, rather than quietly
assuming an overlap the hardware does not offer.

Scheduling is list scheduling over a DAG, not a queue-based event loop, because every
duration here is analytic — nothing is data-dependent, so there is no reason to discover
the timeline incrementally. Ops are submitted in topological order and each one starts at
the latest of: its dependencies finishing, its engine falling idle, and any exclusive
resource it needs being released. That makes a run deterministic by construction, which
the invariant suite checks by simulating twice and comparing bit patterns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

#: Engines an op may occupy. The five NKI exposes, plus ``cc`` for the collective-compute
#: cores, which are a per-device pool rather than a per-unit engine.
ENGINES = frozenset({"tensor", "vector", "scalar", "gpsimd", "dma", "cc"})

#: Exclusive resources an op may hold on its unit. ``sbuf`` is the one the platform's text
#: constraints need: GPSIMD and the tensor engine cannot both be in SBUF, so ops that
#: declare it serialize against one another even though their engines differ.
RESOURCES = frozenset({"sbuf", "psum"})


class ScheduleError(RuntimeError):
    """An op cannot be scheduled: bad engine, bad resource, or a dependency not yet run."""


@dataclass(frozen=True)
class Op:
    """One unit of work on one engine.

    ``seconds`` is the cost model's output and this module never adjusts it. ``deps`` are
    indices into the ops already submitted, which is what forces submission to be
    topological: an op cannot depend on something that does not exist yet.
    """

    name: str
    unit: str
    engine: str
    seconds: float
    module: str = ""
    deps: tuple[int, ...] = ()
    #: Exclusive resources held for the op's whole duration.
    holds: frozenset[str] = frozenset()
    #: Bytes moved, for the trace's communication accounting. Not used for timing — the
    #: cost model has already converted bytes to seconds — but reported, so a reviewer can
    #: check one against the other.
    bytes_moved: int = 0
    #: For a collective: which units took part, so per-link volume can be attributed.
    participants: tuple[str, ...] = ()
    #: Set when the cost model asked for this op to overlap with compute. A collective with
    #: this set does not block its unit's compute engines; its dependents still wait for it.
    overlapped: bool = False


@dataclass
class Scheduled:
    """An op with its resolved start and finish."""

    index: int
    op: Op
    start: float
    finish: float


@dataclass
class Trace:
    """Everything a run recorded. The ranking report's evidence, and the metric's source."""

    scheduled: list[Scheduled] = field(default_factory=list)
    #: Busy seconds per (unit, engine).
    engine_busy: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    #: Bytes crossing each link class, keyed ``intra_device`` / ``inter_device`` / ``tier:<name>``.
    link_bytes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #: Seconds spent in collectives, whether or not overlapped.
    collective_seconds: float = 0.0
    #: Seconds of collective time that was credited as overlapped with compute.
    overlapped_seconds: float = 0.0

    def makespan(self) -> float:
        """Wall-clock seconds from the first start to the last finish."""
        return max((s.finish for s in self.scheduled), default=0.0)

    def critical_path(self) -> list[Scheduled]:
        """The chain of ops that sets the makespan, latest-finishing first then unwound.

        Walks back through whichever dependency finished last, which is the chain a reader
        of the report wants: it is the list of things that would have to get faster.
        """
        if not self.scheduled:
            return []
        by_index = {s.index: s for s in self.scheduled}
        current = max(self.scheduled, key=lambda s: s.finish)
        chain = [current]
        while current.op.deps:
            predecessors = [by_index[d] for d in current.op.deps if d in by_index]
            if not predecessors:
                break
            current = max(predecessors, key=lambda s: s.finish)
            chain.append(current)
        chain.reverse()
        return chain

    def utilization(self) -> dict[str, float]:
        """Fraction of the makespan each engine class was busy, averaged over units."""
        span = self.makespan()
        if span <= 0:
            return {}
        totals: dict[str, float] = defaultdict(float)
        units: dict[str, set[str]] = defaultdict(set)
        for (unit, engine), busy in self.engine_busy.items():
            totals[engine] += busy
            units[engine].add(unit)
        return {
            engine: totals[engine] / (span * len(units[engine]))
            for engine in totals if units[engine]
        }

    def by_module(self) -> dict[str, float]:
        """Busy seconds attributed to each module, summed over engines and units.

        Sums rather than spans, so it exceeds the makespan when work runs in parallel.
        It answers "where does the work live", which is what the report needs; the
        makespan answers "how long does it take".
        """
        totals: dict[str, float] = defaultdict(float)
        for entry in self.scheduled:
            if entry.op.module:
                totals[entry.op.module] += entry.op.seconds
        return dict(totals)

    def busiest_unit(self) -> tuple[str, float]:
        """The unit with the most busy seconds, and how many. The load-balance tell."""
        totals: dict[str, float] = defaultdict(float)
        for (unit, _engine), busy in self.engine_busy.items():
            totals[unit] += busy
        if not totals:
            return ("", 0.0)
        unit = max(totals, key=lambda u: totals[u])
        return (unit, totals[unit])


class Schedule:
    """Accumulates ops and resolves each one's start time as it arrives."""

    def __init__(self) -> None:
        self._ops: list[Op] = []
        self._finish: list[float] = []
        self._engine_free: dict[tuple[str, str], float] = defaultdict(float)
        self._resource_free: dict[tuple[str, str], float] = defaultdict(float)
        self.trace = Trace()

    def __len__(self) -> int:
        return len(self._ops)

    def submit(self, op: Op) -> int:
        """Schedule one op and return its index, for use as a later op's dependency."""
        if op.engine not in ENGINES:
            raise ScheduleError(
                f"op '{op.name}': unknown engine '{op.engine}'. "
                f"Known: {', '.join(sorted(ENGINES))}"
            )
        unknown = set(op.holds) - RESOURCES
        if unknown:
            raise ScheduleError(
                f"op '{op.name}': unknown resource(s) {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(RESOURCES))}"
            )
        if op.seconds < 0:
            raise ScheduleError(f"op '{op.name}': negative duration {op.seconds}")
        for dep in op.deps:
            if not 0 <= dep < len(self._ops):
                raise ScheduleError(
                    f"op '{op.name}': depends on op {dep}, which has not been submitted. "
                    f"Ops must be submitted in dependency order"
                )

        start = 0.0
        for dep in op.deps:
            start = max(start, self._finish[dep])

        # An overlapped collective runs on the CC cores and does not wait for, or block,
        # the unit's compute engines. Its dependents still wait for it to finish, which is
        # constraint 4: overlap is real, but a collective and the compute consuming it
        # cannot overlap with each other.
        engine_key = (op.unit, op.engine)
        if not op.overlapped:
            start = max(start, self._engine_free[engine_key])
        for resource in op.holds:
            start = max(start, self._resource_free[(op.unit, resource)])

        finish = start + op.seconds
        if not op.overlapped:
            self._engine_free[engine_key] = finish
        for resource in op.holds:
            self._resource_free[(op.unit, resource)] = finish

        index = len(self._ops)
        self._ops.append(op)
        self._finish.append(finish)

        self.trace.scheduled.append(Scheduled(index=index, op=op, start=start, finish=finish))
        self.trace.engine_busy[engine_key] += op.seconds
        if op.engine == "cc":
            self.trace.collective_seconds += op.seconds
            if op.overlapped:
                self.trace.overlapped_seconds += op.seconds
        return index

    def finish_of(self, index: int) -> float:
        return self._finish[index]

    def last_index(self) -> int:
        if not self._ops:
            raise ScheduleError("no ops submitted")
        return len(self._ops) - 1

    def record_bytes(self, link: str, nbytes: int) -> None:
        """Attribute moved bytes to a link class, for the trace's communication table."""
        self.trace.link_bytes[link] += int(nbytes)
