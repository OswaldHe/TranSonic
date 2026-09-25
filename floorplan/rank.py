# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 4: rank the three best schemes, blind, then attach the numbers afterwards.

The ranking agent does not see the simulator, its traces, or any predicted latency. It sees
the three floorplans, the hardware description, the model's own source, and the partition
graph — and it reasons about which deployment is best from the architecture alone.

This is deliberate and it is the most unusual decision in the pipeline, so it is worth saying
why. The three schemes arrive already ordered by the simulator, and an agent that could see
that order would be reviewing the simulator's answer rather than forming one. Since the
simulator is uncalibrated by construction — no measured module latency was allowed anywhere
near it — a second, independent judgement is worth more than a confirmation of the first. The
disagreements are the most informative output the whole pipeline produces.

So the blinding is structural, not an instruction. `stage()` builds a sandbox directory that
physically lacks `sim/`, `reports/` and every metric, and the schemes are copied in
anonymized and shuffled: `scheme-a`, `scheme-b`, `scheme-c` in an order that carries no
signal. Nothing tells the agent which the simulator preferred.

Two things the agent *is* told, because withholding them would produce a worse ranking rather
than a less biased one: that every scheme passed the feasibility gate, and what that gate
checked. Otherwise it might rank an infeasible scheme first on the theory that it looks fast.

Afterwards, `attach_measurements` appends the simulator's latencies and an agreement table to
the report. The agent never sees that section; a script writes it once the prose is final.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from floorplan.driver import top_schemes

#: Files the blind agent may see, copied into the sandbox.
VISIBLE = ("PLATFORM.md", "MODEL.md", "README.md")

#: Anonymized names, assigned in shuffled order.
LABELS = ("scheme-a", "scheme-b", "scheme-c")


@dataclass
class Staged:
    """A prepared ranking sandbox and the mapping back to real schemes."""

    directory: Path
    #: label -> the candidate entry it came from
    mapping: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save_key(self, path: Path) -> None:
        """Write the de-anonymization key, outside the sandbox."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            label: {
                "iteration": entry["iteration"],
                "metrics": entry["metrics"],
                "score": entry.get("score"),
                "plan": entry["plan"],
            }
            for label, entry in self.mapping.items()
        }, indent=2, sort_keys=True))


def stage(project: Path, destination: Path, seed: int | None = None,
          count: int = len(LABELS)) -> Staged:
    """Build the blind sandbox.

    ``seed`` exists for the tests; leaving it None shuffles from the system entropy, which is
    what a real run wants — a fixed shuffle that the agent could learn would be no shuffle.
    """
    if not 1 <= count <= len(LABELS):
        raise ValueError(
            f"count must be between 1 and {len(LABELS)} (there are only that many labels), "
            f"got {count}"
        )
    schemes = top_schemes(project, count=count)
    if not schemes:
        raise RuntimeError(
            f"no feasible candidates archived under {project / 'schemes' / 'candidates'}. "
            f"`floorplan run` archives one per gate-passing iteration"
        )

    destination = destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "schemes").mkdir(parents=True)

    for relative in VISIBLE:
        source = project / relative
        if source.exists():
            shutil.copyfile(source, destination / relative)

    # The platform description the agent is allowed: the datasheet, with the probe overlay
    # withheld. It needs the hardware's structure and limits; the fitted efficiency
    # coefficients are the simulator's cost model and would leak its reasoning.
    systems = destination / "systems"
    systems.mkdir()
    for candidate in sorted((project / "systems").glob("*.yaml")):
        if candidate.name == "probed.yaml":
            continue
        shutil.copyfile(candidate, systems / candidate.name)

    labels = list(LABELS[:len(schemes)])
    random.Random(seed).shuffle(labels)
    mapping: dict[str, dict[str, Any]] = {}
    for label, entry in zip(labels, schemes):
        mapping[label] = entry
        text = Path(entry["plan"]).read_text()
        # Strip the archive header: it carries the iteration number and the metrics.
        body = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        ).lstrip("\n")
        (destination / "schemes" / f"{label}.yaml").write_text(
            f"# {label}. One of three candidate deployment schemes, presented in arbitrary\n"
            f"# order. Nothing about this file's name or position indicates its quality.\n\n"
            + body
        )

    staged = Staged(directory=destination, mapping=mapping)
    _write_brief(destination, len(schemes))
    return staged


def _write_brief(destination: Path, count: int) -> None:
    """`TASK.md` — what the blind agent is asked to do."""
    (destination / "TASK.md").write_text(f"""\
# Rank {count} deployment schemes for DeepSeek V4.1 Flash on a 16-device Trainium2 instance

`schemes/scheme-*.yaml` are {count} candidate deployments of the same model on the same
hardware. All {count} are **feasible**: an automated gate has already confirmed that each one
places every module on the inference path, splits only along dimensions that exist, keeps every
memory tier within capacity at contexts of both 128 and 8192 tokens, and satisfies every data
dependency. You do not need to re-check any of that.

Your job is to decide **which is the best deployment, and why**, from the architecture.

## What you have

| | |
|---|---|
| `schemes/scheme-*.yaml` | the {count} candidates, in arbitrary order |
| `PLATFORM.md` | the hardware: hierarchy, memory tiers, interconnect, and the numbered text constraints |
| `MODEL.md` | the modules, their sizes, their dependencies, and the legal split dimensions |
| `systems/*.yaml` | the platform datasheet the briefs were generated from |
| the model's own source | at the artifact path given to you, `modules/*/source.py` and `vendor/model.py` |

## What you deliberately do not have

No performance predictions. No simulator, no timings, no profile, no measurement of any kind —
not withheld by request but absent from this directory.

This is on purpose. These schemes were found by a search driven by a simulator that is
**uncalibrated**: it was built without access to any measured latency for this model, so its
absolute numbers are unverified. Your independent architectural judgement is worth more than a
review of its output, and it is only independent if you cannot see it.

So do not speculate about what a simulator would say, and do not try to estimate the latencies
yourself and rank by that. Rank by reasoning about the hardware: what each scheme asks the
machine to do, which limit it runs into first, and which one asks for something the machine is
good at.

The scheme names carry no information. `scheme-a` is not the favourite.

## What to write

One file, `REPORT.md`, in readable but technical prose. It has to answer, clearly:

1. **What is the best scheme?** Name it and commit to it. Then rank the other two, with the
   ranking criteria you used and where the schemes are close enough that the order is a
   judgement call.

2. **How does it distribute and deploy the workload?** Walk through it: which modules go where,
   how the model is divided across the devices and their logical NeuronCores, what the schedule
   is, and where the weights live. Someone who has not read the YAML should be able to follow
   the deployment from your description.

3. **Why is it the best?** This is the core of the report. Argue from the hardware — refer to
   specific numbers and constraints in `PLATFORM.md`: bank capacity, link bandwidths, hop
   counts, the engine exclusions, the tier latencies. "Better load balance" is not an argument;
   "the 24 GiB bank limit forces Engram off-HBM, and this scheme is the only one that puts the
   resulting tier crossing off the critical path of layers 1 and 14" is.

   Say what each scheme's *first limiting factor* is, and why the winner's is the least bad.
   Distinguish prefill from decode: they stress completely different things, and a scheme can
   be right for one and wrong for the other.

4. **Where a module is decomposed, how do you get the pieces?** For every module the winning
   scheme splits, give the recipe concretely:
   - which dimension, and what that means for this module's actual computation
   - how the tensors divide — which axis of which weight, and what each shard holds
   - what has to be communicated to put the result back together, and when
   - which level of the hierarchy each piece lands on: a device, a logical NeuronCore, or the
     two physical NeuronCore-v3 inside one

   Someone implementing this should not have to re-derive the partition from the dimension name.

5. **What would you check first?** The parts of your reasoning that rest on the datasheet's
   less certain numbers, or on an assumption the scheme makes that you cannot verify —
   `hit_rate` on a tiered module is the obvious one. Name them.

Be direct and technical. Do not hedge every claim, and do not pad. If two schemes are
substantially the same deployment with a cosmetic difference, say so rather than manufacturing
a distinction.

## Required: state your rankings on their own lines

Somewhere in `REPORT.md`, put the overall ranking in exactly this form, best first:

    RANKING: scheme-b > scheme-a > scheme-c

This is parsed to decide which plan is published as rank 1, 2 and 3. Without it the ordering
has to be guessed from the order your prose happens to mention the schemes in, which for a
report that introduces all three before choosing is simply wrong.

Then give **one ranking per configuration**, in the same form with the configuration in
brackets:

    RANKING[prefill_128_b1]: scheme-a > scheme-c > scheme-b
    RANKING[prefill_128_b4]: scheme-a > scheme-c > scheme-b

...and so on for every configuration listed in section 6 below. Rank each one on its own
merits: these are genuinely different problems, and a scheme that wins overall need not win
everywhere.

## 6. Which scheme wins each configuration, and why

The deployment is measured at every combination of three axes:

    phase             prefill (ingest a prompt) or decode (produce one token)
    context length    128 or 8192 tokens
    batch size        1, 4, 8 or 32 samples

For each, say which scheme you expect to win and give the reason in a sentence or two. You are
not being asked to guess latencies — you are being asked which scheme's structure suits that
regime, and the axes interact in ways that should drive your answer:

- **Batch reverses the pipeline-depth argument.** At batch 1 a decode step has one token in
  flight, so every stage boundary is a bubble and depth is pure cost. At batch 32 there are 32
  tokens to fill it and the same depth is nearly free. A deeply pipelined scheme should lose the
  batch-1 decode points and may win the batch-32 ones.
- **Batch multiplies the KV cache.** 8192 tokens at batch 32 is 32x the KV of batch 1. A scheme
  that spends its HBM on resident weights has less room for it, and one that tiers a table has
  more.
- **Batch changes expert routing.** At batch 1 a decode step touches a few of the 384 routed
  experts; at batch 32 it touches many more, which shifts the balance between expert
  parallelism and replication.
- **Context length decides whether attention or weight movement dominates**, and that in turn
  decides whether a sequence split or a head split is the better cut.

Where two schemes are effectively identical for a configuration, say so and pick either —
manufacturing a distinction is worse than admitting a tie.

The configurations, by name:

    prefill_128_b1
    prefill_128_b4
    prefill_128_b8
    prefill_128_b32
    prefill_8192_b1
    prefill_8192_b4
    prefill_8192_b8
    prefill_8192_b32
    decode_128_b1
    decode_128_b4
    decode_128_b8
    decode_128_b32
    decode_8192_b1
    decode_8192_b4
    decode_8192_b8
    decode_8192_b32
""")


# ---------------------------------------------------------------------------------------
# Attaching the measurements afterwards
# ---------------------------------------------------------------------------------------
#: The line the ranking report must contain, e.g. `RANKING: scheme-b > scheme-a > scheme-c`.
#:
#: Leading markdown noise is tolerated — `**RANKING**:`, `## RANKING:`, `- RANKING:` — because
#: the report is prose written by an agent told to emphasize its conclusion, and rejecting a
#: bolded heading would silently fall back to guessing the order from first mentions.
RANKING_LINE = re.compile(
    r"^[\s>#*_\-]*RANKING[\s*_]*:\s*(.+)$", re.MULTILINE | re.IGNORECASE,
)

#: The per-configuration form, e.g. `RANKING[decode_8192_b32]: scheme-c > scheme-a > scheme-b`.
#: One per workload the agent is asked to call, so the report says not only which scheme is best
#: overall but where each one wins — which is the question a deployment actually faces.
PER_CONFIG_LINE = re.compile(
    r"^[\s>#*_\-]*RANKING\s*\[\s*([a-z0-9_]+)\s*\][\s*_]*:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_ranking(report: Path, labels: list[str]) -> list[str]:
    """The agent's ranking, read out of an explicit `RANKING:` line.

    Explicit because inferring it from first mentions was wrong for the most ordinary report
    structure there is. A report that introduces all three schemes, compares them, and *then*
    states a winner would have been read in introduction order — publishing the wrong plans as
    `rank1.yaml`..`rank3.yaml` while the prose above them said something else.

    Falls back to first-mention order when the line is absent, so a report that ignores the
    instruction still produces something, but the caller is told the ranking was not explicit.
    """
    if not report.exists():
        return list(labels)
    text = report.read_text()
    match = RANKING_LINE.search(text)
    if match:
        declared = [
            token.strip().lower().strip(".,`*")
            for token in re.split(r">|,|→|->", match.group(1))
        ]
        ordered = [label for label in declared if label in labels]
        # Any label the line omitted keeps its relative order from the prose.
        for label in _first_mention_order(text, labels):
            if label not in ordered:
                ordered.append(label)
        if ordered:
            return ordered
    return _first_mention_order(text, labels)


def ranking_was_explicit(report: Path) -> bool:
    """Whether the report stated its ranking on a `RANKING:` line rather than implying it."""
    return bool(report.exists() and RANKING_LINE.search(report.read_text()))


def parse_per_config_rankings(report: Path, labels: list[str]) -> dict[str, list[str]]:
    """The agent's per-workload rankings, keyed by workload name.

    Absent keys mean the report did not call that configuration, which is reported rather than
    filled in: a missing opinion and an opinion that happens to match the overall order are
    different things, and only one of them is evidence.
    """
    if not report.exists():
        return {}
    text = report.read_text()
    out: dict[str, list[str]] = {}
    for match in PER_CONFIG_LINE.finditer(text):
        configuration = match.group(1).strip().lower()
        declared = [
            token.strip().lower().strip(".,`*")
            for token in re.split(r">|,|→|->", match.group(2))
        ]
        ordered = [label for label in declared if label in labels]
        if ordered:
            out[configuration] = ordered
    return out


def _first_mention_order(text: str, labels: list[str]) -> list[str]:
    lowered = text.lower()
    positions = []
    for label in labels:
        index = lowered.find(label)
        positions.append((index if index >= 0 else len(lowered) + 1, label))
    return [label for _, label in sorted(positions)]


def _ordered_metrics(key: dict[str, Any]) -> list[str]:
    """Metric names in workload order, not alphabetical.

    Alphabetical puts `b1, b32, b4, b8` next to each other, which is the one ordering that
    makes a batch sweep unreadable.
    """
    from floorplan.sim.runner import WORKLOADS

    present = {name for entry in key.values() for name in (entry.get("metrics") or {})}
    ordered = [f"{w.name}_ms" for w in WORKLOADS if f"{w.name}_ms" in present]
    return ordered + sorted(present - set(ordered))


def simulator_ranking_for(key: dict[str, Any], metric: str, labels: list[str]) -> list[str]:
    """The schemes ordered by one metric, best first. Absent values sort last."""
    def value(label: str) -> float:
        metrics = key[label].get("metrics") or {}
        return float(metrics.get(metric, float("inf")))

    return sorted(labels, key=value)


def _per_config_section(
    key: dict[str, Any], metric_names: list[str],
    per_config: dict[str, list[str]], labels: list[str],
) -> list[str]:
    """The per-configuration comparison: who the agent says wins each point, and who does.

    This is what the batch axis is for. A single overall ranking hides the case the deployment
    actually faces — that one scheme is best for long-context prefill at batch 32 and another
    for short-context decode at batch 1 — and the agent is asked to call each point so its
    architectural reasoning can be checked where it is most likely to be regime-dependent.
    """
    lines = [
        "## Per-configuration rankings",
        "",
        "The agent's call for each point beside the simulator's, best first. `—` means the",
        "report did not rank that configuration; a missing opinion and an opinion that happens",
        "to agree are different things and are not conflated here.",
        "",
        "| configuration | architectural analysis | simulator | agree |",
        "|---|---|---|---|",
    ]
    matched = considered = 0
    for metric in metric_names:
        configuration = metric[:-3] if metric.endswith("_ms") else metric
        simulated = simulator_ranking_for(key, metric, labels)
        claimed = per_config.get(configuration)
        if claimed:
            considered += 1
            agrees = claimed[0] == simulated[0]
            matched += int(agrees)
            mark = "✓" if agrees else "✗"
            rendered = " > ".join(f"`{label}`" for label in claimed)
        else:
            mark = "—"
            rendered = "—"
        lines.append(
            f"| `{configuration}` | {rendered} | "
            + " > ".join(f"`{label}`" for label in simulated)
            + f" | {mark} |"
        )

    lines.append("")
    if considered:
        lines.append(
            f"**The agent called {considered} of {len(metric_names)} configurations and picked "
            f"the simulator's winner in {matched} of them.**"
        )
        if matched < considered:
            lines.append("")
            lines.append(
                "Where they differ, neither is authoritative. The simulator's constants are "
                "uncalibrated, and the architectural reading cannot see queueing or pipeline "
                "fill — but a disagreement concentrated in one region of the grid (all the "
                "batch-32 points, say, or only long context) is a specific and checkable claim "
                "about which effect is being missed."
            )
    else:
        lines.append(
            "The report ranked no individual configuration, so there is nothing to compare "
            "point by point. `TASK.md` asks for a `RANKING[<configuration>]:` line per point."
        )
    lines.append("")
    return lines


def attach_measurements(
    report: Path, staged_key: Path, project: Path, agent_order: list[str] | None = None,
) -> str:
    """Append the simulator's numbers and an agreement table to the finished report.

    Written by a script after the agent's prose is final, so the agent's reasoning cannot have
    been shaped by it. The agreement table is the pipeline's most interesting artifact and it
    costs nothing to compute: two independent judgements of the same three schemes, one from
    architecture and one from simulation.
    """
    key = json.loads(staged_key.read_text())
    labels = sorted(key)
    order = agent_order or parse_ranking(report, labels)
    simulator_order = sorted(labels, key=lambda label: key[label].get("score") or float("inf"))

    metric_names = _ordered_metrics(key)
    per_config = parse_per_config_rankings(report, labels)

    lines = [
        "",
        "---",
        "",
        "# Appendix: the simulator's numbers",
        "",
        "*Appended by `floorplan/rank.py` after the ranking above was written. The ranking",
        "agent did not see any of this — it had no access to the simulator, its traces or",
        "these latencies. Two independent judgements of the same schemes.*",
        "",
        "## Simulated latencies",
        "",
        "All milliseconds, lower is better, across phase x context length x batch size.",
        "**Uncalibrated**: the simulator was built without access to any measured latency for",
        "this model, so these numbers are useful for comparing schemes against each other and",
        "not as predictions of wall-clock time.",
        "",
        "| scheme | " + " | ".join(metric_names) + " | score | iteration |",
        "|---" * (len(metric_names) + 3) + "|",
    ]
    for label in labels:
        entry = key[label]
        metrics = entry.get("metrics") or {}
        cells = " | ".join(
            f"{metrics[name]:.3f}" if name in metrics else "—" for name in metric_names
        )
        score = entry.get("score")
        lines.append(
            f"| `{label}` | {cells} | "
            f"{f'{score:.4f}' if score is not None else '—'} | "
            f"{entry.get('iteration', '—')} |"
        )

    lines += [
        "",
        "The score is the mean of every latency normalized against the best value any scheme",
        "achieved for that metric, so it rewards being near the frontier across the whole grid",
        "rather than winning one point.",
        "",
        "## Where the two overall rankings agree",
        "",
        "| rank | architectural analysis | simulator |",
        "|---|---|---|",
    ]
    for index in range(len(labels)):
        architectural = order[index] if index < len(order) else "—"
        simulated = simulator_order[index] if index < len(simulator_order) else "—"
        mark = " ✓" if architectural == simulated else ""
        lines.append(f"| {index + 1} | `{architectural}` | `{simulated}`{mark} |")

    agreement = sum(
        1 for index in range(min(len(order), len(simulator_order)))
        if order[index] == simulator_order[index]
    )
    lines += [
        "",
        f"**{agreement} of {len(labels)} positions agree.**",
        "",
    ]
    lines += _per_config_section(key, metric_names, per_config, labels)
    if order and simulator_order and order[0] == simulator_order[0]:
        lines.append(
            "The two methods pick the same winner. That is a genuine cross-check: an "
            "architectural argument from the datasheet and a discrete-event simulation of the "
            "timeline are not the same evidence, and agreeing on the top scheme means the "
            "result does not rest on the simulator's uncalibrated constants alone."
        )
    else:
        lines.append(
            f"**The two methods disagree on the winner** — architecture prefers "
            f"`{order[0] if order else '?'}`, the simulator prefers "
            f"`{simulator_order[0] if simulator_order else '?'}`. This is the most "
            f"informative outcome available and should not be resolved by preferring one "
            f"automatically. The simulator is uncalibrated; the architectural analysis cannot "
            f"see second-order effects like queueing and pipeline fill. Read the report's "
            f"reasoning against the losing scheme's latencies and decide which argument "
            f"actually applies."
        )

    lines += [
        "",
        "## Provenance of the hardware numbers",
        "",
    ]
    probed = project / "systems" / "probed.yaml"
    if probed.exists():
        overlay = yaml.safe_load(probed.read_text()) or {}
        provenance = overlay.get("_provenance") or {}
        lines.append(
            f"Probed on {provenance.get('probed_on', 'unknown')} at "
            f"{provenance.get('probed_at', 'unknown')}."
        )
        lines.append("")
        not_probeable = provenance.get("not_probeable") or []
        if not_probeable:
            lines.append("Extrapolated rather than measured, because the dev host has one device:")
            lines.append("")
            for item in not_probeable:
                lines.append(f"- `{item}`")
            lines.append("")
        link = ((overlay.get("shared") or {}).get("links") or {}).get("intra_device") or {}
        if str(link.get("source", "")).startswith("derived"):
            lines += [
                f"**`links.intra_device` is derived, not measured.** "
                f"{link.get('derived_from', '')}. {link.get('note', '')}",
                "",
                "Any conclusion that turns on whether tensor parallelism belongs inside a "
                "device or across devices rests on this number, and it is the first thing to "
                "measure on real 16-device hardware.",
                "",
            ]
        failures = provenance.get("failures") or []
        if failures:
            lines.append("Probes that failed:")
            lines.append("")
            for failure in failures:
                lines.append(f"- `{failure['name']}` — {failure['error']}")
            lines.append("")
    else:
        lines.append("No probe overlay found.")

    appendix = "\n".join(lines)
    with report.open("a") as handle:
        handle.write(appendix)
    return appendix
