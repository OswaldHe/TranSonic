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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from floorplan.driver import top_schemes

#: Files the blind agent may see, copied into the sandbox.
VISIBLE = ("PLATFORM.md", "MODEL.md", "README.md")

#: Directories that must NOT reach the sandbox. The simulator and anything it wrote.
WITHHELD = ("sim", "reports", "schemes", ".autohelix", "systems/probed.yaml")

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


def stage(project: Path, destination: Path, seed: int | None = None) -> Staged:
    """Build the blind sandbox.

    ``seed`` exists for the tests; leaving it None shuffles from the system entropy, which is
    what a real run wants — a fixed shuffle that the agent could learn would be no shuffle.
    """
    schemes = top_schemes(project, count=len(LABELS))
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
""")


# ---------------------------------------------------------------------------------------
# Attaching the measurements afterwards
# ---------------------------------------------------------------------------------------
def parse_ranking(report: Path, labels: list[str]) -> list[str]:
    """The agent's ranking, read out of its report.

    Looks for the first occurrence of each label, ordered by position, which is robust to the
    report's shape: whether it declares a winner up front or builds to one, the winner is
    named before the runners-up. A label never mentioned goes last.
    """
    if not report.exists():
        return list(labels)
    text = report.read_text().lower()
    positions = []
    for label in labels:
        index = text.find(label)
        positions.append((index if index >= 0 else len(text) + 1, label))
    return [label for _, label in sorted(positions)]


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

    metric_names = sorted({
        name for entry in key.values() for name in (entry.get("metrics") or {})
    })

    lines = [
        "",
        "---",
        "",
        "# Appendix: the simulator's numbers",
        "",
        "*Appended by `floorplan/rank.py` after the ranking above was written. The ranking",
        "agent did not see any of this — it had no access to the simulator, its traces or",
        "these latencies. Two independent judgements of the same three schemes.*",
        "",
        "## Simulated latencies",
        "",
        "All milliseconds, batch 1, lower is better. **Uncalibrated**: the simulator was built",
        "without access to any measured latency for this model, so these numbers are useful for",
        "comparing schemes against each other and not as predictions of wall-clock time.",
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
        "The score is the mean of the four latencies each normalized against the best value",
        "any scheme achieved for that metric, so it rewards being near the frontier on all",
        "four rather than winning one.",
        "",
        "## Where the two rankings agree",
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
