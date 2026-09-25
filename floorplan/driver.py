# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four stages, and the state that carries between them.

    init    materialize a project: the platform and model briefs,
            the generated baseline floorplan, and the manifest the gate reads
    build   two agent iterations that write the per-module cost models, gated by the
            invariant suite
    run     five agent iterations that edit `floorplan.yaml`, gated by `checker.py` and
            ranked across the workload grid — an ordinary AutoHelix loop
    rank    one blind agent ranks the top three schemes, then a script appends the numbers

The interesting seam is between `build` and `run`. `build` ends by hashing `sim/` into the
manifest, and from that moment the simulator is frozen: `run`'s editable scope is
`floorplan.yaml` alone and check (d) verifies the bytes. So the loop that *writes* the cost
model and the loop that *exploits* it are deliberately different loops, because an agent that
could do both would have every incentive to make the model cheap rather than the plan good.

`rank` is separated for the mirror-image reason. It runs with no access to the simulator, its
traces or its predicted latencies — see `rank.py`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from floorplan import baseline as baseline_module
from floorplan import checker
from floorplan.parser import Hardware, format_bytes, load_system
from floorplan.schema import Floorplan, legal_dims
from floorplan.sim.runner import EXCLUDED_KINDS, deployable_modules, load_graph

#: Where per-project state lives, relative to the project root.
STATE_DIR = Path(".autohelix") / "floorplan"

#: The framework modules a project reads but does not contain. Their bytes are hashed into the
#: manifest by `freeze()` and verified by gate check (d), because these are the files that
#: actually compute the metrics.
FRAMEWORK_MODULES = (
    "sim/engine.py",
    "sim/collectives.py",
    "sim/memory.py",
    "sim/api.py",
    "sim/runner.py",
    "parser.py",
)

#: Feasible candidates are archived here regardless of whether they beat the metric gate, so
#: `rank` has something to choose from even if four of five iterations regressed.
CANDIDATES_DIR = Path("schemes") / "candidates"


class DriverError(RuntimeError):
    """The project is not in a state the requested stage can run from."""


@dataclass
class Manifest:
    """What the gate needs to know that cannot be a fixed file."""

    artifact: str
    target: str
    systems_dir: str
    created: str
    hashes: dict[str, str] = field(default_factory=dict)
    #: Hashes of the *installed* framework modules — the code that computes the metrics. Kept
    #: separately from `hashes` because those are project-relative and these are absolute.
    framework_hashes: dict[str, str] = field(default_factory=dict)
    frozen_at: str = ""
    build_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "target": self.target,
            "systems_dir": self.systems_dir,
            "created": self.created,
            "hashes": self.hashes,
            "framework_hashes": self.framework_hashes,
            "frozen_at": self.frozen_at,
            "build_iterations": self.build_iterations,
        }

    @classmethod
    def load(cls, project: Path) -> "Manifest":
        path = project / STATE_DIR / "manifest.json"
        if not path.exists():
            raise DriverError(f"{project} is not a floorplan project (no {path})")
        data = json.loads(path.read_text())
        return cls(
            artifact=data["artifact"],
            target=data["target"],
            systems_dir=data.get("systems_dir", ""),
            created=data.get("created", ""),
            hashes=data.get("hashes") or {},
            framework_hashes=data.get("framework_hashes") or {},
            frozen_at=data.get("frozen_at", ""),
            build_iterations=int(data.get("build_iterations", 0)),
        )

    def save(self, project: Path) -> None:
        path = project / STATE_DIR / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# ---------------------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------------------
def init(
    project: Path,
    artifact: Path,
    target: str = "trn2-16device",
    systems_dir: Path | None = None,
    force: bool = False,
) -> Manifest:
    """Materialize a floorplan project.

    Materialized rather than pointed at: the project gets its own briefs, baseline and git
    history, so `git log` afterwards is the record of how the floorplan arrived.
    """
    project = project.resolve()
    if project.exists() and any(project.iterdir()):
        if not force:
            raise DriverError(
                f"{project} is not empty. Pass --force to clear it, git history included"
            )
        shutil.rmtree(project)
    project.mkdir(parents=True, exist_ok=True)

    package = Path(__file__).resolve().parent
    resolved_systems = (systems_dir or package / "systems").resolve()

    modules, graph = load_graph(artifact)
    system = load_system(target, resolved_systems)
    hardware = Hardware.from_system(system)

    # `sim/modules/` is the agent's; `sim/framework/` is a read-only copy of the code that
    # computes the metrics, so the project is self-contained for reading and the agent has no
    # reason to go looking in the installed package — where it would find the gates. See
    # `write_framework_reference` for why it is a copy *and* hashed against the installed files.
    (project / "sim" / "modules").mkdir(parents=True, exist_ok=True)
    (project / "sim" / "modules" / "__init__.py").write_text(
        '"""Agent-written cost models, one per module archetype. See ../../README.md."""\n'
    )
    write_framework_reference(project, package)

    # The systems directory travels with the project: the gate re-reads it every iteration,
    # and a project whose platform description could change under it is not reproducible.
    shutil.copytree(resolved_systems, project / "systems", dirs_exist_ok=True)

    (project / "reports").mkdir(exist_ok=True)
    (project / CANDIDATES_DIR).mkdir(parents=True, exist_ok=True)

    write_platform_brief(project, hardware)
    write_model_brief(project, modules, graph, hardware)
    write_readme(project, hardware, modules)

    plan = baseline_module.build(modules, hardware)
    plan.dump(project / "floorplan.yaml", baseline_module.HEADER)

    (project / ".gitignore").write_text(
        ".autohelix/\n__pycache__/\n*.pyc\nreports/trace.json\n"
    )

    manifest = Manifest(
        artifact=str(Path(artifact).resolve()),
        target=target,
        systems_dir=str((project / "systems").resolve()),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    manifest.save(project)

    _git(project, "init", "-q")
    _git(project, "add", "-A")
    _git(project, "-c", "user.email=floorplan@localhost", "-c", "user.name=floorplan",
         "commit", "-q", "-m",
         f"Floorplan project for {hardware.name}: briefs and generated baseline")
    return manifest


def write_framework_reference(project: Path, package: Path) -> None:
    """Copy the framework into `sim/framework/` for reading, and write an index for it.

    Two requirements pull against each other here, and both have bitten.

    Pointing the agent at the *installed package* — which the first version of this did, by
    listing absolute paths — hands it the whole package directory, and an agent that lists that
    directory finds `checker.py` and `invariants.py`: the exploration gate and the build gate,
    both of which the design depends on it not reading. A build run confirmed this immediately.

    Copying the framework and hashing only the copy is the other failure: the simulator runs out
    of the package, so a project-local copy would be the one the agent reads while a different
    one computes every metric.

    So: copy for reading, and have gate check (d) verify all three of the copy, the installed
    files, and that they are byte-identical. The agent then has no reason to look outside the
    project, and nothing can drift.

    This is "no reason to look", not "cannot look" — a determined agent can still import the
    package and read `__file__`. A hard guarantee needs the filesystem isolation `bootstrap/`
    gets from materializing a repo with no package imports at all; this is the honest middle,
    and `README.md` says so.
    """
    destination = project / "sim" / "framework"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in FRAMEWORK_MODULES:
        target = destination / Path(relative).name
        shutil.copyfile(package / relative, target)

    lines = [
        "# The simulator framework",
        "",
        "Read-only reference copies of the code that computes your metrics. You write",
        "`sim/modules/*.py` and `sim/constraints.py`; everything here is frozen.",
        "",
        "| file | what it does |",
        "|---|---|",
        "| `api.py` | **the contract your cost models are written against — start here** |",
        "| `engine.py` | the timeline: engines, dependencies, SBUF exclusion |",
        "| `collectives.py` | what rejoining a split costs, by group shape |",
        "| `memory.py` | where the bytes are, and whether the plan fits |",
        "| `runner.py` | walks the module DAG, once per workload |",
        "| `parser.py` | turns the system YAML into the hardware model |",
        "",
        "These are copies. The simulator executes the installed package, and the gate verifies",
        "that these bytes, those bytes, and the hashes recorded at freeze time all agree — so",
        "what you read here is what runs, and a change to either is reported rather than",
        "silently altering results.",
        "",
    ]
    (destination / "INDEX.md").write_text("\n".join(lines))


def _git(project: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args], cwd=project, capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != 0:
        raise DriverError(
            f"git {' '.join(args)} failed in {project}: "
            f"{(completed.stderr or completed.stdout).strip()[:200]}"
        )


# ---------------------------------------------------------------------------------------
# The briefs
# ---------------------------------------------------------------------------------------
def write_platform_brief(project: Path, hardware: Hardware) -> None:
    """`PLATFORM.md` — the hardware, rendered from the system YAML rather than restated.

    Rendered, not written by hand, because a brief that drifts from the YAML the simulator
    reads is worse than no brief: the agent would be optimizing for a machine that is not the
    one being simulated.
    """
    lines = [
        f"# The platform: {hardware.name}",
        "",
        hardware.describe(),
        "",
        "Generated from `systems/` by `floorplan/driver.py`. The simulator reads the same",
        "file, so this cannot drift from what your floorplan will actually be costed against.",
        "",
        "## Addressing",
        "",
        "A placement names one or more units as `d<device>.l<logical_nc>`:",
        "",
        f"- devices `d0` .. `d{len(hardware.devices) - 1}`",
        f"- logical NeuronCores `l0` .. `l{len(hardware.devices[0].logical_ncs) - 1}` per device",
        f"- **{hardware.unit_count()} units in total**",
        "",
        "Doc terminology, which is worth getting right because the informal usage is inverted:",
        "a *logical NeuronCore* is the group of two, and the two cores inside it are *physical",
        f"NeuronCore-v3*. At LNC={hardware.lnc} the runtime dispatches to the pair, so the pair is",
        "the placement grain. `.p0`/`.p1` may be appended to say how the pair divides one",
        "placement's work; it is not a separate level of the hierarchy.",
        "",
        "## Per logical NeuronCore",
        "",
    ]
    unit = hardware.devices[0].logical_ncs[0]
    lines += [
        f"| HBM bank | **{format_bytes(unit.hbm_bank_bytes)}** — not 96 GiB; that is the device's four banks |",
        "|---|---|",
        f"| bank bandwidth | {unit.hbm_bank_bandwidth_bytes_per_s / 1e9:.0f} GB/s |",
        f"| SBUF | {format_bytes(unit.sbuf_bytes)} (the pair's, not each core's) |",
        f"| PSUM | {format_bytes(unit.psum_bytes)} |",
        f"| physical cores | {unit.physical_cores} |",
        f"| engines | {', '.join(sorted(unit.engines))} |",
        "",
        "## Peak compute, per device",
        "",
        "| dtype | TFLOPS | achieved fraction (probed) |",
        "|---|---|---|",
    ]
    for key, label in (
        ("fp8_flops", "fp8"), ("bf16_flops", "bf16"), ("fp32_flops", "fp32"),
    ):
        value = hardware.compute.get(key)
        if value is None:
            continue
        measured = hardware.efficiency.get(
            "matmul_fp8" if key == "fp8_flops" else "matmul_bf16"
        )
        lines.append(
            f"| {label} | {value / 1e12:.0f} | "
            f"{f'{measured:.3f}' if measured is not None else 'unmeasured'} |"
        )
    lines += [
        "",
        "Divide by four for one logical NeuronCore. The achieved fraction is what the",
        "simulator actually uses; `matmul_small_k` is separate and much lower, which is the",
        "single most important cost-model fact for MoE.",
        "",
        "## Memory tiers",
        "",
        "Capacity is enforced at every tier, per scope. Nearest first:",
        "",
        "| tier | scope | capacity | bandwidth | latency | notes |",
        "|---|---|---|---|---|---|",
    ]
    for name, tier in hardware.tiers.items():
        if not tier.inference_path:
            lines.append(
                f"| `{name}` | {tier.scope} | — | "
                f"{(tier.bandwidth_bytes_per_s or 0) / 1e9:.0f} GB/s | — | "
                f"**load time only** — may not be read during a forward pass |"
            )
            continue
        lines.append(
            f"| `{name}` | {tier.scope} | "
            f"{format_bytes(tier.capacity_bytes) if tier.capacity_bytes else '—'} | "
            f"{(tier.bandwidth_bytes_per_s or 0) / 1e9:.1f} GB/s | "
            f"{tier.latency_us if tier.latency_us is not None else '?'} us | "
            f"{'reached through ' + ', '.join(tier.via) + '; ' if tier.via else ''}"
            f"{f'{tier.random_read_iops:.0f} IOPS' if tier.random_read_iops else ''} |"
        )

    lines += [
        "",
        "## Interconnect",
        "",
        f"Topology: **{hardware.topology_shape[0]}x{hardware.topology_shape[1]} "
        f"{'torus' if hardware.topology_wrap else 'mesh'}**"
        if hardware.topology_kind != "single" else "Topology: a single device.",
        "",
    ]
    if hardware.topology_kind != "single":
        lines += [
            "Hop counts are not uniform, and a parallel group's shape on the torus is part of",
            "the design:",
            "",
            "```",
        ]
        for device in range(0, min(len(hardware.devices), 16), 4):
            row = "  ".join(
                f"d0->d{other}: {hardware.hops(0, other)}"
                for other in range(device, min(device + 4, len(hardware.devices)))
            )
            lines.append(f"  {row}")
        lines.append("```")
    lines += [
        "",
        f"- intra-device (core to core in one device): "
        f"{(hardware.intra_device_bandwidth_bytes_per_s or 0) / 1e9:.0f} GB/s, "
        f"{hardware.intra_device_latency_us} us",
        f"- inter-device (over the torus): "
        f"{(hardware.inter_device_bandwidth_bytes_per_s or 0) / 1e9:.0f} GB/s, "
        f"{hardware.inter_device_hop_latency_us} us per hop",
        "",
        "## Hardware features and constraints",
        "",
        "Prose, because these are not numbers. Each is implemented somewhere in `sim/` and",
        "cited by number; several of them decide whole strategies rather than trimming a",
        "percent.",
        "",
        hardware.constraints_text,
        "",
        "## Provenance",
        "",
        "Which numbers are measured and which are extrapolated, from `systems/probed.yaml`:",
        "",
    ]
    probed = project / "systems" / "probed.yaml"
    if probed.exists():
        overlay = yaml.safe_load(probed.read_text()) or {}
        provenance = overlay.get("_provenance", {})
        for item in provenance.get("not_probeable", []):
            lines.append(f"- **extrapolated, not measured:** `{item}`")
        if provenance.get("not_probeable_reason"):
            lines += ["", provenance["not_probeable_reason"], ""]
        link = ((overlay.get("shared") or {}).get("links") or {}).get("intra_device") or {}
        if link.get("source", "").startswith("derived"):
            lines += [
                f"- **`links.intra_device` is derived, not measured** "
                f"({link.get('derived_from', '')}). {link.get('note', '')}",
                "",
            ]
    else:
        lines.append("- no probe has run; the simulator will refuse to start.")
    (project / "PLATFORM.md").write_text("\n".join(lines) + "\n")


def write_model_brief(
    project: Path, modules: dict[str, dict[str, Any]], graph: dict[str, Any],
    hardware: Hardware,
) -> None:
    """`MODEL.md` — what has to be placed, how big it is, and what it depends on."""
    required = deployable_modules(modules)
    total_params = sum(int(modules[m].get("param_bytes") or 0) for m in required)

    by_kind: dict[str, tuple[int, int]] = {}
    for module_id in required:
        entry = modules[module_id]
        kind = str(entry.get("kind"))
        count, nbytes = by_kind.get(kind, (0, 0))
        by_kind[kind] = (count + 1, nbytes + int(entry.get("param_bytes") or 0))

    biggest = sorted(
        required, key=lambda m: -int(modules[m].get("param_bytes") or 0),
    )[:12]

    lines = [
        f"# The model: {graph.get('model')}",
        "",
        f"Revision `{graph.get('revision')}`, {graph.get('num_layers')} layers, "
        f"stored `{graph.get('dtype')}`.",
        "",
        f"**{len(required)} modules must be placed**, totalling "
        f"{format_bytes(total_params)} of parameters. "
        f"{', '.join(sorted(EXCLUDED_KINDS))} is excluded — it is not on the "
        f"`tokens -> logits` path — and placing it is an error.",
        "",
        f"For scale: the instance holds "
        f"{format_bytes(sum(d.hbm_bytes for d in hardware.devices))} of device memory across "
        f"{hardware.unit_count()} banks of "
        f"{format_bytes(hardware.devices[0].logical_ncs[0].hbm_bank_bytes)}.",
        "",
        "## By kind",
        "",
        "| kind | modules | parameters | share |",
        "|---|---|---|---|",
    ]
    for kind, (count, nbytes) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        lines.append(
            f"| {kind} | {count} | {format_bytes(nbytes)} | "
            f"{nbytes / total_params * 100:.1f}% |"
        )

    lines += [
        "",
        "## The largest modules",
        "",
        "These are where placement decisions are won and lost.",
        "",
        "| module | kind | parameters | activations | splittable along |",
        "|---|---|---|---|---|",
    ]
    for module_id in biggest:
        entry = modules[module_id]
        dims = ", ".join(sorted(legal_dims(module_id, entry)))
        lines.append(
            f"| `{module_id}` | {entry.get('kind')} | "
            f"{format_bytes(int(entry.get('param_bytes') or 0))} | "
            f"{format_bytes(int(entry.get('activation_bytes') or 0))} | {dims} |"
        )

    bank = hardware.devices[0].logical_ncs[0].hbm_bank_bytes
    oversized = [
        m for m in required if int(modules[m].get("param_bytes") or 0) > bank
    ]
    lines += [
        "",
        "## Modules that do not fit one bank",
        "",
    ]
    if oversized:
        lines += [
            f"{len(oversized)} module(s) exceed a single {format_bytes(bank)} bank, so each",
            "must be split, tiered, or placed off-HBM. There is no placement that avoids the",
            "question:",
            "",
        ]
        for module_id in sorted(oversized, key=lambda m: -int(modules[m]["param_bytes"])):
            nbytes = int(modules[module_id]["param_bytes"])
            lines.append(
                f"- `{module_id}` — {format_bytes(nbytes)}, "
                f"{nbytes / bank:.1f}x a bank "
                f"(a {hardware.unit_count()}-way split still leaves "
                f"{format_bytes(nbytes / hardware.unit_count())} per unit)"
            )
    else:
        lines.append("None.")

    lines += [
        "",
        "## Partition dimensions",
        "",
        "The closed vocabulary the simulator understands. A split along a dimension the",
        "module does not have is rejected — not as a bad idea but as a meaningless one, since",
        "it would produce shards of size zero.",
        "",
        "| dimension | meaning |",
        "|---|---|",
        "| `head` | attention heads |",
        "| `hidden` | the model dim, or an MLP's intermediate dim |",
        "| `expert` | MoE routed experts |",
        "| `seq` | sequence positions (context/sequence parallel) |",
        "| `batch` | batch members (data parallel) — free, no collective |",
        "| `layer` | whole layers, for a module spanning several (pipeline parallel) |",
        "| `vocab` | vocabulary, for embed and lm_head |",
        "| `ngram` | Engram n-gram table rows |",
        "",
        "## Data dependencies",
        "",
        "Taken from the partition graph: a module consuming a tensor waits for whoever",
        "produces it. You do not declare dependencies — they are the model's — but the",
        "schedule you choose has to be consistent with them, and the `stage` field is how you",
        "say what runs when.",
        "",
        f"The graph has {len(modules)} modules and "
        f"{len(graph.get('tensors') or [])} named tensors, entry `"
        f"{', '.join(graph.get('entry_tensors') or [])}` and output `"
        f"{', '.join(graph.get('output_tensors') or [])}`.",
        "",
    ]
    (project / "MODEL.md").write_text("\n".join(lines) + "\n")


def write_readme(project: Path, hardware: Hardware, modules: dict[str, Any]) -> None:
    """`README.md` — the schema for the one editable file."""
    template = Path(__file__).resolve().parent / "templates" / "project_readme.md"
    text = template.read_text()
    text = text.replace("{{TARGET}}", hardware.name)
    text = text.replace("{{UNITS}}", str(hardware.unit_count()))
    text = text.replace("{{DEVICES}}", str(len(hardware.devices)))
    text = text.replace("{{PER_DEVICE}}", str(len(hardware.devices[0].logical_ncs)))
    text = text.replace("{{MODULES}}", str(len(deployable_modules(modules))))
    text = text.replace(
        "{{BANK}}", format_bytes(hardware.devices[0].logical_ncs[0].hbm_bank_bytes),
    )
    (project / "README.md").write_text(text)


# ---------------------------------------------------------------------------------------
# build: two agent iterations that write the cost models
# ---------------------------------------------------------------------------------------
#
# Driven directly rather than through the AutoHelix harness, and the reason is the same one
# `bootstrap/` documents: the harness discards a rejected iteration's worktree, which is
# correct when a working codebase is being improved and wrong here. At iteration 0 the cost
# model registry is empty, so the invariant suite is red by construction; a discarding loop
# would throw away iteration 1's work and start iteration 2 from nothing, forever.
#
# `bootstrap` solves this by subclassing `Harness` to merge anyway, but it needs the rest of
# the harness — metric ranking, parallel workers, worktree isolation between competing
# iterations. `build` needs none of that: there is no metric, no competition, and nothing to
# isolate, because the two iterations are meant to build on each other. What is left once
# those are removed is a linear two-step process, and writing it as one is shorter and easier
# to follow than overriding a loop into not looping.
#
# `run` is the opposite case and does use the harness unchanged: by then the baseline is green
# and iterations genuinely do compete.
# ---------------------------------------------------------------------------------------
@dataclass
class BuildOutcome:
    """What one build iteration produced."""

    iteration: int
    agent_ok: bool
    invariants_passed: bool
    checks: list[Any] = field(default_factory=list)
    review: str = ""
    verdict: str = ""
    error: str = ""


def build_loop(
    project: Path,
    iterations: int = 2,
    on_event=None,
) -> list[BuildOutcome]:
    """Run the cost-model loop. Returns one outcome per iteration.

    Each iteration: render the prompt (the preset's goal, plus the previous iteration's
    review), run the agent in the project directory, run the invariant suite, run the
    reviewer, commit. Work is kept whether or not the suite passes.
    """
    from autohelix.agents import AgentConfig, create_agent
    from autohelix.config import Config

    from floorplan import invariants

    manifest = Manifest.load(project)
    preset = yaml.safe_load((Path(__file__).resolve().parent / "build_preset.yaml").read_text())
    config = Config.from_dict(preset)
    agent = create_agent(config.agent)
    reviewer_config = AgentConfig(
        type=config.agent.type,
        model=(config.reviewer.model if config.reviewer else None),
    )
    reviewer = create_agent(reviewer_config)

    state = project / STATE_DIR
    state.mkdir(parents=True, exist_ok=True)
    outcomes: list[BuildOutcome] = []
    previous_review = ""

    for iteration in range(1, iterations + 1):
        def emit(message: str) -> None:
            if on_event:
                on_event(iteration, message)

        emit("agent starting")
        prompt = _build_prompt(config.goal, manifest, previous_review, iteration, iterations)
        log = state / f"build-iter-{iteration}.log"
        result = agent.run(
            worktree_path=project, prompt=prompt, iteration=iteration,
            log_path=log, event_callback=lambda event: None, project_path=project,
        )
        outcome = BuildOutcome(
            iteration=iteration, agent_ok=result.success, invariants_passed=False,
            error=result.error or "",
        )

        emit("running the invariant suite")
        try:
            report = invariants.run(
                project, Path(manifest.artifact), project / "systems", manifest.target,
            )
            outcome.invariants_passed = report.passed
            outcome.checks = report.checks
            (state / f"build-invariants-{iteration}.json").write_text(
                json.dumps(report.to_dict(), indent=2),
            )
        except Exception as exc:                        # noqa: BLE001 — recorded, not hidden
            outcome.error = f"{type(exc).__name__}: {exc}"

        emit("running the reviewer")
        # The reviewer is held to being read-only by snapshot-and-restore, not by instruction.
        # It gets the same writable project and is required to write a report into it, so
        # nothing stops it editing `sim/` too — and those edits would land after the invariant
        # suite had already passed, get committed, and be frozen on the strength of a green
        # result describing different code. The verdict recorded beside a commit has to
        # describe the simulator that actually landed.
        snapshot = _snapshot_sim(project)
        try:
            review_log = state / f"build-review-{iteration}.log"
            reviewer.run(
                worktree_path=project,
                prompt=_reviewer_prompt(config, iteration, state),
                iteration=iteration, log_path=review_log,
                event_callback=lambda event: None, project_path=project,
            )
            review_path = project / "reports" / f"build-review-{iteration}.md"
            outcome.review = review_path.read_text() if review_path.exists() else ""
            outcome.verdict = _parse_verdict(outcome.review)
            previous_review = outcome.review
        except Exception as exc:                        # noqa: BLE001
            outcome.error = (outcome.error + f" reviewer: {exc}").strip()
        finally:
            reverted = _restore_sim(project, snapshot)
            if reverted:
                outcome.error = (
                    outcome.error
                    + f" reviewer edited the simulator and was reverted: "
                      f"{', '.join(reverted[:4])}"
                ).strip()
                emit(f"reverted {len(reverted)} reviewer edit(s) to sim/")

        _commit_build(project, iteration, outcome)
        outcomes.append(outcome)
        emit(
            f"invariants {'PASS' if outcome.invariants_passed else 'FAIL'}"
            + (f", verdict {outcome.verdict}" if outcome.verdict else "")
        )
        if outcome.invariants_passed and outcome.verdict != "circumventing":
            break

    return outcomes


def _build_prompt(
    goal: str, manifest: Manifest, previous_review: str, iteration: int, total: int,
) -> str:
    """The build agent's prompt: the goal, where the artifact is, and the last review."""
    parts = [
        goal,
        "",
        "---",
        "",
        f"This is iteration {iteration} of {total}. Your work is kept whether or not the "
        f"checks pass, so build on what is already in `sim/` rather than restarting.",
        "",
        f"The partition artifact is at `{manifest.artifact}`. Read the module sources there; "
        f"you may read anything under that path.",
        "",
        f"The target platform is `{manifest.target}`, described in `PLATFORM.md` and defined "
        f"in `systems/{manifest.target}.yaml`.",
    ]
    if previous_review:
        parts += [
            "",
            "## The previous iteration's review",
            "",
            "A reviewer read your last iteration against the checks and wrote this. It is "
            "your brief — start from what it says is left rather than re-deriving it.",
            "",
            previous_review,
        ]
    return "\n".join(parts)


def _reviewer_prompt(config: Any, iteration: int, state: Path) -> str:
    """The reviewer's prompt: the preset's text plus where to write and what to read."""
    verdict_path = state / f"build-invariants-{iteration}.json"
    return "\n".join([
        config.reviewer.prompt if config.reviewer else "Review this iteration.",
        "",
        "---",
        "",
        f"The invariant suite's verdict for this iteration is at `{verdict_path}`.",
        "",
        f"Write your review to `reports/build-review-{iteration}.md` and change nothing else.",
    ])


def _snapshot_sim(project: Path) -> dict[Path, str]:
    """Contents of every agent-writable simulator file, for restoring after the reviewer."""
    snapshot: dict[Path, str] = {}
    for path in sorted((project / "sim").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            snapshot[path] = path.read_text()
        except OSError:
            continue
    return snapshot


def _restore_sim(project: Path, snapshot: dict[Path, str]) -> list[str]:
    """Put back anything the reviewer changed. Returns the paths it had to restore."""
    reverted: list[str] = []
    for path, content in snapshot.items():
        try:
            if path.read_text() != content:
                path.write_text(content)
                reverted.append(str(path.relative_to(project)))
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            reverted.append(str(path.relative_to(project)))
        except OSError:
            continue
    # A file the reviewer *added* is also an edit to the simulator.
    for path in sorted((project / "sim").rglob("*.py")):
        if "__pycache__" in path.parts or path in snapshot:
            continue
        try:
            path.unlink()
            reverted.append(str(path.relative_to(project)))
        except OSError:
            continue
    return reverted


def _parse_verdict(review: str) -> str:
    """The `VERDICT:` line the reviewer is asked to end with."""
    for line in reversed(review.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            return stripped.split(":", 1)[1].strip().lower()
    return ""


def _commit_build(project: Path, iteration: int, outcome: BuildOutcome) -> None:
    """Commit the iteration's work, passing or not. Never fails the loop."""
    try:
        _git(project, "add", "-A")
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project,
            capture_output=True, text=True, timeout=120,
        )
        if not status.stdout.strip():
            return
        passing = sum(1 for check in outcome.checks if getattr(check, "passed", False))
        message = (
            f"build iteration {iteration}: cost models"
            f" ({passing}/{len(outcome.checks)} invariants"
            f"{', ' + outcome.verdict if outcome.verdict else ''})"
        )
        _git(project, "-c", "user.email=floorplan@localhost", "-c", "user.name=floorplan",
             "commit", "-q", "-m", message)
    except DriverError:
        pass


# ---------------------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------------------
def freeze(project: Path, iterations: int = 0) -> Manifest:
    """Hash the simulator into the manifest. After this, only `floorplan.yaml` may change.

    Two sets of hashes, because the simulator lives in two places: the agent-written cost
    models and platform YAML in the project, and the framework in the installed package.
    Recording only the first would leave the code that computes every metric unverified.

    The moment the build loop hands over to the exploration loop. Separated into its own
    function because `build` calls it on success and an operator may need to call it by hand
    after fixing a cost model.
    """
    manifest = Manifest.load(project)
    manifest.hashes = checker.hash_tree(project)
    manifest.framework_hashes = checker.hash_framework()
    manifest.frozen_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest.build_iterations = iterations or manifest.build_iterations
    manifest.save(project)
    return manifest


# ---------------------------------------------------------------------------------------
# candidate archive
# ---------------------------------------------------------------------------------------
def archive_candidate(
    project: Path, iteration: int, metrics: dict[str, float], accepted: bool,
) -> Path:
    """Keep a feasible floorplan whether or not it beat the metric gate.

    The metric gate governs what the *next* iteration builds on; it does not govern what gets
    retained. Five iterations under a four-metric ratchet could plausibly accept one and
    reject four, and `rank` would then have nothing to rank — so anything that passed the gate
    is archived here with its numbers, and `rank` chooses from the archive.
    """
    destination = project / CANDIDATES_DIR / f"iter-{iteration}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan_text = (project / "floorplan.yaml").read_text()
    header = [
        f"# Candidate from iteration {iteration}"
        f" ({'accepted' if accepted else 'rejected by the metric gate'}).",
        "#",
    ]
    for name, value in sorted(metrics.items()):
        header.append(f"#   {name}: {value:.4f}")
    header.append("")
    destination.write_text("\n".join(header) + "\n" + plan_text)
    (destination.with_suffix(".json")).write_text(json.dumps({
        "iteration": iteration,
        "metrics": metrics,
        "accepted": accepted,
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    return destination


def candidates(project: Path) -> list[dict[str, Any]]:
    """Every archived candidate, with its metrics, newest last."""
    out: list[dict[str, Any]] = []
    for path in sorted((project / CANDIDATES_DIR).glob("iter-*.json")):
        payload = json.loads(path.read_text())
        payload["plan"] = str(path.with_suffix(".yaml"))
        out.append(payload)
    return sorted(out, key=lambda entry: entry["iteration"])


def rank_score(metrics: dict[str, float], best: dict[str, float]) -> float:
    """Mean of the four normalized ratios; lower is better.

    Normalized against the best value each metric reached, so a scheme is scored on how close
    it comes to the frontier on every metric rather than on a sum of milliseconds — which
    would be dominated by `prefill_8192_ms` and effectively ignore decode.
    """
    ratios = [
        metrics[name] / best[name]
        for name in sorted(best)
        if name in metrics and best.get(name)
    ]
    return sum(ratios) / len(ratios) if ratios else float("inf")


def top_schemes(project: Path, count: int = 3) -> list[dict[str, Any]]:
    """The best `count` archived candidates by `rank_score`."""
    archive = candidates(project)
    if not archive:
        return []
    names = sorted({name for entry in archive for name in entry["metrics"]})
    best = {
        name: min(
            entry["metrics"][name] for entry in archive if name in entry["metrics"]
        )
        for name in names
    }
    scored = sorted(archive, key=lambda entry: rank_score(entry["metrics"], best))
    for entry in scored:
        entry["score"] = rank_score(entry["metrics"], best)
    return scored[:count]
