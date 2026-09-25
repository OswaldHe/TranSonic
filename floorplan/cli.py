# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix floorplan` — decide what runs where on a Trainium instance.

    autohelix floorplan probe                          # measure the primitives, once per host
    autohelix floorplan init <project> --artifact <dir> # materialize a project
    autohelix floorplan build --path <project>         # 2 iterations: write the cost models
    autohelix floorplan run   --path <project>         # 5 iterations: search for a floorplan
    autohelix floorplan rank  --path <project>         # blind ranking + the final report
    autohelix floorplan all   <project> --artifact <dir>  # probe, init, build, run, rank

    autohelix floorplan check  --path <project>        # run the gate once, by hand
    autohelix floorplan report --path <project>        # per-iteration verdicts
    autohelix floorplan show   --path <project>        # what the current floorplan says
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import click

from floorplan import checker, driver, invariants, rank as rank_module
from floorplan.parser import Hardware, format_bytes
from floorplan.schema import Floorplan

#: Where the presets live. Fixed files read straight from the package, like bootstrap's — so
#: the preset you read in a diff is the one that runs, and editing it changes the next run.
PACKAGE = Path(__file__).resolve().parent
BUILD_PRESET = PACKAGE / "build_preset.yaml"
RUN_PRESET = PACKAGE / "preset.yaml"
SYSTEMS = PACKAGE / "systems"

DEFAULT_ARTIFACT = Path("/home/ubuntu/workspace/partition-artifact")


@click.group(name="floorplan")
def floorplan() -> None:
    """Decide how to distribute a model across a Trainium instance's hierarchy."""


# ---------------------------------------------------------------------------------------
@floorplan.command()
@click.option("--systems", type=click.Path(path_type=Path), default=None,
              help="the systems/ directory to write probed.yaml into")
@click.option("--scratch", type=click.Path(path_type=Path), default=None,
              help="where the storage probe may write (nothing persists)")
@click.option("--skip", multiple=True,
              type=click.Choice(["device", "storage", "collective"]),
              help="skip a probe group (repeatable)")
def probe(systems: Path | None, scratch: Path | None, skip: tuple[str, ...]) -> None:
    """Measure the primitives on this host and write systems/probed.yaml.

    Run once per machine. The simulator refuses to start while any efficiency coefficient is
    unmeasured, because substituting peak would make arithmetic free and rank every scheme on
    communication alone.
    """
    from floorplan.probe import suite

    directory = (systems or SYSTEMS).resolve()
    click.echo(f"probing this host; writing {directory / 'probed.yaml'}")
    click.echo("(each probe traces and compiles an NKI kernel — this takes a few minutes)\n")
    started = time.monotonic()
    results = suite.run(
        systems_dir=directory,
        scratch=(scratch or Path("/tmp/floorplan-probe")).resolve(),
        skip=frozenset(skip),
    )
    click.echo(suite.report(results))
    click.echo(f"\nprobed in {time.monotonic() - started:.0f}s")

    hardware = Hardware.load("trn2-16device", directory)
    unresolved = hardware.unresolved()
    if unresolved:
        click.echo(
            f"\n{len(unresolved)} value(s) still unmeasured; the simulator will refuse:\n  "
            + "\n  ".join(unresolved)
        )
        raise SystemExit(1)
    click.echo("\nevery value the simulator needs is now resolved.")


# ---------------------------------------------------------------------------------------
@floorplan.command()
@click.argument("project", type=click.Path(path_type=Path))
@click.option("--artifact", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_ARTIFACT, show_default=True,
              help="the partition artifact holding plan/partition_graph.yaml")
@click.option("--target", default="trn2-16device", show_default=True,
              help="which system YAML to deploy against")
@click.option("--systems", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True, help="clear a non-empty target first, git history included")
def init(project: Path, artifact: Path, target: str, systems: Path | None, force: bool) -> None:
    """Materialize a floorplan project: framework, briefs and the generated baseline."""
    manifest = driver.init(project, artifact, target, systems or SYSTEMS, force)
    hardware = Hardware.load(target, Path(manifest.systems_dir))
    plan = Floorplan.load(project / "floorplan.yaml")
    click.echo(f"{project.resolve()}")
    click.echo(f"  target     {hardware.describe()}")
    click.echo(f"  artifact   {manifest.artifact}")
    click.echo(f"  baseline   {len(plan.placements)} placements over "
               f"{len(plan.units_used())} unit(s)")
    click.echo(f"\nnext: autohelix floorplan build --path {project}")


# ---------------------------------------------------------------------------------------
def _harness(project: Path, preset: Path, environment: dict[str, str] | None = None) -> int:
    """Run the AutoHelix harness in-process against a preset.

    In-process rather than shelling out to `autohelix run`, so the live display, the history
    and the exit status are the harness's own. `FLOORPLAN_ARTIFACT` has to be exported rather
    than passed, because the preset's constraint command references it — that keeps the preset
    a fixed file with no per-project path substituted into it.
    """
    from autohelix.harness import AutoHelixRunError, Harness

    for key, value in (environment or {}).items():
        os.environ[key] = value
    try:
        harness = Harness(project, verbose=False, config_file=preset)
        harness.run()
    except AutoHelixRunError as exc:
        raise click.ClickException(str(exc)) from exc
    return 0


@floorplan.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="the floorplan project")
@click.option("--iterations", "-n", type=int, default=2, show_default=True,
              help="how many build iterations to allow")
@click.option("--no-freeze", is_flag=True,
              help="do not hash sim/ into the manifest when the loop finishes")
def build(path: Path, iterations: int, no_freeze: bool) -> None:
    """Two agent iterations that write the per-module cost models.

    On success the simulator is frozen: `sim/` and `systems/` are hashed into the manifest and
    the exploration loop's gate verifies those bytes every iteration.
    """
    project = path.resolve()
    manifest = driver.Manifest.load(project)
    click.echo(f"building cost models for {manifest.target} in {project}")
    click.echo(f"artifact: {manifest.artifact}\n")

    def progress(iteration: int, message: str) -> None:
        click.echo(f"  iteration {iteration}: {message}")

    outcomes = driver.build_loop(project, iterations=iterations, on_event=progress)

    click.echo()
    final = outcomes[-1] if outcomes else None
    if final is None:
        raise click.ClickException("the build loop produced no iterations")
    for check in final.checks:
        click.echo(f"  [{check.check}] {'PASS' if check.passed else 'FAIL'}  {check.title}")
        for finding in check.findings:
            click.echo(f"          - {str(finding).splitlines()[0][:140]}")
    if final.error:
        click.echo(f"\n  note: {final.error[:300]}")

    if not final.invariants_passed:
        click.echo("\nthe invariant suite is not green; the simulator is NOT frozen.")
        click.echo(
            "A simulator that fails its own invariants would mislead the search, so "
            "exploration should not start from here. Fix sim/ and re-run `build`, or freeze "
            "deliberately with --no-freeze off after fixing by hand."
        )
        raise SystemExit(1)
    # A green suite is necessary and not sufficient. `build_loop` moves on when a reviewer says
    # `circumventing`, but if the *last* allowed iteration says it there is nowhere to move on
    # to, and checking only the invariant flag would freeze a simulator the reviewer explicitly
    # rejected — turning an exhausted budget into the metric the whole search is scored by.
    if final.verdict == "circumventing":
        click.echo(
            f"\nthe invariant suite is green but the reviewer's verdict on the final "
            f"iteration is `circumventing`; the simulator is NOT frozen."
        )
        click.echo(
            "Read reports/build-review-*.md. Exhausting the iteration budget is not a reason "
            "to accept a cost model the reviewer judged to be gaming its own checks."
        )
        raise SystemExit(1)
    if not no_freeze:
        frozen = driver.freeze(project, iterations=len(outcomes))
        click.echo(f"\nsimulator frozen: {len(frozen.hashes)} file(s) hashed at "
                   f"{frozen.frozen_at}")
    click.echo(f"\nnext: autohelix floorplan run --path {project}")


# ---------------------------------------------------------------------------------------
@floorplan.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None,
              help="an alternative run preset")
def run(path: Path, config: Path | None) -> None:
    """Five agent iterations that edit floorplan.yaml, ranked by four latencies."""
    project = path.resolve()
    manifest = driver.Manifest.load(project)
    if not manifest.hashes:
        raise click.ClickException(
            "the simulator is not frozen — run `autohelix floorplan build` first. "
            "Without the hashes, gate check (d) cannot tell whether a cost model was edited "
            "to make a floorplan look fast"
        )
    code = _harness(project, config or RUN_PRESET, {"FLOORPLAN_ARTIFACT": manifest.artifact})
    _archive_from_history(project)
    schemes = driver.top_schemes(project)
    click.echo()
    if schemes:
        click.echo(f"{len(schemes)} candidate(s) retained:")
        for index, entry in enumerate(schemes, 1):
            metrics = ", ".join(
                f"{name.replace('_ms', '')} {value:.1f}"
                for name, value in sorted(entry["metrics"].items())
            )
            click.echo(f"  {index}. iter-{entry['iteration']}  score "
                       f"{entry['score']:.4f}  {metrics}")
        click.echo(f"\nnext: autohelix floorplan rank --path {project}")
    else:
        click.echo("no feasible candidate was archived; nothing to rank.")
    raise SystemExit(code if code else 0)


def _archive_from_history(project: Path) -> int:
    """Archive every gate-passing iteration's floorplan from the AutoHelix history.

    Done after the loop rather than during it because AutoHelix discards a rejected
    iteration's worktree, so each plan has to be recovered from git afterwards. That is what
    makes a rejected-but-feasible scheme available to `rank` instead of lost.

    Recovered by the **commit sha the history records for that iteration**. The first version
    of this guessed at branch names (`autohelix/iter-N`, `iter-N`) and fell back to `HEAD`,
    which was silently catastrophic: AutoHelix merges an accepted iteration and deletes its
    branch, so no guess ever resolved and every candidate got the *final* plan. Six archived
    files, six different sets of metrics in their headers, one identical floorplan — and
    `rank` then ranked the same scheme three times without anything noticing. A plan that
    cannot be recovered is now skipped, which loses a candidate instead of inventing one.
    """
    history = project / ".autohelix" / "history.jsonl"
    if not history.exists():
        return 0
    archived = 0
    for line in history.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        iteration = int(entry.get("iteration", 0))
        metrics = entry.get("metrics") or {}
        if not metrics:
            continue
        destination = project / driver.CANDIDATES_DIR / f"iter-{iteration}.yaml"
        if destination.exists():
            continue
        # Iteration 0 is the baseline, which has no commit of its own: `init` committed it as
        # the project's root commit. Any *other* entry without a commit was rejected and never
        # merged, so its plan is not in git at all — the gate's own `--archive` is what captures
        # those, during the iteration, before the worktree is deleted. Falling back to the root
        # commit here would file the baseline under a rejected iteration's metrics, which is the
        # same class of mistake as the `HEAD` fallback this replaced.
        commit = entry.get("commit") or (_root_commit(project) if iteration == 0 else None)
        text = _plan_at_commit(project, commit) if commit else None
        if text is None:
            click.echo(
                f"  note: iteration {iteration}'s floorplan is not in git "
                f"({'rejected, never merged' if not entry.get('commit') else 'commit missing'}); "
                f"relying on the gate's own archive for it"
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = [f"# Candidate from iteration {iteration}"
                  f" ({'accepted' if entry.get('accepted') else 'gate passed, metric gate rejected'}).",
                  "#"]
        for name, value in sorted(metrics.items()):
            header.append(f"#   {name}: {float(value):.4f}")
        destination.write_text("\n".join(header) + "\n\n" + text)
        destination.with_suffix(".json").write_text(json.dumps({
            "iteration": iteration,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "accepted": bool(entry.get("accepted")),
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        archived += 1
    return archived


def _plan_at_commit(project: Path, commit: str) -> str | None:
    """`floorplan.yaml` as of one commit. None if that commit has no such file.

    Deliberately no fallback: an unrecoverable plan must read as absent, never as some other
    iteration's plan.
    """
    completed = subprocess.run(
        ["git", "show", f"{commit}:floorplan.yaml"],
        cwd=project, capture_output=True, text=True, timeout=120,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout
    return None


def _root_commit(project: Path) -> str | None:
    """The project's first commit, which is the one `init` made with the baseline."""
    completed = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=project, capture_output=True, text=True, timeout=120,
    )
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


# ---------------------------------------------------------------------------------------
@floorplan.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True)
@click.option("--count", type=int, default=3, show_default=True,
              help="how many schemes to rank")
@click.option("--model", default=None,
              help="model id for the ranking agent (default: the CLI's own default). "
                   "Note this is a *model id* like claude-opus-5, not an AutoHelix agent "
                   "type — passing 'claude' here is rejected by the CLI as an invalid model.")
def rank(path: Path, count: int, model: str | None) -> None:
    """Rank the top schemes with a blind agent, then attach the simulator's numbers.

    The agent runs in a sandbox that physically lacks the simulator, its traces and every
    predicted latency — so its ranking is an independent architectural judgement rather than a
    review of the search's answer. The numbers are appended afterwards by a script.
    """
    project = path.resolve()
    manifest = driver.Manifest.load(project)
    sandbox = project / "reports" / "ranking"
    staged = rank_module.stage(project, sandbox, count=count)
    key = project / ".autohelix" / "floorplan" / "ranking_key.json"
    staged.save_key(key)

    click.echo(f"staged {len(staged.mapping)} scheme(s) in {sandbox}, anonymized and shuffled")
    click.echo(f"key written to {key} (outside the sandbox)")

    prompt = (
        f"Read TASK.md and do what it says. Write REPORT.md in this directory.\n\n"
        f"The model's own source is at {manifest.artifact} — its `modules/*/source.py` and "
        f"`vendor/model.py` are the authority on what each module computes. You may read "
        f"anything under that path.\n\n"
        f"You have no performance data and that is deliberate; TASK.md explains why. Rank by "
        f"reasoning about the hardware."
    )
    command = ["claude", "-p", prompt, "--permission-mode", "acceptEdits",
               "--add-dir", manifest.artifact]
    if model:
        command += ["--model", model]
    click.echo("\nrunning the blind ranking agent...\n")
    completed = subprocess.run(command, cwd=str(sandbox))

    report = sandbox / "REPORT.md"
    if not report.exists():
        raise click.ClickException(
            f"the ranking agent wrote no REPORT.md in {sandbox} "
            f"(exit {completed.returncode})"
        )
    order = rank_module.parse_ranking(report, sorted(staged.mapping))
    rank_module.attach_measurements(report, key, project, order)

    # Publish: the agent's ranking decides the filenames, which is the point of step 4.
    schemes_dir = project / "schemes"
    schemes_dir.mkdir(exist_ok=True)
    for position, label in enumerate(order, 1):
        entry = staged.mapping[label]
        destination = schemes_dir / f"rank{position}.yaml"
        source = Path(entry["plan"])
        text = source.read_text()
        destination.write_text(
            f"# Rank {position} of {len(order)}, by the blind architectural analysis in "
            f"reports/REPORT.md.\n"
            f"# Presented to that analysis as `{label}`; produced by iteration "
            f"{entry['iteration']}.\n"
            f"#\n" + text
        )
    final = project / "REPORT.md"
    final.write_text(report.read_text())
    click.echo(f"\nranking: {' > '.join(order)}")
    click.echo(f"  {final}")
    for position in range(1, len(order) + 1):
        click.echo(f"  {schemes_dir / f'rank{position}.yaml'}")


# ---------------------------------------------------------------------------------------
@floorplan.command(name="all")
@click.argument("project", type=click.Path(path_type=Path))
@click.option("--artifact", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_ARTIFACT, show_default=True)
@click.option("--target", default="trn2-16device", show_default=True)
@click.option("--skip-probe", is_flag=True, help="reuse an existing systems/probed.yaml")
@click.option("--force", is_flag=True)
@click.pass_context
def run_all(ctx: click.Context, project: Path, artifact: Path, target: str,
            skip_probe: bool, force: bool) -> None:
    """probe, init, build, run, rank — the whole pipeline, in order."""
    if not skip_probe:
        ctx.invoke(probe, systems=SYSTEMS, scratch=None, skip=())
    ctx.invoke(init, project=project, artifact=artifact, target=target,
               systems=SYSTEMS, force=force)
    try:
        ctx.invoke(build, path=project, iterations=2, no_freeze=False)
    except SystemExit as exit_code:
        if exit_code.code:
            raise click.ClickException(
                "build did not leave a green invariant suite; stopping before exploration. "
                "A simulator that fails its own invariants would mislead the search"
            ) from exit_code
    try:
        ctx.invoke(run, path=project, config=None)
    except SystemExit:
        pass
    ctx.invoke(rank, path=project, count=3, model=None)


# ---------------------------------------------------------------------------------------
@floorplan.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True)
@click.option("--timeout", type=int, default=900, show_default=True)
def check(path: Path, timeout: int) -> None:
    """Run the gate once, by hand."""
    raise SystemExit(checker.main(["--repo", str(path.resolve()), "--timeout", str(timeout)]))


@floorplan.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True)
def report(path: Path) -> None:
    """Print each iteration's verdict and metrics."""
    project = path.resolve()
    history = project / ".autohelix" / "history.jsonl"
    if not history.exists():
        raise click.ClickException(f"no history at {history}")
    for line in history.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        metrics = entry.get("metrics") or {}
        rendered = "  ".join(
            f"{name.replace('_ms', '')}={float(value):.1f}"
            for name, value in sorted(metrics.items())
        )
        label = "baseline" if entry.get("iteration") == 0 else f"iter-{entry.get('iteration')}"
        state = "accept" if entry.get("accepted") else "reject"
        click.echo(f"{label:>10}  {state}  {rendered}  {entry.get('reason', '')}")


@floorplan.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True)
@click.option("--plan", type=click.Path(path_type=Path), default=None,
              help="a specific floorplan (default: the project's floorplan.yaml)")
def show(path: Path, plan: Path | None) -> None:
    """Summarize what a floorplan says, without simulating it."""
    project = path.resolve()
    manifest = driver.Manifest.load(project)
    parsed = Floorplan.load(plan or project / "floorplan.yaml")
    hardware = Hardware.load(parsed.target, Path(manifest.systems_dir))
    modules, _ = __import__(
        "floorplan.sim.runner", fromlist=["load_graph"],
    ).load_graph(Path(manifest.artifact))

    click.echo(hardware.describe())
    click.echo(f"{len(parsed.placements)} placement(s), "
               f"{len(parsed.modules())} module(s), "
               f"{len(parsed.units_used())} unit(s) in use")
    click.echo(f"runtime: chunk {parsed.runtime.prefill_chunk_tokens}, "
               f"micro-batch {parsed.runtime.decode_micro_batch}, "
               f"pipelined {parsed.runtime.pipeline_chunks}")

    by_tier: dict[str, int] = {}
    by_dim: dict[str, int] = {}
    for placement in parsed.placements:
        nbytes = int((modules.get(placement.module) or {}).get("param_bytes") or 0)
        by_tier[placement.residency.tier] = by_tier.get(placement.residency.tier, 0) + nbytes
        for split in placement.splits:
            by_dim[f"{split.dim} x{split.factor}"] = by_dim.get(
                f"{split.dim} x{split.factor}", 0,
            ) + 1
    click.echo("\nweights by tier:")
    for tier, nbytes in sorted(by_tier.items(), key=lambda kv: -kv[1]):
        click.echo(f"  {tier:<12} {format_bytes(nbytes)}")
    click.echo("\nsplits:")
    for label, count in sorted(by_dim.items(), key=lambda kv: -kv[1]):
        click.echo(f"  {label:<16} {count} placement(s)")


if __name__ == "__main__":
    floorplan()
