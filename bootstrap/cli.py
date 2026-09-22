# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix bootstrap` — turn a partition module into a repo and loop on it.

    autohelix bootstrap init  <repo> --artifact <dir> --module <id>   # build the repo
    autohelix bootstrap run   --path <repo>                           # loop on it
    autohelix bootstrap check --path <repo>                           # gate it once, by hand
    autohelix bootstrap report --path <repo>                          # what the gate last said
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from bootstrap import materialize as mat

#: The test target: layer 0's attention is the one pure sliding-window attention in its
#: group (compress_ratio 0 — no compressor, no indexer, no compressed-KV concatenation), so
#: it is the smallest correct kernel the group admits.
DEFAULT_MODULE = "layers.0.attention"

#: Seconds `inference.py` gets inside the gate, and seconds the gate gets inside the loop.
#: The outer number is the one the preset was specified with; the inner is smaller so a
#: hung device run comes back as a failing check.
DEFAULT_RUN_TIMEOUT = 900
DEFAULT_CONSTRAINT_TIMEOUT = 1200


@click.group(name="bootstrap")
def bootstrap() -> None:
    """Bootstrap a Trainium NKI kernel for one partition-artifact module."""


@bootstrap.command()
@click.argument("repo", type=click.Path(path_type=Path))
@click.option("--artifact", required=True, type=click.Path(exists=True, path_type=Path),
              help="The downloaded partition artifact (the directory holding modules/)")
@click.option("--module", "module_id", default=DEFAULT_MODULE, show_default=True,
              help="Which module of the artifact to bootstrap")
@click.option("--group", default=None,
              help="The implementation group holding it (found from --module if omitted)")
@click.option("--sample", default=None, help="Recorded sample id (the run's first by default)")
@click.option("--step", type=int, default=None, help="Decode step (prefill by default)")
@click.option("--call", "call_index", type=int, default=0, show_default=True,
              help="Which invocation, when the pass called the group more than once")
@click.option("--iterations", type=int, default=10, show_default=True, help="Iteration budget")
@click.option("--iteration-time", default=None, help="Per-iteration deadline, e.g. 45m")
@click.option("--agent", "agent_type", default="claude", show_default=True,
              help="Agent backend: claude, codex, opencode, mock")
@click.option("--model", default=None, help="Override the agent's model")
@click.option("--run-timeout", type=int, default=DEFAULT_RUN_TIMEOUT, show_default=True,
              help="Seconds inference.py gets inside the gate")
@click.option("--constraint-timeout", type=int, default=DEFAULT_CONSTRAINT_TIMEOUT,
              show_default=True, help="Seconds the gate gets")
@click.option("--force", is_flag=True, help="Overwrite a non-empty target directory")
def init(
    repo: Path, artifact: Path, module_id: str, group: str | None, sample: str | None,
    step: int | None, call_index: int, iterations: int, iteration_time: str | None,
    agent_type: str, model: str | None, run_timeout: int, constraint_timeout: int, force: bool,
) -> None:
    """Materialize a module of ARTIFACT into REPO as a git repo to bootstrap in."""
    repo = repo.resolve()
    if repo.exists() and any(repo.iterdir()) and not force:
        raise click.ClickException(f"{repo} is not empty; pass --force to overwrite")

    try:
        resolved_group = group or mat.find_group(artifact, module_id)
    except mat.MaterializeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"artifact : {artifact}")
    click.echo(f"group    : {resolved_group}")
    click.echo(f"module   : {module_id}")
    click.echo(f"into     : {repo}")
    click.echo("materializing tensors (this reads the fetched checkpoint shards)...")

    try:
        result = mat.materialize(
            artifact=artifact.resolve(), group=resolved_group, module_id=module_id, repo=repo,
            sample_id=sample, step=step, call_index=call_index,
        )
        config_path = mat.install_run_state(
            result,
            python=sys.executable,
            iterations=iterations,
            iteration_time=iteration_time,
            run_timeout=run_timeout,
            constraint_timeout=constraint_timeout,
            agent_type=agent_type,
            model=model,
        )
        head = mat.git_init(repo)
    except mat.MaterializeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("")
    click.echo(f"  sample   : {result.sample_id}#{result.step} (invocation {result.call_index})")
    click.echo(f"  chain    : {' -> '.join(s for s in result.submodules if s)}")
    click.echo(f"  tensors  : {len(result.tensors)} file(s), "
               f"{result.total_bytes / (1 << 20):.1f} MiB")
    click.echo(f"  config   : {config_path.relative_to(repo)}")
    click.echo(f"  baseline : {head}")
    click.echo("")
    click.echo(f"Next: autohelix bootstrap run --path {repo}")


@bootstrap.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="The module repo")
@click.option("--iterations", "-n", type=int, default=None, help="Override the iteration budget")
@click.option("--verbose", is_flag=True, help="Show the agent prompt and per-phase detail")
def run(path: Path, iterations: int | None, verbose: bool) -> None:
    """Loop on a bootstrap repo until the gate passes or the budget runs out."""
    from bootstrap.driver import BootstrapLoop

    repo = path.resolve()
    if not (repo / mat.CONFIG_REL).is_file():
        raise click.ClickException(
            f"{repo} has no bootstrap config; run `autohelix bootstrap init` first"
        )
    loop = BootstrapLoop(repo, verbose=verbose)
    passed = loop.run(max_iterations=iterations)
    raise SystemExit(0 if passed else 1)


@bootstrap.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="The module repo")
@click.option("--timeout", type=int, default=DEFAULT_RUN_TIMEOUT, show_default=True,
              help="Seconds inference.py gets")
def check(path: Path, timeout: int) -> None:
    """Run the gate once against the repo as it stands, outside the loop."""
    repo = path.resolve()
    manifest = repo / mat.MANIFEST_REL
    if not manifest.is_file():
        raise click.ClickException(
            f"{repo} has no bootstrap manifest; run `autohelix bootstrap init` first"
        )
    completed = subprocess.run(
        [sys.executable, "-m", "bootstrap.nki_checker", "--repo", str(repo),
         "--manifest", str(manifest), "--timeout", str(timeout)],
        cwd=repo,
    )
    raise SystemExit(completed.returncode)


@bootstrap.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="The module repo")
def report(path: Path) -> None:
    """Print the gate's verdict for each iteration so far."""
    repo = path.resolve()
    reports = sorted(
        (repo / ".autohelix" / "bootstrap").glob("iter-*.json"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not reports:
        raise click.ClickException(f"no verdicts recorded under {repo}/.autohelix/bootstrap/")
    for entry in reports:
        payload = json.loads(entry.read_text())
        checks = payload.get("checks") or []
        passing = [c["check"] for c in checks if c.get("passed")]
        failing = [c["check"] for c in checks if not c.get("passed")]
        label = "baseline" if entry.stem == "iter-0" else entry.stem
        state = "PASS" if payload.get("passed") else f"failing {', '.join(failing) or '?'}"
        click.echo(f"{label:>10}  {len(passing)}/{len(checks)}  {state}")


if __name__ == "__main__":
    bootstrap()
