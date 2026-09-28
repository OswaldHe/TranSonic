# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix optimize` — make one bootstrapped module fast on a single Trainium device."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from optimization import presets
from optimization.config import ConfigError, PipelineConfig
from optimization.driver import Pipeline, StageError

console = Console()

#: The default config filename, looked for in the working directory.
DEFAULT_CONFIG = "optimization.yaml"


def _load(config: str | None) -> PipelineConfig:
    path = Path(config) if config else Path(DEFAULT_CONFIG)
    if not path.is_file():
        raise click.ClickException(
            f"no config at {path}. Write one with `autohelix optimize template > "
            f"{DEFAULT_CONFIG}` and fill it in."
        )
    try:
        return PipelineConfig.load(path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _pipeline(config: str | None, verbose: bool) -> Pipeline:
    return Pipeline(_load(config), console=console, verbose=verbose)


def _run(stage_name: str, fn) -> int:
    try:
        outcome = fn()
        # A stage that returned `ok=False` reported a failure without raising, and this used to
        # discard it: the command exited 0 and `optimize all` carried on to the next stage.
        if getattr(outcome, "ok", True) is False:
            console.print(f"\n[red]{stage_name} did not succeed:[/red] "
                          f"{getattr(outcome, 'detail', 'no detail')}")
            return 1
    except StageError as exc:
        console.print(f"\n[red]{stage_name} stopped:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted during {stage_name}. Progress is saved.[/yellow]")
        return 130
    return 0


_config_option = click.option(
    "--config", "-c", default=None,
    help=f"the pipeline config (default: ./{DEFAULT_CONFIG})",
)
_verbose_option = click.option("--verbose", "-v", is_flag=True, help="stream agent output")


@click.group(name="optimize")
def optimize() -> None:
    """Optimize one bootstrapped module: cut it to one rank, speed that up, reassemble it.

    Six stages. Run them one at a time while finding your footing, or `all` once you trust the
    config:

    \b
      init       validate the config and project the floorplan placement onto one device
      submodule  an agent cuts the module down to the part one NeuronCore runs
      run        the optimization loop on that rank, under the per-iteration constraint schedule
      assemble   an agent rebuilds all the ranks, rejoined with nki.collectives
      run-full   the optimization loop on the whole module
      feedback   an agent reads both loops' notes and reports what stopped them getting faster

    \b
    And then, once you have read the notes and written what they taught you into the config:
      rerun-full  another round of run-full, starting from the kernel the last round produced

    See `optimization/README.md` for what each stage checks and why the seams are where they are.
    """


@optimize.command()
def template() -> None:
    """Print the config template. Redirect it to optimization.yaml and fill it in."""
    click.echo(presets.config_template())


@optimize.command()
@_config_option
def check(config: str | None) -> None:
    """Validate the config and show the projection, without creating anything."""
    pipeline = _pipeline(config, verbose=False)
    problems = pipeline.config.validate()
    if problems:
        for problem in problems:
            console.print(f"  [red]FAIL[/red] {problem}")
        console.print(f"\n[red]{len(problems)} problem(s).[/red]")
        raise SystemExit(1)
    console.print("  [green]OK[/green] the config is complete")
    for warning in pipeline.config.warnings():
        console.print(f"  [yellow]note[/yellow] {warning}")
    try:
        projection = pipeline.projection()
    except StageError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise SystemExit(1) from exc
    console.print(projection.describe())
    console.print(f"  submodule repo: {pipeline.config.submodule_repo}")
    console.print(f"  full repo:      {pipeline.config.full_repo}")
    for stage in ("submodule", "full"):
        spec = getattr(pipeline.config, stage)
        console.print(f"\n  {stage}: {spec.iterations} iteration(s)")
        console.print("  " + spec.schedule.summary_table().replace("\n", "\n  "))


@optimize.command()
@_config_option
@_verbose_option
def init(config: str | None, verbose: bool) -> None:
    """Validate the config, project the placement, and make the workspace."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("init", pipeline.init))


@optimize.command()
@_config_option
@_verbose_option
def submodule(config: str | None, verbose: bool) -> None:
    """Have an agent cut the module down to the part one NeuronCore runs."""
    pipeline = _pipeline(config, verbose)

    def go() -> None:
        pipeline.init()
        pipeline.submodule()

    raise SystemExit(_run("submodule", go))


@optimize.command(name="compile-constraints")
@_config_option
@_verbose_option
@click.option("--stage", type=click.Choice(["submodule", "full"]), default="submodule")
def compile_constraints(config: str | None, verbose: bool, stage: str) -> None:
    """Turn each constraint slot's prose into a checker script, once, before the loop."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("compile-constraints", lambda: pipeline.compile_constraints(stage)))


@optimize.command()
@_config_option
@_verbose_option
def run(config: str | None, verbose: bool) -> None:
    """Run the optimization loop on the single-rank submodule."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("run", lambda: pipeline.run_loop("submodule")))


@optimize.command()
@_config_option
@_verbose_option
def assemble(config: str | None, verbose: bool) -> None:
    """Have an agent rebuild the whole module across all ranks, rejoined with nki.collectives."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("assemble", pipeline.assemble))


@optimize.command(name="run-full")
@_config_option
@_verbose_option
def run_full(config: str | None, verbose: bool) -> None:
    """Run the optimization loop on the whole distributed module."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("run-full", lambda: pipeline.run_loop("full")))


@optimize.command(name="rerun-full")
@_config_option
@_verbose_option
@click.option("--from-commit", default=None,
              help="start from this commit instead of the last round's best")
@click.option("--note", default="",
              help="why this round exists, recorded with the previous round's state")
def rerun_full_stage(config: str | None, verbose: bool,
                     from_commit: str | None, note: str) -> None:
    """Run another round of the whole-module loop, starting from the kernel you already have.

    For the case the first round creates: you read its notes, you learn what blocked it, you write
    that into `full.iteration_constraints`, and you want the loop to carry on from its own best
    kernel rather than from the assembly baseline. The previous round's history moves to
    `.autohelix/rounds/round-N/`; its notes and reviews stay where the next agent will read them.
    """
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run(
        "rerun-full", lambda: pipeline.rerun_full(from_commit=from_commit, note=note),
    ))


@optimize.command()
@_config_option
@_verbose_option
def feedback(config: str | None, verbose: bool) -> None:
    """Have an agent read both loops' notes and reviews and report what blocked them."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("feedback", pipeline.feedback))


@optimize.command(name="all")
@_config_option
@_verbose_option
def run_all(config: str | None, verbose: bool) -> None:
    """Run all six stages in order."""
    pipeline = _pipeline(config, verbose)
    raise SystemExit(_run("the pipeline", pipeline.all))


@optimize.command()
@_config_option
@click.option("--stage", type=click.Choice(["submodule", "full"]), default=None,
              help="only this stage's gate")
def gate(config: str | None, stage: str | None) -> None:
    """Run a gate once, by hand, against the repo as it stands."""
    pipeline = _pipeline(config, verbose=False)
    stages = [stage] if stage else ["submodule", "full"]
    failed = False
    ran = 0
    for name in stages:
        repo = (pipeline.config.submodule_repo if name == "submodule"
                else pipeline.config.full_repo)
        if not repo.is_dir():
            # A missing repo is a failure when it was asked for by name, and only a note when the
            # command is the sweep over both. Exiting 0 for "nothing existed to check" would let a
            # CI step treat an absent artifact as a validated one.
            if stage:
                console.print(f"  [red]FAIL[/red] {name}: {repo} does not exist")
                failed = True
            else:
                console.print(f"  [dim]{name}: {repo} does not exist yet[/dim]")
            continue
        module = (f"optimization.{'submodule_checker' if name == 'submodule' else 'module_checker'}")
        ok, _ = pipeline._run_gate(repo, module, name)
        failed = failed or not ok
        ran += 1
    if not ran and not stage:
        console.print("  [red]FAIL[/red] neither repo exists, so no gate ran")
        failed = True
    raise SystemExit(1 if failed else 0)


@optimize.command()
@_config_option
def report(config: str | None) -> None:
    """Write the run's report: what each stage achieved and what it cost."""
    from optimization.report import write_report

    pipeline = _pipeline(config, verbose=False)
    path = write_report(pipeline.config, console=console)
    console.print(f"\n[green]Wrote[/green] {path}")


if __name__ == "__main__":
    sys.exit(optimize())
