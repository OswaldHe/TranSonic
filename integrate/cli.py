# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`autohelix integrate` — drive a repo to a state where its constraints hold."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from integrate import presets
from integrate.config import DEFAULT_CONFIG_NAME, ConfigError, IntegrateConfig
from integrate.driver import IntegratePass, StageError

console = Console()

#: The config filename, looked for in the project directory. Not `autohelix.yaml`: a repo may
#: already have one of those for a plain `autohelix run`.
DEFAULT_CONFIG = DEFAULT_CONFIG_NAME


def _config_path(path: str, config: str | None) -> Path:
    return Path(config) if config else Path(path) / DEFAULT_CONFIG


def _load(path: str, config: str | None) -> IntegrateConfig:
    target = _config_path(path, config)
    if not target.is_file():
        raise click.ClickException(
            f"no config at {target}. Write one with `autohelix integrate init` and fill it in."
        )
    try:
        return IntegrateConfig.load(path, target)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _pass(path: str, config: str | None, verbose: bool = False) -> IntegratePass:
    return IntegratePass(_load(path, config), console=console, verbose=verbose)


def _run(stage_name: str, fn) -> int:
    try:
        outcome = fn()
        if getattr(outcome, "ok", True) is False:
            console.print(
                f"\n[yellow]{stage_name} did not finish:[/yellow] "
                f"{getattr(outcome, 'detail', 'no detail')}"
            )
            return 1
    except StageError as exc:
        console.print(f"\n[red]{stage_name} stopped:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted during {stage_name}. Progress is saved.[/yellow]")
        return 130
    return 0


#: AutoHelix's own two options, meaning the same here: the project directory and the config file.
_path_option = click.option(
    "--path", "-p", type=click.Path(exists=True), default=".",
    help="project directory (default: current dir)",
)
_config_option = click.option(
    "--config", "-c", default=None,
    help=f"the config file (default: <path>/{DEFAULT_CONFIG})",
)
_verbose_option = click.option("--verbose", "-v", is_flag=True, help="stream agent output")


@click.group(name="integrate")
def integrate() -> None:
    """Drive a repo that does not satisfy its constraints to one that does.

    The inverse of `autohelix optimize`. There the baseline passes its gate and the loop makes
    it faster. Here the baseline is *expected* to fail, and the loop's only job is to reach a
    passing state — so there is no metric you supply and no iteration budget you have to pick.
    The metric is how many constraints hold, and the run ends when all of them do.

    \b
      init           write or validate the config, check the repo, make the workspace
      run-baseline   run the loop until every constraint holds

    \b
    And for working on the config itself, without running a loop:
      template          print the config template
      check             say what the config is still missing
      gate              run every constraint once against the repo as it stands
      check-constraint  try one agent constraint's prompt and criteria
      status            show the last verdict

    The config is an ordinary `autohelix.yaml` — goal, constraints, scope, agent, budget,
    reviewer — with one addition: a constraint may be `kind: agent`, a prompt and a criteria
    judged by a model reading the repo, for conditions a script cannot express without becoming
    gameable.

    See `integrate/README.md` for why the seams are where they are.
    """


@integrate.command()
def template() -> None:
    """Print the config template. Redirect it to integrate.yaml and fill it in."""
    click.echo(presets.config_template())


@integrate.command()
@_path_option
@_config_option
def check(path: str, config: str | None) -> None:
    """Say what the config is still missing, without creating anything."""
    cfg = _load(path, config)
    problems = cfg.validate()
    if problems:
        for problem in problems:
            console.print(f"  [red]FAIL[/red] {problem}")
        console.print(f"\n[red]{len(problems)} problem(s).[/red]")
        raise SystemExit(1)
    console.print("  [green]OK[/green] the config is complete")
    for warning in cfg.warnings():
        console.print(f"  [yellow]note[/yellow] {warning}")

    console.print(f"\n  project     {cfg.project_path}")
    console.print(
        "  budget      "
        f"{f'{cfg.max_iterations} iteration(s)' if cfg.max_iterations else 'unbounded'}"
    )
    console.print(f"  editable    {', '.join(cfg.editable)}")
    console.print(f"\n  {len(cfg.constraints)} constraint(s), evaluated in this order:")
    for i, constraint in enumerate(cfg.constraints, start=1):
        console.print(
            f"    {i}. [cyan]{constraint.name}[/cyan] "
            f"({constraint.kind}, timeout {constraint.timeout}s)"
        )
    console.print(f"\n  gate budget {cfg.gate_timeout()}s per iteration")


@integrate.command()
@_path_option
@_config_option
@_verbose_option
def init(path: str, config: str | None, verbose: bool) -> None:
    """Write the config template, or validate it and make the workspace.

    With no config present this writes the template for you to fill in, and stops.
    """
    target = _config_path(path, config)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(presets.config_template())
        console.print(f"[green]Wrote[/green] {target}")
        console.print(
            "\nFill it in, then run [cyan]autohelix integrate init[/cyan] again.\n"
            "`autohelix integrate check` lists what is still missing at any point."
        )
        return
    raise SystemExit(_run("init", _pass(path, config, verbose).init))


@integrate.command(name="run-baseline")
@_path_option
@_config_option
@_verbose_option
@click.option(
    "--max-iterations", "-n", type=int, default=None,
    help="stop after this many iterations even if the constraints do not hold. Overrides the "
         "config's backstop; use it to take one careful step at a time.",
)
def run_baseline(
    path: str, config: str | None, verbose: bool, max_iterations: int | None
) -> None:
    """Run the agent loop until every constraint holds.

    Unbounded by default: it ends when the constraints are satisfied, not after a set number of
    tries. `budget.max_cost_usd` and `budget.time` are the valves for a run that cannot
    converge; with neither set, nothing will stop it.

    Resumable — re-running picks up from the last recorded iteration.
    """
    pipeline = _pass(path, config, verbose)
    raise SystemExit(
        _run("run-baseline", lambda: pipeline.run_baseline(max_iterations=max_iterations))
    )


@integrate.command()
@_path_option
@_config_option
@click.option("--stop-early", is_flag=True,
              help="stop at the first failing constraint instead of running them all")
def gate(path: str, config: str | None, stop_early: bool) -> None:
    """Run every constraint once against the repo as it stands.

    Exits 0 only when all of them hold, so this is the form for CI or for answering "is it done
    yet". The loop runs the same evaluation in advisory mode, where a failure is the expected
    state rather than an error.
    """
    outcome = _pass(path, config).gate(stop_early=stop_early)
    raise SystemExit(0 if outcome.ok else 1)


@integrate.command(name="check-constraint")
@_path_option
@_config_option
@click.argument("name")
@click.option("--verbose", "-v", is_flag=True, help="stream the judge's output")
def check_constraint(path: str, config: str | None, name: str, verbose: bool) -> None:
    """Run one agent constraint by name, to try out its prompt and criteria.

    An agent constraint's wording is the whole of its behaviour, and the loop runs it every
    iteration — so it is worth seeing what it says about the repo as it stands before committing
    a run to it.
    """
    from integrate.agentcheck import main as agentcheck_main

    cfg = _load(path, config)
    argv = ["--config", str(cfg.source), "--name", name, "--repo", str(cfg.project_path)]
    if verbose:
        argv.append("--verbose")
    raise SystemExit(agentcheck_main(argv))


@integrate.command()
@_path_option
@_config_option
def status(path: str, config: str | None) -> None:
    """Show the last verdict and how the loop is progressing."""
    import json

    cfg = _load(path, config)
    gate_path = cfg.gate_json
    if not gate_path.is_file():
        console.print(f"no verdict at {gate_path} yet — the loop has not run.")
        return

    payload = json.loads(gate_path.read_text())
    passing = payload.get("constraints_passing", 0)
    total = payload.get("constraints_total", 0)
    colour = "green" if payload.get("satisfied") else "yellow"
    console.print(f"\n  [{colour}]{passing}/{total}[/{colour}] constraint(s) hold\n")
    for entry in payload.get("constraints") or []:
        mark = "PASS" if entry.get("passed") else "FAIL"
        c = "green" if entry.get("passed") else "red"
        console.print(f"  [{c}]{mark}[/{c}] {entry.get('name')} ({entry.get('kind')})")
        detail = str(entry.get("detail") or "").strip()
        if detail and not entry.get("passed"):
            console.print(f"       {detail}")

    summary_path = IntegratePass(cfg, console=console).summary_path()
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        ran = [i for i in summary.get("iterations", []) if i.get("iteration", 0) > 0]
        console.print(
            f"\n  {len(ran)} iteration(s) recorded; best {summary.get('best_passing')}/{total}"
        )


if __name__ == "__main__":
    sys.exit(integrate())
