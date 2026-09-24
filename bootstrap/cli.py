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
from bootstrap import memory as mem
from bootstrap.preset import PRESET_PATH

#: The test target: layer 0's attention is the one pure sliding-window attention in its
#: group (compress_ratio 0 — no compressor, no indexer, no compressed-KV concatenation), so
#: it is the smallest correct kernel the group admits.
DEFAULT_MODULE = "layers.0.attention"

#: Seconds `inference.py` gets when the gate is run by hand with `bootstrap check`. Inside
#: the loop this comes from `preset.yaml`'s own `--timeout`, not from here.
DEFAULT_RUN_TIMEOUT = 1200


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
@click.option("--no-state", "include_state", is_flag=True, default=True, flag_value=False,
              help="Omit recorded cross-module state. For a group that publishes it rather "
                   "than reads it, where the snapshot is only its own output and stale residue")
@click.option("--force", is_flag=True,
              help="Clear a non-empty target directory first, git history included")
def init(
    repo: Path, artifact: Path, module_id: str, group: str | None, sample: str | None,
    step: int | None, call_index: int, include_state: bool, force: bool,
) -> None:
    """Materialize a module of ARTIFACT into REPO as a git repo to bootstrap in.

    Writes no config: `bootstrap/preset.yaml` is a fixed file the loop reads directly, so
    every module repo runs under the same reviewed preset. Only the tensor manifest is
    per-repo.
    """
    repo = repo.resolve()
    existing = list(repo.iterdir()) if repo.is_dir() else []
    if existing and not force:
        raise click.ClickException(f"{repo} is not empty; pass --force to clear it")
    if existing:
        click.echo(f"clearing {len(existing)} existing entr{'y' if len(existing) == 1 else 'ies'} "
                   f"from {repo}")
        mat.clear(repo)

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
            include_state=include_state,
        )
        manifest_path = mat.write_manifest(result)
        head = mat.git_init(repo)
    except mat.MaterializeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("")
    click.echo(f"  sample   : {result.sample_id}#{result.step} (invocation {result.call_index})")
    click.echo(f"  chain    : {' -> '.join(s for s in result.submodules if s)}")
    click.echo(f"  tensors  : {len(result.tensors)} file(s), "
               f"{result.total_bytes / (1 << 20):.1f} MiB")
    state_count = sum(1 for t in result.tensors if t.role == "state")
    if not result.state_included:
        click.echo("  state    : omitted (--no-state), so the kernel is not handed values "
                   "this group publishes")
    elif state_count:
        click.echo(f"  state    : {state_count} cross-module tensor(s) as kernel inputs — see "
                   f"the README, some may be this group's own output")
    click.echo(f"  manifest : {manifest_path.relative_to(repo)}")
    click.echo(f"  config   : {PRESET_PATH} (fixed, read directly)")
    click.echo(f"  baseline : {head}")
    click.echo("")
    click.echo(f"Next: autohelix bootstrap run --path {repo}")


@bootstrap.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="The module repo")
@click.option("--iterations", "-n", type=int, default=None, help="Override the iteration budget")
@click.option("--config", "-c", "config_file", default=None,
              type=click.Path(exists=True, path_type=Path),
              help=f"An alternative preset [default: {PRESET_PATH}]")
@click.option("--verbose", is_flag=True, help="Show the agent prompt and per-phase detail")
def run(path: Path, iterations: int | None, config_file: Path | None, verbose: bool) -> None:
    """Loop on a bootstrap repo until the gate passes or the budget runs out.

    Reads `bootstrap/preset.yaml` directly unless `--config` points elsewhere. Pass an
    alternative to try a variant preset without editing the reviewed one.
    """
    from bootstrap.driver import BootstrapLoop

    repo = path.resolve()
    if not (repo / mat.MANIFEST_REL).is_file():
        raise click.ClickException(
            f"{repo} has no bootstrap manifest; run `autohelix bootstrap init` first"
        )
    loop = BootstrapLoop(repo, verbose=verbose, config_file=config_file)
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
         "--timeout", str(timeout)],
        cwd=repo,
    )
    raise SystemExit(completed.returncode)


@bootstrap.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="A module repo; the memory beside it is the one meant")
@click.option("--clear", "do_clear", is_flag=True,
              help="Delete every entry. Off by default: memory accumulates across modules")
def memory(path: Path, do_clear: bool) -> None:
    """List what earlier modules left behind, or clear it.

    Entries accumulate as modules are bootstrapped, and each iteration reads a seeded copy.
    Clearing is for when the accumulated kernels stop being relevant — a different model, or
    a partition whose archetypes do not resemble the recorded ones.
    """
    store = mem.location(Path(path).resolve())
    found = mem.entries(store)
    if do_clear:
        if not found:
            click.echo(f"{store} holds nothing to clear")
            return
        removed = mem.clear(store)
        click.echo(f"cleared {removed} entr{'y' if removed == 1 else 'ies'} from {store}")
        return

    click.echo(f"{store}\n")
    if not found:
        click.echo("  empty — the first module bootstrapped here will start it")
        return
    for entry in found:
        state = "passed" if entry.passed else "failed"
        click.echo(f"  {entry.name}")
        click.echo(f"    {entry.module_id}  sample {entry.sample_id}  "
                   f"{state} after {entry.iterations} iteration(s)")
    click.echo(f"\n{len(found)} entr{'y' if len(found) == 1 else 'ies'}. "
               f"Clear with `autohelix bootstrap memory --clear`.")


def _commit_bar(repo: Path, changed: list[str]) -> bool:
    """Commit retune's rewrite of the tracked README. True if the tree is left clean."""
    import subprocess

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(("git", "-C", str(repo)) + args,
                              capture_output=True, text=True, check=False)

    if git("rev-parse", "--git-dir").returncode != 0:
        return True  # not a repo, so nothing can be left dirty
    if not git("diff", "--quiet", "--", "README.md").returncode:
        return True  # the README already matched; nothing to record
    message = (f"Retune the numerical bar: {', '.join(changed)}\n\n"
               "Written by `autohelix bootstrap retune`, which re-derives the bar from the\n"
               "recorded tensors. inference.py still declares the old values; the gate names\n"
               "the mismatch and the next iteration fixes it.\n")
    # `--only` with a pathspec commits that path alone. A plain `git commit` would fold in
    # whatever the operator already had staged, silently attributing their work in progress to
    # an automatic bar change.
    return git("commit", "-q", "--only", "-m", message, "--", "README.md").returncode == 0


@bootstrap.command()
@click.option("--path", "-p", type=click.Path(exists=True, path_type=Path), default=".",
              show_default=True, help="The module repo")
@click.option("--dry-run", is_flag=True, help="Report what would change and write nothing")
def retune(path: Path, dry_run: bool) -> None:
    """Re-derive the numerical bar for an existing repo, keeping its history.

    Use this after the bar's derivation changes. It rewrites the manifest and the README's
    bar section and commits the README, so the next `bootstrap run` still starts on a clean
    tree — `inference.py` is the agent's, and the gate will name the mismatch for it to fix on
    the next iteration. Run `bootstrap init --force` instead if you want a clean repo, but
    note that discards the git history the loop has accumulated.
    """
    repo = path.resolve()
    try:
        before, after = mat.derive_bar(repo) if dry_run else mat.retune(repo)
    except mat.MaterializeError as exc:
        raise click.ClickException(str(exc)) from exc

    names = sorted(set(before) | set(after))
    changed = [n for n in names if before.get(n) != after.get(n)]
    if not changed:
        click.echo("the bar is already current; nothing changed")
        return
    click.echo(f"{repo}\n")
    for name in names:
        old, new = before.get(name), after.get(name)
        mark = "  " if old == new else "->"
        click.echo(f"  {mark} {name:18} {('-' if old is None else f'{old:g}'):>10}"
                   f"   {('-' if new is None else f'{new:g}'):>10}")
    click.echo(f"\n{len(changed)} value(s) changed. inference.py still declares the old ones;"
               f"\nthe gate will name them and the next iteration fixes them.")

    # README.md is tracked, so leaving the rewrite uncommitted makes the next `bootstrap run`
    # refuse to start at ensure_clean_working_tree() — the retune workflow would stop on a
    # manual step nothing told the operator about. The bar is the harness's to set, not the
    # agent's, so recording it is part of retuning rather than something to hand back.
    if not _commit_bar(repo, changed):
        click.echo("\ncould not commit the bar; commit README.md before `bootstrap run`.")


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
