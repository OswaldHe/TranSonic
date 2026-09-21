# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the partition loop, exposed as ``autohelix partition``."""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import click
import yaml

def _resource_dir(name: str) -> Path:
    """Locate a bundled resource directory.

    Inside an installed wheel these sit beside the package; in the source tree they
    are one level up, under ``partition/``. Checking both means a ``pip install``
    keeps its bundled specs, defaults and input sets.
    """
    here = Path(__file__).resolve().parent
    installed = here / name
    return installed if installed.is_dir() else here.parent / name


CONFIG_DIR = _resource_dir("config")
DEFAULTS_FILE = CONFIG_DIR / "defaults.yaml"
MODELS_DIR = CONFIG_DIR / "models"


def _load_defaults(path: Path | None = None) -> dict[str, Any]:
    source = path or DEFAULTS_FILE
    if not source.is_file():
        return {}
    payload = yaml.safe_load(source.read_text()) or {}
    return payload.get("loop", payload) if isinstance(payload, dict) else {}


def build_options(config: Path | None = None, spec: Any = None, **overrides: Any):
    """Build LoopOptions, layering defaults, the spec, then the command line.

    A spec's ``overrides`` are loop settings a particular model needs — no bf16
    dequant mirror of a 500 GB fp8 checkpoint, for instance — so they sit above the
    shared defaults and below anything asked for explicitly.
    """
    from model_partition.loop.stages import LoopOptions

    prompt_file = overrides.pop("partition_prompt_file", None)
    if prompt_file and not overrides.get("partition_prompt"):
        overrides["partition_prompt"] = Path(prompt_file).read_text()

    known = {f.name for f in fields(LoopOptions)}
    values = {k: v for k, v in _load_defaults(config).items() if k in known}
    if spec is not None:
        unknown = set(getattr(spec, "overrides", {})) - known
        if unknown:
            raise click.ClickException(
                f"Spec {spec.name!r} overrides unknown loop option(s): "
                f"{', '.join(sorted(unknown))}"
            )
        values.update({k: v for k, v in spec.overrides.items() if k in known})
    values.update({k: v for k, v in overrides.items() if v is not None and k in known})
    if "retention_layers" in values and not isinstance(values["retention_layers"], tuple):
        values["retention_layers"] = tuple(values["retention_layers"])
    return LoopOptions(**values)


def resolve_spec(target: str):
    """Resolve a spec argument: a bundled name, a YAML path, or an HF repo id."""
    from model_partition.spec import load_spec, parse_spec

    candidate = Path(target)
    if candidate.is_file():
        return load_spec(candidate)
    bundled = MODELS_DIR / f"{target}.yaml"
    if bundled.is_file():
        return load_spec(bundled)
    if candidate.is_dir() or "/" in target or target.startswith("hf:"):
        return parse_spec({"source": target})
    available = ", ".join(sorted(p.stem for p in MODELS_DIR.glob("*.yaml")))
    raise click.ClickException(
        f"Unknown spec {target!r}. Give a YAML path, an HF repo id, a local "
        f"directory, or a bundled name ({available})."
    )


# -- shared option decorators ------------------------------------------------


def loop_options(func):
    for decorator in reversed([
        click.option("--artifact-root", type=click.Path(), default=None,
                     help="Where artifacts live (default ~/transonic_artifacts)"),
        click.option("--config", type=click.Path(exists=True, path_type=Path), default=None,
                     help="Defaults file (default partition/config/defaults.yaml)"),
        click.option("--device", default=None, help="Device for tracing (default cuda)"),
        click.option("--headroom", type=float, default=None,
                     help="Fraction of GPU memory one module may use"),
        click.option("--gpu-memory-gib", type=float, default=None,
                     help="Plan for a GPU of this size without one attached"),
        click.option("--seq-len", type=int, default=None,
                     help="Sequence length used for sizing (default: longest sample)"),
        click.option("--max-layers-per-module", type=int, default=None),
        click.option("--one-layer-per-module", is_flag=True, default=None),
        click.option("--experts-per-group", type=int, default=None),
        click.option("--split-attention-ffn", is_flag=True, default=None,
                     help="Give attention and the FFN/MoE of every layer their own module"),
        click.option("--partition-prompt", default=None,
                     help="Instruction on how to partition, handed to the agent"),
        click.option("--partition-prompt-file", type=click.Path(exists=True, path_type=Path),
                     default=None, help="Read the partition instruction from a file"),
        click.option("--cache-weights/--no-cache-weights", default=None,
                     help="Persist per-module weight dumps; off reads them from the checkpoint"),
        click.option("--cache-dequant/--no-cache-dequant", default=None,
                     help="Persist the bf16 dequant mirror of quantized weights"),
        click.option("--slice-long", is_flag=True, default=None,
                     help="Window long-context feature maps to a head/tail; those "
                          "records are then not numerically verified"),
    ]):
        func = decorator(func)
    return func


@click.group(name="partition")
def partition() -> None:
    """Partition, trace and verify a large language model for deployment."""


@partition.command("list")
def list_specs() -> None:
    """List the bundled model specs."""
    rows = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        status = "" if payload.get("enabled", True) else "  (disabled)"
        rows.append(f"  {path.stem:<26} {payload.get('source', '?')}{status}")
    if not rows:
        click.echo("No bundled specs found.")
        return
    click.echo("Bundled specs:")
    click.echo("\n".join(rows))


@partition.command("inspect")
@click.argument("target")
@click.option("--artifact-root", type=click.Path(), default=None)
def inspect_model(target: str, artifact_root: str | None) -> None:
    """Show a model's structure and projected artifact size, doing no work."""
    from model_partition.hardware import detect_host, format_bytes, resolve_budget
    from model_partition.ingest import ingest
    from model_partition.layout import RunLayout
    from model_partition.sizing import ModelInventory

    spec = resolve_spec(target)
    result = ingest(spec)
    included = tuple(n for n, on in (("vision", spec.scope.vision), ("mtp", spec.scope.mtp),
                                     ("engram", spec.scope.engram)) if on)
    inventory = ModelInventory.build(result.index, result.config, include=included)
    layout = RunLayout.create(spec.slug, artifact_root)
    host = detect_host(layout.root)

    click.echo(f"source     : {spec.source}")
    click.echo(f"revision   : {result.revision or '(local)'}")
    click.echo(f"loader     : {result.loader}")
    click.echo(f"checkpoint : {format_bytes(result.index.total_bytes)} "
               f"in {len(result.index.shard_bytes)} shard(s)")
    click.echo(f"dtypes     : " + ", ".join(
        f"{name} {format_bytes(size)}" for name, size in sorted(inventory.index.bytes_by_dtype().items())))
    click.echo(f"layers     : {len(inventory.layers)}  hidden {inventory.hidden_size}  "
               f"vocab {inventory.vocab_size}")
    click.echo(f"signatures : {len(inventory.signature_groups())} distinct layer structure(s)")
    click.echo(f"representative layers: {inventory.representative_layers()}")
    if inventory.subtree_bytes:
        click.echo("excluded   : " + ", ".join(
            f"{name} {format_bytes(size)}" for name, size in sorted(inventory.subtree_bytes.items())))
    if inventory.quant.quantized:
        click.echo(f"quantized  : {inventory.quant.method} block={inventory.quant.block_size} "
                   f"scale={inventory.quant.scale_fmt} experts={inventory.quant.expert_dtype}")
        click.echo(f"dequant to bf16 would add {format_bytes(inventory.dequant_bytes)}")
    try:
        budget = resolve_budget()
        click.echo(f"budget     : {format_bytes(budget.usable_bytes)} on {budget.gpu.name}"
                   if budget.gpu else "budget     : (no GPU)")
    except RuntimeError:
        click.echo("budget     : (no GPU detected)")
    click.echo(f"disk free  : {format_bytes(host.disk_free_bytes)} at {layout.root}")


@partition.command("plan")
@click.argument("target")
@loop_options
def plan_only(target: str, config: Path | None, **kwargs: Any) -> None:
    """Produce the partition plan and stop."""
    from model_partition.hardware import format_bytes
    from model_partition.loop.driver import PartitionLoop
    from model_partition.loop.stages import stage_ingest, stage_plan

    spec = resolve_spec(target)
    options = build_options(config, spec=spec, **kwargs)
    ctx = PartitionLoop(spec=spec, options=options).build_context()

    ingest_result = stage_ingest(ctx)
    click.echo(f"ingest: {ingest_result.detail}")
    plan_result = stage_plan(ctx)
    click.echo(f"plan  : {plan_result.detail}")
    if not ctx.graph:
        raise click.ClickException("planning produced no graph")

    click.echo("")
    for module in ctx.graph.partitioned_modules:
        layers = ",".join(str(i) for i in module.layer_indices) or "-"
        click.echo(f"  {module.id:<28} {module.kind:<16} layers={layers:<12} "
                   f"params={format_bytes(module.param_bytes):>10} "
                   f"resident={format_bytes(module.resident_bytes):>10}")
    click.echo(f"\n{len(ctx.graph.signature_groups())} deduplicated implementation group(s)")
    click.echo(f"plan written to {ctx.layout.graph_path}")


@partition.command("run")
@click.argument("target")
@loop_options
@click.option("--iterations", "--max-iterations", "-n", "iterations", type=int, default=None,
              help="Maximum loop iterations before the run stops (default 5)")
@click.option("--max-new-tokens", type=int, default=None, help="Tokens to generate when emulating")
@click.option("--judge", "judge_kind", type=click.Choice(["claude", "stub"]), default=None,
              help="Judge backend (stub is offline)")
@click.option("--judge-model", default=None, help="Model id for the judge")
@click.option("--min-judge-score", type=int, default=None, help="Passing judge score (1-5)")
@click.option("--agent-model", default=None, help="Model id for the planning/repair agent")
@click.option("--no-agent", is_flag=True, default=False, help="Disable agent plan repair")
@click.option("--refine-plan", is_flag=True, default=False,
              help="Ask the agent to improve the seed plan for kernel development")
@click.option("--no-retain", is_flag=True, default=False, help="Keep all traced layers")
@click.option("--allow-overflow", is_flag=True, default=False,
              help="Warn instead of failing when projected artifacts exceed disk")
@click.option("--force", is_flag=True, default=False,
              help="Re-run even if this run already passed and was pruned")
def run_loop(target: str, config: Path | None, iterations: int | None, no_agent: bool,
             no_retain: bool, allow_overflow: bool, force: bool, refine_plan: bool,
             **kwargs: Any) -> None:
    """Run the partition loop until verification passes."""
    from model_partition.loop.driver import PartitionLoop

    spec = resolve_spec(target)
    if not spec.enabled:
        raise click.ClickException(
            f"Spec {spec.name!r} is marked enabled: false. {spec.notes or ''}".strip()
        )
    options = build_options(
        config,
        spec=spec,
        max_iterations=iterations,
        use_agent_planner=False if no_agent else None,
        retain=False if no_retain else None,
        strict_storage=False if allow_overflow else None,
        force=True if force else None,
        refine_plan=True if refine_plan else None,
        **kwargs,
    )
    result = PartitionLoop(spec=spec, options=options).run()
    raise SystemExit(0 if result.passed else 1)


@partition.command("replay")
@click.argument("run_dir", type=click.Path(exists=True))
@click.argument("module_id")
@click.option("--sample", default=None, help="Sample id (default: first traced)")
@click.option("--device", default="cpu")
def replay(run_dir: str, module_id: str, sample: str | None, device: str) -> None:
    """Replay one module from artifacts alone, without the checkpoint."""
    from model_partition.runtime.standalone import StandaloneError, replay_module

    try:
        comparisons = replay_module(run_dir, module_id, sample, device=device)
    except StandaloneError as exc:
        raise click.ClickException(str(exc)) from exc
    for comparison in comparisons:
        click.echo(comparison.summary())
    raise SystemExit(0 if all(c.passed for c in comparisons) else 1)


@partition.command("report")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Emit the raw state as JSON")
def report(run_dir: str, as_json: bool) -> None:
    """Show a run's stage status and reports."""
    from model_partition.layout import RunLayout
    from model_partition.loop.state import LoopState

    layout = RunLayout.at(run_dir)
    state = LoopState.load(layout.state_file)
    if as_json:
        click.echo(json.dumps(state.to_dict(), indent=2, default=str))
        return
    click.echo(f"run       : {layout.root}")
    click.echo(f"iteration : {state.iteration}   passed: {state.passed}")
    click.echo("stages:")
    click.echo(state.render())
    if layout.summary_file.is_file():
        click.echo(f"\nsummary: {layout.summary_file}")
    if layout.tokens_file.is_file():
        click.echo(f"tokens : {layout.tokens_file}")


@partition.command("tokens")
@click.argument("run_dir", type=click.Path(exists=True))
def tokens(run_dir: str) -> None:
    """Print the sampled continuations for human verification."""
    from model_partition.layout import RunLayout

    path = RunLayout.at(run_dir).tokens_file
    if not path.is_file():
        raise click.ClickException(f"No sampled tokens at {path}; run the loop first.")
    click.echo(path.read_text())


def main(argv: list[str] | None = None) -> int:
    return partition.main(args=argv if argv is not None else sys.argv[1:], standalone_mode=False)


if __name__ == "__main__":
    raise SystemExit(main())
