# `autohelix integrate`

Drive a repo that does **not** satisfy its constraints to one that does.

The inverse of `optimization/`. There, the baseline passes its gate and the loop makes it
faster — a failing baseline is a broken run. Here the failing baseline *is* the starting
condition, and the loop's only job is to reach a passing one.

```bash
autohelix integrate init           # write or validate the config, check the repo
autohelix integrate run-baseline   # run the loop until every constraint holds
```

Takes AutoHelix's own `--path/-p` (project directory, default `.`) and `--config/-c` (default
`<path>/integrate.yaml`).

## The config is an AutoHelix config

Not a separate schema. The file is an ordinary `autohelix.yaml` — `goal`, `constraints`,
`scope`, `agent`, `budget`, `reviewer` — parsed by `autohelix.config.load_config`, so every key
means here what it means to `autohelix run`. Two things differ:

**`constraints:` is required**, and a failing one is the expected starting state.

**A constraint may be `kind: agent`.** `autohelix.config` reads only `command` and `timeout` off
a constraint entry and ignores the rest, so the extra keys ride along in the same list without a
schema fork — the same way `optimization/` carries `iteration_constraints`, `memory` and
`feedback_archive` in a config the shared dataclass does not model.

**`metrics:` and `acceptance:` are not yours here.** The pass supplies them, and setting either
is an error rather than an override: they are what make an unbounded loop converge. Everything
else you write is carried through to the derived loop config untouched.

Validation runs `Config.validate` first, so an empty or still-default `goal`, a typo'd
top-level key and everything else AutoHelix already knows about are reported by the code that
owns them.

First stage only: `baseline`, an unbounded agent loop that exits when the constraints are
satisfied.

## Why it is a separate pass and not a mode of `optimize`

Two things differ, and each one changes the loop's shape rather than its settings.

**There is no metric to improve.** `optimize` ranks iterations by `latency_ms` and keeps the
fastest. Nothing here is a speed. So the metric is `constraints_passing` — how many of the
declared constraints hold — and the run ends when it equals the total. That is also what makes
an unbounded loop converge: acceptance is a metric gate at **zero** tolerance, so an iteration
satisfying fewer constraints than the best so far is rejected and its work discarded. Without
it a loop with no end can trade one constraint for another forever.

**The gate is advisory.** In `optimize`, a failing gate means a broken candidate and rejecting
it is right. Here a failing constraint is the normal state for most of the run — it is what the
loop exists to fix — so a rejecting gate would discard every iteration's progress and the run
could only succeed by accident on a single iteration. `integrate.gate` therefore exits 0 under
`--advisory`, which is how the loop runs it; run by hand or from CI it exits 1 when anything
fails.

A consequence worth knowing: `Harness._capture_baseline` runs the metric commands but not the
constraints, and this pass's metric command only *reads back* a verdict. So the loop evaluates
the constraints against the repo once before iterating, which both gives the baseline a metric
and makes the starting count — "you begin at 1 of 3" — a recorded fact.

## Constraints

The point of the pass. Evaluated in the order the config lists them; the count that hold is the
metric.

### `kind: script`

A command. Exit 0 passes; anything else — a non-zero exit, a crash, a timeout — fails. Prefer
these: cheap, repeatable, and not a matter of opinion.

A bare string, exactly as in `autohelix.yaml`:

```yaml
constraints:
  - pytest tests/
```

Or written out, when you want a name or a longer timeout. Without a `name` the command is the
name, which is what `autohelix run` already shows:

```yaml
  - name: imports-clean
    command: python -c "import my_package.model"
    timeout: 300
    rationale: nothing else can be checked if the module will not import
```

Run through `autohelix.checks.run_constraint`, so a timed-out command has its whole process
group killed rather than being left running detached, and it sees the same worktree environment
every other autohelix check does.

### `kind: agent`

A prompt and a criteria, judged by a model reading the repo. For conditions a script cannot
express, or can only express as something gameable — "does not silently fall back to a reference
implementation", "the registration is wired as the guide describes", "the output is not
gibberish".

```yaml
  - name: no-silent-fallback
    kind: agent
    rationale: |
      A port that quietly falls back to the reference path passes every smoke test while
      measuring nothing. No script distinguishes that from a real implementation.
    prompt: |
      Read model.py. Decide whether the forward path genuinely runs the kernels under kernels/,
      or whether it falls back to a torch implementation on any path — including inside a
      try/except, behind a config flag, or via a default argument.
    criteria: |
      PASS if every path through `forward` reaches a kernel from kernels/.
      FAIL if any path computes a result without them, or if you cannot tell.
    model: claude-sonnet-5
    effort: medium
```

The judge runs through `autohelix.agents`, so `agent.type` means the same thing here as
everywhere else — a run configured for codex judges with codex, and the `mock` backend makes the
whole path testable offline. The verdict travels as a JSON file the judge writes, because the
backends stream to a log rather than returning output, so a file is the only thing every backend
can return alike.

**It fails closed.** No verdict file, unparseable JSON, a verdict that is neither PASS nor FAIL,
a backend error, a timeout — all count as failing. A judge that could not answer has not cleared
the constraint. A stale verdict from a previous iteration is deleted before the judge runs, so
it can never read back as this iteration's answer.

**A PASS must cite evidence.** The rubric requires at least one `path:line` or quoted snippet,
and a PASS with an empty `evidence` list is downgraded to a failure. An agent cornered by a
constraint it cannot satisfy writes assertions; this is what makes an unsupported one visible.

**It ignores the repo's instructions.** The tree contains files the agent being judged just
wrote, and a `CLAUDE.md` or comment asserting the constraint is met is exactly what gets
written. The rubric says so explicitly.

Agent constraints cost a model call per iteration and their verdict is a judgement, not a
measurement. `check` warns if every constraint is one.

## What this reuses

Almost all of an iteration is `Harness`'s and is not reimplemented: the preflight, the gitignore
and clean-tree checks, the worktree, the agent run, the constraint and metric commands, the
reviewer, the acceptance decision, the commit, the history and the dashboard. `BaselineLoop`
drives `Harness.run` one iteration at a time and checks the exit condition between them, so the
budget valves and the resume logic keep working as they do everywhere else.

| what | where it comes from |
|---|---|
| the iteration | `autohelix.harness.Harness` |
| the config schema | `autohelix.config.load_config` / `Config.validate` |
| script constraints | `autohelix.checks.run_constraint` |
| the judge's backend | `autohelix.agents.create_agent` / `AgentConfig.from_dict` |
| clean-tree and editable checks | `autohelix.sandbox.Sandbox` |
| history, dashboard, valves | `autohelix.history`, `Harness` |
| config → derived loop config | the `optimization/config.py` pattern |

## Files

| file | what it is |
|---|---|
| `cli.py` | `init`, `run-baseline`, plus `template`, `check`, `gate`, `check-constraint`, `status` |
| `config.py` | the operator's config, the `Constraint` model, and the derived loop config |
| `gate.py` | runs every constraint once, writes the verdict, reports the count |
| `agentcheck.py` | runs one agent constraint and reads back its verdict |
| `readback.py` | reports the count the gate already measured, as the loop's metric command |
| `loop.py` | `BaselineLoop`: the exit condition, the failing baseline, the prompt |
| `driver.py` | the `init` and `run-baseline` stages |
| `templates/` | the config template and the per-iteration prompt |

## Gotchas

**Editable files must be committed**, including ones that do not exist yet. The Harness requires
every path in `scope.editable` to be tracked; create it empty and commit it. `init` checks with
`Sandbox`'s own rule, so passing `init` is a real statement that the loop can start — and
untracked files like `integrate.yaml` or a `__pycache__/` left by a constraint are fine, because
that rule only cares about uncommitted changes to *tracked* files.

**The gate mirrors its verdict into the project.** It runs with the iteration's worktree as its
working directory, so `--json` lands in a tree that is discarded; the copy under
`.autohelix/integrate/gate.json` is what the next prompt and the `status` command read. Safe —
`.autohelix/` is gitignored, and the gate is pipeline-owned, not agent-owned.

**Nothing stops an unbounded run that cannot converge.** That is the intended mode, but set
`budget.max_cost_usd` or `budget.max_time` if the run is unattended. `check` warns when none of
the three is set.

## Tests

```bash
pytest -m integrate      # no device, no network, no model calls
```

Agent constraints are exercised against the `mock` backend, including every fail-closed path.
