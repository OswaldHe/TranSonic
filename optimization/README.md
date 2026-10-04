# `autohelix optimize`

Makes one module fast on a single Trainium device. It takes a correct but slow NKI kernel from
`bootstrap` and a placement from `floorplan`. It cuts the module down to one rank, optimizes that
rank, rejoins the ranks with a collective, then optimizes the whole module.

Terms in **bold** on first use are defined in [`CONTEXT.md`](../CONTEXT.md).

## Quick start

```bash
autohelix optimize template > optimization.yaml   # fill in each <FILL IN>
autohelix optimize check                          # validate the config, print the projection
autohelix optimize all                            # run every stage
```

`check` uses no device. Run it after each config edit.

`all` runs for hours or days. Each **iteration** is one agent run plus one on-device measurement.
Run it detached:

```bash
setsid nohup autohelix optimize all -c optimization.yaml > run.log 2>&1 &
```

## Stages

| # | command | action | acceptance |
|---|---|---|---|
| 1 | `init` | Narrow the planned placement to one device. Create the workspace. | none |
| 2 | `submodule` | An agent cuts the module to one **rank**. | **submodule gate**, 7 checks |
| 3 | `run` | N iterations on the single-rank kernel, under the **constraint schedule**. | submodule gate per iteration |
| 4 | `assemble` | An agent rejoins the ranks with `nki.collectives`. | **module gate**, 9 checks |
| 5 | `run-full` | N iterations on the whole module. | module gate per iteration |
| 6 | `feedback` | An agent reads both loops' notes and writes `FEEDBACK.md`. | structural check only |

`all` runs stages 1 to 6 in order. It stops at the first stage that does not succeed.

### Resume and rebuild

`all` resumes. It skips a stage that recorded a passing gate if that stage's repository is still
present.

`submodule` and `assemble` move the repository they find to the **attic** and have an agent build a
new one. On a stage that already passed, this discards a tuned kernel and its history. Both commands
refuse unless you add `--rebuild`.

| goal | command |
|---|---|
| continue a stopped run | `optimize all` |
| run the loop again on the current kernel | `optimize run` or `optimize run-full` |
| build a different cut or a different assembly | `optimize submodule --rebuild`, `optimize assemble --rebuild` |

## Commands

All commands accept `-c, --config PATH` (default `./optimization.yaml`). All commands that run an
agent accept `-v, --verbose`, which streams agent output to the terminal as well as to its log.

| command | action | options |
|---|---|---|
| `template` | Print the config template. | |
| `check` | Validate the config and print the projection. Uses no device. | |
| `init` | Create the workspace and record the projection. | |
| `submodule` | Cut the module to one rank. | `--rebuild` |
| `compile-constraints` | Compile each **slot**'s prose into a **checker**. | `--stage submodule\|full` |
| `run` | Run the loop on the single-rank kernel. | |
| `assemble` | Rejoin the ranks with a collective. | `--rebuild` |
| `run-full` | Run the loop on the whole module. | |
| `rerun-full` | Run another **round** of stage 5 from the last round's best commit. | `--from-commit SHA`, `--note TEXT` |
| `feedback` | Write `FEEDBACK.md` from both loops' notes. | |
| `report` | Write `REPORT.md`. | |
| `gate` | Run a gate once against the repository as it is. | `--stage submodule\|full` |
| `all` | Run stages 1 to 6 in order. | |

`gate` reports whether a repository would pass now, without starting a loop. `rerun-full` keeps the
previous round's notes and reviews, archives its other run state, and recompiles the checkers from
the current prose.

## Config

`optimize template` prints every key with comments. The keys to set:

| block | key | effect |
|---|---|---|
| `module` | `id` | which module to optimize |
| | `bootstrap_repo` | the bootstrapped kernel and the recorded tensors |
| | `artifact` | the partition artifact |
| `floorplan` | `scheme` | the ranked placement to project |
| | `target_units` | logical NeuronCores on the target device |
| | `on_oversized` | `project` or `error` when the module does not fit |
| `workspace` | `root` | where the repositories and run state go |
| | `venv` | the Python environment the validator runs in |
| `memory` | `path`, `prompt` | a directory of earlier work, shared by both stages |
| `submodule`, `full` | `goal` | what the loop must achieve |
| | `budget.iterations` | iteration count for the stage |
| | `budget.iteration_time` | wall-clock limit per iteration |
| | `acceptance.max_regression_pct` | the **metric gate**: how much slower than the best an iteration may be |
| | `memory.iterations` | which iterations read the memory |
| | `iteration_constraints` | the constraint schedule |
| | `reviewer` | the **reviewer**'s model, timeout and prompt |
| `agent` | `type` | the agent backend |
| `preparation` | `retries`, `timeout` | attempts and time limit for stages 2 and 4 |
| `constraint_compiler` | `model`, `timeout` | the **constraint compiler** |
| `feedback` | `model`, `timeout` | the feedback agent |

`check` rejects an incomplete block. A `memory:` block with a selector and no `path:` is an error,
not a disabled block.

## Constraint schedule

A **slot** is a run of consecutive iterations under one **constraint**, written as prose. Slots do
not overlap.

```yaml
submodule:
  iteration_constraints:
    - from: 1
      to: 3
      enforcement: hard
      text: |
        Only the NKI API in source.py. Do not use torch.
    - at: 4
      enforcement: soft
      text: |
        Replace some linear projections with torch.nn.Linear.
    - iterations: [5, 6]
      enforcement: off
      text: |
        No restrictions. Explore for latency.
```

Select iterations with `at: N`, `from:`/`to:`, or `iterations: [N, M]`. Use one form per slot.

**Enforcement** sets what a violation costs on one iteration:

| value | effect |
|---|---|
| `hard` | The iteration is rejected. Its work is discarded. |
| `soft` | The checker runs **advisory**. The iteration is kept only if it is correct and strictly faster than the best so far. It loses `acceptance.max_regression_pct`. |
| `off` | The text reaches the agent's prompt. Nothing checks it. |

`soften_last` (default `true`) sets a `hard` slot's last iteration to `soft`.

The constraint compiler writes one checker per slot before iteration 1. The optimizing agent reads
the prose and never reads the checker.

Two rules for the prose:

- A `#` line inside a `text: |` block is not a YAML comment. The whole block reaches the agent.
- Editing a slot's prose after its checker was compiled is **drift**. The stage recompiles the
  checker before the loop starts.

A slot with `enforcement: hard` whose checker cannot be compiled stops the stage. Set
`enforcement: soft` to accept the text as advice instead.

## Memory

`memory:` points at a directory of earlier work. Before each iteration that reads it, the directory
is copied into the worktree at `.autohelix/memory/`, read-only.

```yaml
memory:                         # both stages inherit these
  path: ./memory                # absolute, or relative to the config file
  prompt: |
    submodule/ is the last round's single-rank kernel. Read README.md first.

submodule:
  memory:
    iterations: [1, 2, 3]       # all (default), none, [1, 4], from:/to:, or at: 3
full:
  memory:
    iterations: [1]
    preparation: false          # keep the stage-4 agent blind to it
    prompt: |                   # replaces the shared prose for this stage
      module/ is the whole-module assembly, including where the collective sits.
```

`iterations` selects loop iterations. `preparation` (default `true`) selects the one-shot agent that
builds the repository the loop runs on.

| reader | key |
|---|---|
| stage 2, the cut | `submodule.memory.preparation` |
| stage 3, one rank | `submodule.memory.iterations` |
| stage 4, the assembly | `full.memory.preparation` |
| stage 5, four ranks | `full.memory.iterations` |

A stage that names any selector key replaces the shared selector completely.

Constraints on the directory:

- It must not contain symlinks. A snapshot cannot hold a live pointer.
- It must not contain the worktree. Point `path:` beside the workspace root, not above it.
- A missing directory is a warning. The loop reports how many files it seeded.

## Gates

A **gate** runs a repository's **validator** once and answers a fixed list of **checks**. Two exist.
Name which one. Neither is `acceptance.max_regression_pct`.

**Submodule gate** (`submodule_checker.py`), 7 checks, all module-agnostic:

| | check |
|---|---|
| a | The declared entry point exists at module scope. The validator reaches it. |
| b | `source.py` and `inference.py` are self-contained. |
| c | A fresh profile and a real latency. |
| d | The baseline passes at the pinned **numerical bar**. |
| e | The tensors are the recorded bytes. |
| f | The cut is declared and its **reassembly recipe** verifies. |
| g | The run used one core. |

**Module gate** (`module_checker.py`), 9 checks:

| | check |
|---|---|
| a | `inference.py` is what `assemble` froze, and drives `source.py`. |
| b | Both files are self-contained. |
| c | The reduction is `nki.collectives`, resolved through the import table. |
| d | Every rank ran, exited 0, and reported its latency. |
| e | Rank 0's post-collective output matches the module **golden** at the module's bar. |
| f | A fresh profile covering every rank. `latency_ms` is one of the printed numbers. |
| g | Data provenance, byte for byte. |
| h | Faster than the bootstrapped single-core module. |
| i | No slower than 1.1x the submodule. |

Checks (h) and (i) are re-measured on this host at assembly time.

## The numerical bar

The bar is five constants in `inference.py`: `RTOL`, `ATOL`, `MIN_COSINE`, `MIN_PASS_FRACTION` and
`MAX_ABS_ERR`. An agent may not change them. A gate rejects a validator whose constants differ from
the manifest.

Stage 2 derives the bar from the rank's own recorded output. At the start of stage 3 the pipeline
re-pins `MAX_ABS_ERR` to the cut's own measured error plus 10%, then rewrites `inference.py`, the
manifest and the recorded hash together. This only tightens the bar. It is idempotent, so restarting
stage 3 does not tighten it again.

Without the re-pin the bar describes the recorded output and not the achievable error. A cut far
better than its derived bar leaves the loop free to spend accuracy it does not need.

## Validator and metric

In both loop stages `scope.editable` is `[source.py]`. `inference.py` is frozen after its stage's
gate accepts it.

The kernel's argument shapes, dtypes and layout are therefore fixed for the whole run. An iteration
that needs a different I/O contract records that in its notes. The reviewer carries it to the
operator.

The **metric** is `latency_ms`, the fastest rank. It isolates compute from load imbalance. It
understates the module's cost in a pipeline, where the slowest rank gates the next stage. `REPORT.md`
states this.

The metric is read back from the gate's **verdict** (`readback.py`) rather than measured again. The
gate is the only thing that runs a candidate.

## Output

| path | contents |
|---|---|
| `<root>/<module>-rank0/` | the single-rank repository, one per run |
| `<root>/<module>-full/` | the whole-module repository |
| `<root>/.optimization/` | projection, baselines, per-attempt verdicts, agent logs |
| `<root>/.optimization/attempts/` | the **attic**: set-aside preparation attempts |
| `<repo>/.autohelix/logs/run-<ts>.log` | the run log, one line per event |
| `<repo>/.autohelix/logs/iter-N/` | `agent.log`, `prompt.txt`, `review.log`, `review.md` |
| `<repo>/.autohelix/notes/iter-N.md` | the agent's own notes |
| `<repo>/.autohelix/optimization/candidates/iter-N/` | the **candidate archive**: every constraint-passing kernel with its metrics |
| `<root>/REPORT.md` | what each stage achieved |
| `<root>/FEEDBACK.md` | **blockers**, grouped by who would have to fix them |

The deliverable each stage hands to the next is the best accepted commit, not `HEAD`. With
regression slack, `HEAD` can be slower than the best iteration.

## Error messages

| message | cause | action |
|---|---|---|
| `N configuration problem(s); nothing was created` | the config is incomplete or inconsistent | the problems are listed above the message |
| `<module> projects onto <shape>, which is more than one dimension` | the placement splits on two dimensions | pick a module with a one-dimensional split, or narrow the scheme's placement |
| `<scheme> no longer projects <module> the way this workspace was initialized` | the scheme changed after `init` | restore the scheme, or set a new `workspace.root` |
| `the submodule repo did not pass its gate in N attempt(s)` | the cut is wrong or incomplete | read the last verdict in `.optimization/`; a repeated failure is the prompt or the module |
| `the assembly did not pass its gate in N attempt(s)` | the assembly misses a check or a latency bound | if it matches the golden but misses a bound, revisit the cut |
| ``N checker(s) for `enforcement: hard` slot(s) are still unusable`` | the compiler cannot write a checker for that prose | rewrite the slot's text, or set `enforcement: soft` |
| `a compiled checker has changed since it was written` | a checker was edited after the manifest recorded its hash | `optimize compile-constraints --stage <stage>` |
| `the <stage> loop recorded no iteration, so it did not start` | the repository is dirty, a scoped file is missing, or the config drifted | the causes print above the message |
| `<stage> already passed its gate and <repo> is still there` | a one-shot stage was named after it passed | use `optimize run`/`run-full`, or `--rebuild` |
| `could not measure <label>` | the validator exited non-zero or printed no latency | the last 15 lines of its output follow the message |
| `N of the bootstrapped module's recorded tensors changed` | a recorded tensor was written during the run | restore `bootstrap_repo` from its source |
| `memory.path <p> contains N symlink(s)` | the memory tree holds links | replace each link with a copy |
| `<stage> did not succeed (<detail>)` | `all` stopped at that stage | fix the stage, run it alone, then `optimize all` |

## Layout

```
optimization/
  cli.py                   the commands
  driver.py                the stages, the one-shot agents, the manifests
  config.py                optimization.yaml -> the pipeline config, and the derived
                           per-stage AutoHelix configs
  projection.py            a placement -> what one device holds
  materialize.py           the two repositories, and the numerical bar
  recipe.py                the reassembly recipe, checked arithmetically
  strip.py                 removing an earlier agent's comments (**stripping**)
  slotcheck.py             running a compiled checker, enforcing or advisory
  loop.py                  the loop, the candidate archive, the best commit
  constraints.py           the schedule, the checkers, their manifest
  memory.py                seeding the memory directory read-only
  custody.py               holding the fields a gate reads back outside the agent's repo
                           (**custody**)
  candidate.py             running a candidate repository and reading facts out of it
  submodule_checker.py     the submodule gate
  module_checker.py        the module gate
  readback.py              the metric command
  report.py                REPORT.md
  feedback.py              FEEDBACK.md, and the report's structural check
  presets.py               reading the packaged templates
  templates/               the config template and the five agent prompts
```

## See also

- [ADR 0001](../docs/adr/0001-the-agent-chooses-the-cut.md): why the cut is an agent's judgement and
  not a script.
- [ADR 0002](../docs/adr/0002-a-stages-gate-is-the-only-thing-that-runs-a-candidate.md): why the
  metric is read back from the gate's verdict.
- [ADR 0003](../docs/adr/0003-projecting-onto-one-device-diverges-from-the-plan.md): what projecting
  onto one device gives up, and why a result here is not evidence about the planned deployment.
