# Floorplanning: deciding what runs where

`autohelix floorplan` takes a model that has already been cut into modules and decides how to
spread it across a Trainium instance — which of the 64 logical NeuronCores on a trn2.48xlarge
holds each module, how each module is divided, where its weights live, and in what order it all
runs.

It exists because that question cannot be answered by measurement. The target is a 16-device
instance; development happens on a one-device box. So the loop optimizes against a simulator,
and the interesting part is what it takes to make a simulated metric worth optimizing against.

For the design and the reasoning, see [`floorplan/README.md`](../floorplan/README.md). This page
is how to run it.

## Before you start

You need:

- a **published partition artifact** — the output of `autohelix partition`, holding
  `plan/partition_graph.yaml`, `modules/<group>/` and `vendor/`
- a **Trainium host**. The target can be any size, but the probe has to run on real silicon, and
  the NKI toolchain (`torch_neuronx`, `nki`, `neuronx-cc`) has to import

```bash
source .venv/bin/activate
neuron-ls          # should report at least one device
```

## The short version

```bash
autohelix floorplan all my-floorplan --artifact /path/to/partition-artifact
```

That runs all five stages in order and takes several hours: two agent iterations to write the
cost models, five to search for a floorplan, one to rank the results. Each stage is also a
command of its own, which is what you want if anything needs a second look.

## Stage by stage

### 1. `probe` — measure the primitives

```bash
autohelix floorplan probe
```

Times a matmul, a DMA, a gather, a collective and some storage reads on this host, and writes
the achieved coefficients to `floorplan/systems/probed.yaml`. Run once per machine; it takes
10–20 minutes, mostly compiling.

**The simulator refuses to run until this has happened.** That is deliberate: substituting the
datasheet's peak for a measured coefficient would make arithmetic free and rank every scheme on
communication alone.

Every value lands with provenance. Some cannot be measured on a one-device host — inter-device
NeuronLink, torus hop latency — and stay at their datasheet figures, flagged as extrapolations
in the final report. On this toolchain the intra-device link turns out not to be measurable
either (NKI 0.6.0 has no collective primitive and the XLA path fails to compile), so it is
*derived* from the probed DMA efficiency and tagged `derived_from_probe`.

If a probe fails, the command says which and exits non-zero rather than filling in a guess.

### 2. `init` — materialize a project

```bash
autohelix floorplan init my-floorplan --artifact /path/to/partition-artifact
```

Builds a self-contained git repository:

| | |
|---|---|
| `floorplan.yaml` | **the one editable file** — the generated baseline to begin with |
| `sim/` | the simulation framework, copied in. Agent-written cost models land in `sim/modules/` |
| `systems/` | the platform description, so the project cannot have it change underneath it |
| `PLATFORM.md` | the hardware, rendered from the YAML rather than restated |
| `MODEL.md` | what has to be placed, how big it is, what it depends on |
| `README.md` | the schema for `floorplan.yaml`, field by field |
| `schemes/candidates/` | where feasible floorplans are archived |

`--target` picks the system YAML (`trn2-16device` by default; `trn2-1device` is the dev host).
`--force` clears a non-empty directory, git history included.

### 3. `build` — write the cost models

```bash
autohelix floorplan build --path my-floorplan
```

Two agent iterations that write `sim/modules/*.py` — one cost model per module archetype — and
`sim/constraints.py`, the executable form of the platform's prose constraints. Gated by a
nine-check invariant suite.

On success the simulator is **frozen**: `sim/` is hashed into the manifest, and from then on the
exploration loop may only edit `floorplan.yaml`. If the suite is not green, nothing is frozen and
the command tells you so — a simulator that fails its own invariants would mislead the search,
so `floorplan all` stops here rather than continuing.

To freeze by hand after fixing something:

```python
from pathlib import Path
from floorplan import driver
driver.freeze(Path("my-floorplan"))
```

### 4. `run` — search for a floorplan

```bash
autohelix floorplan run --path my-floorplan
```

Five iterations, an hour each. An ordinary AutoHelix loop: the agent edits `floorplan.yaml`, a
hidden gate checks the plan is deployable, and four metrics rank it:

```
prefill_128_ms   prefill_8192_ms   decode_128_ms   decode_8192_ms
```

All at batch 1, lower better, each with a **10% regression gate against its own best-so-far**.
A change that wins on decode and loses 15% on prefill is rejected.

Every gate-passing iteration's floorplan is archived to `schemes/candidates/` whether or not it
beat the metric gate, so the ranking step has something to choose from even if most iterations
regressed.

Watch it from a second terminal, as with any AutoHelix run:

```bash
autohelix watch --path my-floorplan
autohelix hint  --path my-floorplan "try expert parallelism across devices"
```

### 5. `rank` — the report

```bash
autohelix floorplan rank --path my-floorplan
```

Takes the best three archived schemes and hands them to an agent that **cannot see the
simulator, its traces, or any predicted latency** — the sandbox does not contain them, and the
schemes are anonymized and shuffled. It ranks them on architectural reasoning alone.

Then a script appends the simulator's numbers and an agreement table. Outputs:

| | |
|---|---|
| `REPORT.md` | the ranking, the winning deployment, why it wins, and the decomposition recipes |
| `schemes/rank{1,2,3}.yaml` | the three schemes in the agent's ranked order |

Nothing is discarded. Where the two rankings disagree, the appendix says so and leaves it to a
reader — that disagreement is the most informative thing the pipeline produces.

## Inspecting things by hand

```bash
autohelix floorplan check  --path my-floorplan   # run the gate once and print its verdict
autohelix floorplan report --path my-floorplan   # per-iteration metrics and accept/reject
autohelix floorplan show   --path my-floorplan   # what the current floorplan says
autohelix floorplan show   --path my-floorplan --plan schemes/rank1.yaml
```

And the simulator directly, which is the fastest way to understand a plan:

```bash
cd my-floorplan
python -m floorplan.sim.runner \
    --plan floorplan.yaml \
    --artifact /path/to/partition-artifact \
    --trace reports/trace.json
```

That prints the four latencies, per-engine utilization and the memory table, and writes the full
timeline — critical path, per-link volume, per-tier high-water mark — as JSON.

## Editing the floorplan yourself

The schema is in the project's own `README.md`. A placement looks like this:

```yaml
- module: layers.0.attention
  units: [d0.l0, d0.l1, d0.l2, d0.l3]
  splits:
    - dim: head
      factor: 4
      collective: allreduce
  weights:
    tier: hbm_bank
  stage: 0
```

Three things catch people out:

- **A logical NeuronCore has a 24 GiB HBM bank**, not 96 GiB. The 96 GiB is the device's four
  banks. Capacity is checked per bank and at every workload, so a plan that fits at 128 tokens
  can fail at 8192 when the KV cache is 64× bigger.
- **`units` must have exactly one entry per shard**, in shard order. A one-way split on four
  units is an error, not a replication — use `fraction` to say what you mean.
- **Fractions for a module must sum to exactly 1.** Anything else leaves part of the model
  undeployed, and the gate says so.

## Changing the platform description

`floorplan/systems/trn2-16device.yaml` is the target. Numbers carry `source:` and `confidence:`;
correcting one is an ordinary edit.

For anything that is not a number, add to `constraints_text` — a numbered list in prose, which is
where operational knowledge with no field in the schema belongs:

```yaml
constraints_text: |
  1. GPSIMD and the Tensor Engine cannot access SBUF simultaneously on Trainium 2.
  ...
  10. Your new constraint here.
```

The `build` agent turns each item into a check or a cost term and cites its number; invariant (g)
reports any item nothing cites, so a new item will not be quietly ignored. Changing
`constraints_text` means re-running `build`, since the cost models are what implement it.

## A different target

The system YAML is parameterized, so a one-device target is a first-class configuration:

```bash
autohelix floorplan init dev-floorplan --artifact <dir> --target trn2-1device
```

Useful because it is the only configuration where any part of the prediction can be checked
against hardware. A floorplan built for one target will not validate against the other — the
gate refuses a plan whose `target` differs from the one the cost models were built for.

## Resetting a run

Same shape as the rest of AutoHelix, and `clear` alone is not enough:

```bash
autohelix clear --path my-floorplan            # archive the run state
cd my-floorplan && git reset --hard <initial>   # roll back the agent's commits
rm -rf schemes/candidates reports              # and the archives
```

The tree has to be commit-clean or `run` refuses to start. To go all the way back, re-run `init
--force`, which rebuilds the project from the artifact — but note that discards the cost models
`build` produced, so `build` has to run again too.
