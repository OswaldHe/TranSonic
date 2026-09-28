# AutoHelix Documentation

**Autonomous iterative improvement harness.**

AutoHelix runs AI agents in a loop — each iteration is isolated in a git worktree, validated against constraints, and only merged when everything checks out. What doesn't pass gets discarded; what the agent learns persists.

See the [main README](../README.md) for an overview, quick start, and the config reference.

## Design records

- **[CONTEXT.md](../CONTEXT.md)** — the glossary. Every term this project coined, and for each one
  the synonyms to avoid. Read it before naming a new concept, and when a term in the code reads as
  ambiguous — `gate` alone was doing three jobs until it was split here.
- **[docs/adr/](adr/)** — decisions that were hard to reverse and surprising without their reasons,
  each with the alternative that was rejected and why. Three so far, all from `optimization/`:
  who chooses how a module is cut, why a gate is the only thing that runs a candidate, and what
  projecting a 16-device placement onto one device gives up.

## Guides

- **[Getting Started](getting-started.md)** — install, initialize a project, run your first loop
- **[Configuration](config.md)** — the full `autohelix.yaml` surface
- **[CLI Reference](cli.md)** — `init`, `run`, `clear`, `watch`, `report`
- **[Docker Sandbox](docker.md)** — run the agent with filesystem isolation
- **[Concepts](concepts.md)** — how iteration, isolation, and verified improvement
  work, and how to design a loop
- **[ML Experiments](ml-experiments.md)** — patterns for training/eval loops (post-training, benchmarks)
- **[Model Partitioning](../partition/README.md)** — `autohelix partition`: split a model
  into locally-runnable modules, trace real per-module IO, verify each module, and
  emulate end-to-end inference for accelerator bring-up
- **[NKI Bootstrapping](../bootstrap/README.md)** — `autohelix bootstrap`: turn one module
  of a partition artifact into a standalone repo and loop until it holds a working
  Trainium NKI kernel with a validator that proves it reproduces the reference
- **[Floorplanning](floorplan.md)** — `autohelix floorplan`: decide how to distribute a
  partitioned model across a Trainium instance's device / logical-NeuronCore hierarchy,
  by probing the hardware, building a simulator, searching for a deployment, and ranking
  the results
- **[Module Optimization](../optimization/README.md)** — `autohelix optimize`: make one
  bootstrapped module fast on a single device, by cutting it down to what one NeuronCore
  runs, optimizing that under a per-iteration constraint schedule, and rejoining the ranks
  with `nki.collectives`
