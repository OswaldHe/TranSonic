# AutoHelix Documentation

**Autonomous iterative improvement harness.**

AutoHelix runs AI agents in a loop — each iteration is isolated in a git worktree, validated against constraints, and only merged when everything checks out. What doesn't pass gets discarded; what the agent learns persists.

See the [main README](../README.md) for an overview, quick start, and the config reference.

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
