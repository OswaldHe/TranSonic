# `floorplan.yaml`

The one file you may edit. It says where every module of the model runs, how it is split, in
which memory tier its weights live, and in what order. Everything else in this repository is
frozen and reverted if touched.

Target: **{{TARGET}}** — {{DEVICES}} devices, {{PER_DEVICE}} logical NeuronCores each,
**{{UNITS}} units**. {{MODULES}} modules to place. A logical NeuronCore has a **{{BANK}}**
HBM bank.

Read `PLATFORM.md` for the hardware and `MODEL.md` for what has to go on it. Read `sim/` for
what your choices will cost — it is the definition, and the cost model for a module is the
fastest way to learn what actually drives that module's time.

## Shape of the file

```yaml
version: 1
target: {{TARGET}}

runtime:
  prefill_chunk_tokens: 2048   # prefill is processed in chunks this size
  decode_micro_batch: 1
  pipeline_chunks: true        # may a stage start chunk N+1 before the next stage finishes N

placements:
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

## `placements`

One entry per module, or several if you are splitting a module's *work* across groups (see
`fraction`). Every module in `MODEL.md` needs at least one.

| field | meaning |
|---|---|
| `module` | a module id from the partition graph. `MODEL.md` lists them. |
| `units` | ordered list of `d<device>.l<logical_nc>` addresses, **one per shard, in shard order**. Shard *i* lands on `units[i]`. |
| `splits` | how the module is partitioned. Omit for an unsplit module on a single unit. |
| `fraction` | share of the module handled by this placement. Defaults to 1.0. Fractions for a module must sum to **exactly 1**. |
| `weights` | which tier holds the parameters, and whether they are tiered. |
| `stage` | schedule position, an integer ≥ 0. Modules in the same stage may run concurrently if their dependencies allow. |
| `overlap_collectives` | `true` lets this placement's collectives run on the CC cores while compute proceeds. Dependents still wait. |

`len(units)` must equal the product of the split factors. A one-way split on four units is an
error, not a replication — say what you mean with `fraction`.

## `splits`

| field | meaning |
|---|---|
| `dim` | one of `head`, `hidden`, `expert`, `seq`, `batch`, `layer`, `vocab`, `ngram`. Must be legal for the module — `MODEL.md` says which are. |
| `factor` | how many ways. |
| `collective` | what rejoining costs: `none`, `allreduce`, `allgather`, `reduce_scatter`, `all_to_all`, `p2p`. |

Several splits compose: `head` by 4 and `seq` by 2 is an 8-shard placement needing 8 units.

Two collectives are genuinely free and it is worth knowing which:

- `collective: none` on a `batch` split — data parallelism needs no exchange.
- **any** split whose units are all within one logical NeuronCore. The two physical cores at
  LNC=2 share one address space, so there is nothing to transfer.

Everything else is priced by group *shape*. Four units inside one device use the intra-device
link; four units spread across the torus pay the worst hop distance in the group on every
step. `PLATFORM.md` has the hop table.

## `weights`

| field | meaning |
|---|---|
| `tier` | `hbm_bank`, `device_hbm`, `peer_hbm`, `host_dram` or `nvme`. |
| `resident_fraction` | below 1.0 makes this a tiered placement. |
| `cache_tier` | required when `resident_fraction < 1.0`: where the resident part is held. |
| `hit_rate` | required when tiered: the fraction of accesses the cache serves. |

A tiered module charges its backing tier for **all** of its bytes and the cache tier for the
resident fraction, because a cache is a copy — putting 10% of Engram in HBM does not remove
10% of it from host DRAM.

`hit_rate` is an assumption you are making, not a measurement, and the reviewer will ask you
to justify it. A 0.99 hit rate on a 384-million-row n-gram table at a 1% resident fraction is
a strong claim about locality; have a reason.

## What the gate checks

Seven checks. You do not see the checker, but you see its findings.

| | |
|---|---|
| a | the floorplan parses and is legal for this hardware and this model |
| b | every module placed, fractions summing to 1, nothing off-path placed |
| c | every placed module's producers are placed too |
| d | `sim/` and `systems/` byte-identical to the build — only this file may change |
| e | no memory tier over capacity, at any workload |
| f | all four workloads simulate and publish a metric |
| g | a second run produces identical metrics |

Capacity is the one that catches people, and it is checked **per bank**: {{BANK}} per logical
NeuronCore, not 96 GiB. It is also checked at every workload, so a plan that fits at 128
tokens can fail at 8192 when the KV cache is 64x larger.

## The metrics

```
prefill_128_ms     prefill_8192_ms     decode_128_ms     decode_8192_ms
```

Lower is better on all four. An iteration is **rejected** if any metric is more than 10%
worse than the best that metric has reached — so a change cannot buy decode with prefill.

They are simulated. The absolute numbers are uncalibrated; comparisons between floorplans on
the same simulator are what they are for.

## Where to look when it is slow

`reports/trace.json`, written every iteration:

| key | what it tells you |
|---|---|
| `critical_path` | the chain of ops that set the latency. Start here. |
| `utilization` | per-engine busy fraction. A low tensor number with a high `cc` number means you are communication-bound. |
| `link_bytes` | volume per link class. `inter_device` far above `intra_device` means a parallel group is spread across the torus. |
| `busiest_unit` | the load-balance tell. One unit far above the rest is a placement problem, not a cost-model problem. |
| `top_modules_ms` | where the work is. |
| `memory` | high-water mark and fill per tier. |

## Working on this

Change one thing at a time. This is a {{MODULES}}-entry file; a rewrite touching forty
placements will move the metrics without telling you which change did it, and if the result
regresses past 10% the whole iteration is discarded and you learn nothing.

Write down what you tried and what it did to each of the four metrics — including what made
things worse. A later iteration reading "deeper pipeline cost 3x on decode_128" is saved an
hour.
