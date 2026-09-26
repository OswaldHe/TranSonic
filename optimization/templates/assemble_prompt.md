You are putting an optimized per-rank kernel back together into the **whole module**, running on
{{ ranks }} ranks of one Trainium device, with the ranks rejoining through `nki.collectives`.

This is the stage where the pipeline's looseness gets paid for. Everything before it was judged
against goldens an agent chose; from here the target is the bootstrapped module's own recorded
output, and the bar is the bar that admitted the bootstrapped kernel.

## What you are given

`{{ repo }}` holds, all frozen:

| | |
|---|---|
| `submodule/` | the optimized single-rank kernel — `source.py` at its best accepted commit, its `inference.py`, its `submodule.json` declaration and its `SUBMODULE.md` |
| `module/` | the bootstrapped module: `source.py`, `inference.py`, `README.md`, and `tensors/` with the input, every weight, and the golden |
| `reference/` | the artifact's frozen references |

The numbers you have to beat, measured on this host rather than copied from a log:

- the bootstrapped module, one core: **{{ bootstrap_latency }} ms**
- the optimized submodule, one core: **{{ submodule_latency }} ms**

## What you must produce

**`source.py`** — the whole module on {{ ranks }} ranks. Each rank runs the submodule's work for its
own shard and the ranks rejoin with a collective **from `nki.collectives`**, inside the traced graph.
`torch.distributed` may set the process group up (`init_process_group`, `barrier`, `get_rank`,
`destroy_process_group`) but may not move tensor data: a host-side or XLA reduction would work and
would measure a different machine than the one the floorplan is about.

Six things about `nki.collectives` on this toolchain, each of which fails with an internal compiler
error that names something else. They are not optional and they are not discoverable cheaply:

1. **Invoke the kernel as `kernel[2](...)`.** NKI defaults to `lnc=1`; this host runs LNC=2, and a
   LNC=1 NEFF leaves the collective's buffer unallocated on the logical core's second physical core
   (`NCC_ILLC059 Could not find MemoryLocation ... on core 1`).
2. **Pass `name=` to the collective's `src`/`dst` `nl.ndarray`s**, or DRAM allocation fails
   (`NCC_IBIR440`).
3. **A collective may not read or write IO tensors** (`NCC_INLA001`). `nisa.dma_copy` the input into
   a named scratch buffer, collective into a second named scratch, `nisa.dma_copy` that into the
   returned buffer.
4. **Every `src` and `dst` must be `nl.shared_hbm`** — buffer kinds may not be mixed.
5. **Build `ReplicaGroup` outside the kernel and pass it in.** The tracer rejects `range` in a traced
   body; `ReplicaGroup([[0, 1, ..., {{ last_rank }}]])`.
6. **Build input tensors on CPU and `.to(device)` them.** `torch.full(..., device=xla)` emits a
   broadcast HLO that hits `NCC_ISMP902`, and that error is the broadcast, not the collective.

`aws-neuron/nki-library`, at
`src/nkilib_src/nkilib/experimental/collectives/collectives.py`, is the reference pattern for all of
the above.

**`inference.py`** — the validator, then **frozen for the whole optimization loop that follows**:

  - runs under `torchrun --nproc_per_node={{ ranks }}`, every rank exiting 0;
  - loads the input, weights and golden from `tensors/` in this repo — the module's recorded bytes;
  - traces `{{ entry_point }}` from `source.py`, runs it on all {{ ranks }} ranks, and compares
    **rank 0's post-collective output** — the whole module's output, after the ranks have rejoined —
    against the golden. One comparison against the whole module, not {{ ranks }} partial ones;
  - prints `##autohelix[passed=1]` and exits 0 on a match, non-zero otherwise;
  - prints `##autohelix[max_abs_err=<number>]` and fails when it exceeds `MAX_ABS_ERR`;
  - declares the five constants with exactly the values in `module/README.md` under "The numerical
    bar". **Unchanged.** The golden is the same tensor the bootstrap loop matched, so the bar that
    admitted that kernel admits this one. The {{ ranks }}-rank reduction order differs from the
    reference's single all-reduce, so accumulation order shifts a little — absorbing that is what a
    pass fraction and a cosine are for, not a reason to loosen anything;
  - profiles the collective run and reports the latency:

        neuron-explorer capture -n <neff> --io-from=runtime \
          --collectives-worker-count {{ ranks }} --collectives-workers-per-node {{ ranks }} \
          --collectives-worker-start-id 0 --collectives-profile-id all -s profile.ntff
        neuron-explorer view -n <neff> -s profile_rank_<N>.ntff \
          --output-format=summary-json --disable-ui

    That writes one `profile_rank_<N>.ntff` per rank. `total_exec_time` in the summary is in
    **seconds**. Print every rank as `##autohelix[latency_rank_<N>_ms=<number>]` and the **fastest**
    of them as `##autohelix[latency_ms=<number>]`. All {{ ranks }} per-rank lines are required: without
    them "the fastest rank" is a claim nobody can check. The summary also carries `cc_op_time`, which
    isolates the collective — worth printing into your notes, since it is the number stage 5 has to
    shrink.

**`tensors/`** — the module's input, weights and golden, copied byte-for-byte from `module/tensors/`.
Not regenerated and not re-sliced: this stage's whole purpose is to be judged against the recorded
module, so its inputs are the recorded inputs.

**`.autohelix/optimization/module.json`** — the manifest, written last:

```json
{
  "module": "{{ module }}",
  "entry_point": "{{ entry_point }}",
  "ranks": {{ ranks }},
  "tolerance": { "RTOL": ..., "ATOL": ..., "MIN_COSINE": ..., "MIN_PASS_FRACTION": ...,
                 "MAX_ABS_ERR": ... },
  "tensors": { "<name>": {"file": "tensors/....bin", "sha256": "...", "bytes": 123} },
  "frozen":  { "inference.py": "<sha256 of the validator you just froze>" },
  "baselines": { "bootstrap_latency_ms": {{ bootstrap_latency }},
                 "submodule_latency_ms": {{ submodule_latency }} }
}
```

**`ASSEMBLY.md`** — for a human: how the ranks divide the work, which collective rejoins them and
where in the graph it sits, what is replicated, and what the first measurement showed — total, per
rank, and `cc_op_time`.

## The bar you have to clear before the loop starts

Your baseline has to be green *and* already satisfy both bounds:

- faster than **{{ bootstrap_latency }} ms** — {{ ranks }} ranks that cannot beat one core have spent
  their parallelism on overhead;
- no slower than **{{ overhead_ceiling }} ms**, which is 1.1x the submodule. Each rank does one
  rank's work, so anything past this is the collective plus imbalance, and 10% is what they get.

If you cannot clear both, say so plainly in `ASSEMBLY.md` with the measurements — an honest failure
here is worth far more than a validator bent until it passes. The loop that follows optimizes from
this baseline, so a baseline that does not hold gives it nothing to stand on.
