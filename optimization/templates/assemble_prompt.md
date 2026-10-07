You are putting an optimized per-rank kernel back together into the **whole module**, running on
{{ ranks }} ranks of one Trainium device, with the ranks rejoining through `nki.collectives`.

This is the stage where the pipeline's looseness gets paid for. Everything before it was judged
against goldens an agent chose; from here the target is the bootstrapped module's own recorded
output, at the bar that admitted the bootstrapped kernel.

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

{% if memory %}
{{ memory }}
{% endif %}

{% if archive %}
{{ archive }}
{% endif %}

## What you must produce

**`source.py`** — the whole module on {{ ranks }} ranks. Each rank runs the submodule's work for its
own shard, and the ranks rejoin with a collective **from `nki.collectives`**, inside the traced
graph. `torch.distributed` may set the process group up (`init_process_group`, `barrier`,
`get_rank`, `destroy_process_group`) but may not move tensor data: a host-side or XLA reduction
would work and would measure a different machine than the one the floorplan is about.

Six things about `nki.collectives` on this toolchain. Each one fails with an internal compiler error
that names something else, so none of them is cheap to rediscover.
`aws-neuron/nki-library`, at `src/nkilib_src/nkilib/experimental/collectives/collectives.py`, is the
reference pattern for all six.

1. **Invoke the kernel as `kernel[2](...)`.** NKI defaults to `lnc=1`, this host runs LNC=2, and a
   LNC=1 NEFF leaves the collective's buffer unallocated on the second physical core
   (`NCC_ILLC059`).
2. **Pass `name=` to the collective's `src`/`dst` `nl.ndarray`s**, or DRAM allocation fails
   (`NCC_IBIR440`).
3. **A collective may not read or write IO tensors** (`NCC_INLA001`). `nisa.dma_copy` the input into
   a named scratch buffer, collective into a second named scratch, then `nisa.dma_copy` that into
   the returned buffer.
4. **Every `src` and `dst` must be `nl.shared_hbm`.** Buffer kinds may not be mixed.
5. **Build `ReplicaGroup` outside the kernel and pass it in**, as
   `ReplicaGroup([[0, 1, ..., {{ last_rank }}]])`. The tracer rejects `range` in a traced body.
6. **Build input tensors on CPU and `.to(device)` them.** `torch.full(..., device=xla)` emits a
   broadcast HLO that hits `NCC_ISMP902`, and that error names the broadcast, not the collective.

**`inference.py`** — the validator, then **frozen for the whole loop that follows**.
`module/inference.py` is a working one for this module on a single core; write yours to the same
contract, across {{ ranks }} ranks, and keep these properties:

  - it runs under `torchrun --nproc_per_node={{ ranks }}`, every rank exiting 0;
  - it loads the input, weights and golden from `tensors/` in this repo, the module's recorded bytes;
  - it traces `{{ entry_point }}` from `source.py`, runs it on all {{ ranks }} ranks, and compares
    **rank 0's post-collective output** against the golden. One comparison against the whole
    module, not {{ ranks }} partial ones;
  - it prints `##autohelix[passed=1]` and exits 0 on a match, non-zero otherwise, and prints
    `##autohelix[max_abs_err=<number>]`, failing when that exceeds `MAX_ABS_ERR`;
  - it declares the five constants at exactly the values **this repo's own `README.md`** states
    under "What it has to hit", **unchanged**. Take them from there and nowhere else. Other files
    here state a different bar and carry a note saying so, because they belong to the stage they
    were copied from. The {{ ranks }}-rank reduction order shifts accumulation a little, which is
    what a pass fraction and a cosine absorb, not a reason to loosen anything;
  - it profiles the collective run and reports the latency:

        neuron-explorer capture -n <neff> --io-from=runtime \
          --collectives-worker-count {{ ranks }} --collectives-workers-per-node {{ ranks }} \
          --collectives-worker-start-id 0 --collectives-profile-id all -s profile.ntff
        neuron-explorer view -n <neff> -s profile_rank_<N>.ntff \
          --output-format=summary-json --disable-ui

    That writes one `profile_rank_<N>.ntff` per rank, and `total_exec_time` in the summary is in
    **seconds**. Print every rank as `##autohelix[latency_rank_<N>_ms=<number>]` and the **fastest**
    of them as `##autohelix[latency_ms=<number>]`. All {{ ranks }} per-rank lines are required, or
    "the fastest rank" is a claim nobody can check. The summary also carries `cc_op_time`, which
    isolates the collective; print it into your notes, since stage 5 has to shrink it.

**`tensors/`** — the module's input, weights and golden, copied byte-for-byte from
`module/tensors/`. Not regenerated and not re-sliced: this stage exists to be judged against the
recorded module, so its inputs are the recorded inputs.

**`.autohelix/optimization/module.json`** — the manifest. **Edit the file already there**; the
pipeline owns every field but one and restores the rest after you finish, `tensors` included. Yours
is the hash that lets the gate tell that the validator judging iteration 5 is the one gated here:

```json
{
  "frozen": { "inference.py": "<sha256 of the validator you just froze>" }
}
```

**`ASSEMBLY.md`** — for a human: how the ranks divide the work, which collective rejoins them and
where in the graph it sits, what is replicated, and what the first measurement showed — total, per
rank, and `cc_op_time`.

## The bar you have to clear before the loop starts

Your baseline has to be green *and* already satisfy both bounds:

- faster than **{{ bootstrap_latency }} ms**, since {{ ranks }} ranks that cannot beat one core have
  spent their parallelism on overhead;
- no slower than **{{ overhead_ceiling }} ms**, which is 1.1x the submodule. Each rank does one
  rank's work, so anything past this is the collective plus imbalance, and 10% is what they get.

If you cannot clear both, say so plainly in `ASSEMBLY.md` with the measurements. An honest failure
here is worth far more than a validator bent until it passes: the loop optimizes from this baseline,
so a baseline that does not hold gives it nothing to stand on.
