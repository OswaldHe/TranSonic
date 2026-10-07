You are cutting one module of a Trainium model down to the part a **single NeuronCore** runs, so
that a later loop can optimize that part in isolation and a stage after it can rejoin the ranks with
a collective.

This is preparation, not optimization: nothing here is measured against a target. Leave behind a
correct, self-contained, *measurable* repository, and write down how your cut reassembles into the
whole module. The next stage reads that and cannot proceed without it.

## What you are given

`{{ repo }}` already holds, all frozen:

| | |
|---|---|
| `module/` | the bootstrapped module: its working NKI `source.py`, its `inference.py`, its `README.md`, and `tensors/` — every input, weight and the golden output, as recorded bytes |
| `FLOORPLAN.md` | how the ranked floorplan places this module, and the projection onto this one device |
| `reference/` | `reference_torch.py` (what the module computes), `reference_inference.py` (how it ran), `reference_numerics.py` (what counts as matching), plus `vendor/` and `compat/` |

Read `module/README.md`, then `FLOORPLAN.md`.

Two authorities answer different questions. `module/tensors/` decides **results**: whatever you
build has to agree with those bytes. `reference_torch.py` decides **semantics**: read it to find out
what a per-rank computation even is here. Where the reference already carries a `rank`/`world_size`
notion, the cut you need may be the one it already describes.

{% if memory %}
{{ memory }}
{% endif %}

{% if archive %}
{{ archive }}
{% endif %}

## The cut is yours to choose

`FLOORPLAN.md` fixes the **width and the dimension**, and that part is binding. The rest is your
judgement about what this module computes: which tensors replicate and which shard, what one rank's
output actually is, and whether a term belongs inside a rank's partial or after the ranks rejoin.

Nothing here catches a wrong cut — stage 4 does, when your {{ factor }} ranks will not reassemble
and a whole optimization loop has gone into the wrong subproblem. So before you commit, build a
host-side torch version of the cut in a scratch file, dump its intermediates, and check them against
`module/tensors/`.

## What you must produce

**`source.py`** — one top-level function named `{{ entry_point }}` taking its tensors as arguments
and returning this rank's output. Seed it with the bootstrapped module's kernel narrowed to one
rank, not with a stub: iteration 0 has to capture a real latency, and a baseline that does not run
gives the loop no metric to improve.

**`inference.py`** — the validator, then **frozen for the whole loop that follows**. Every iteration
is judged by it and nothing else, so it deserves more care than the kernel. `module/inference.py` is
a working one for the whole module; write yours to the same contract, against this rank's tensors and
golden, and keep these properties:

  - it loads every tensor from `tensors/` in this repo, at the dtypes and shapes `README.md` records;
  - it traces the *function* `{{ entry_point }}`, so a later iteration may reimplement it in torch
    instead of NKI without the validator changing;
  - it prints `##autohelix[latency_ms=<number>]` from a fresh `neuron-explorer` profile, leaving a
    `.neff` and a `.ntff` behind;
  - it prints `##autohelix[passed=1]` and exits 0 on a match, non-zero otherwise, and prints
    `##autohelix[max_abs_err=<number>]`, failing when that exceeds `MAX_ABS_ERR`;
  - it declares `RTOL`, `ATOL`, `MIN_COSINE`, `MIN_PASS_FRACTION` and `MAX_ABS_ERR` as module-level
    number literals, at exactly the values `SUBMODULE.md` states;
  - it imports only the standard library, torch, torch_neuronx, numpy, nki and `source`, opens
    nothing outside this repository, and runs under `NEURON_RT_NUM_CORES=1`.

**`tensors/`** — this rank's input, weights and golden, as raw little-endian bytes with no header,
**sliced from `module/tensors/` byte-for-byte**. Do not regenerate them, round-trip them through a
framework, or synthesize them. If your cut needs an intermediate the recorded tensors do not hold, a
router decision or a partial sum, produce it by *running* the frozen reference or the bootstrapped
kernel, and say in `SUBMODULE.md` which one you used and why you trust it.

**`submodule.json`** — the declaration the next stage reads:

```json
{
  "module": "{{ module }}",
  "dim": "{{ dim }}",
  "factor": {{ factor }},
  "shard": 0,
  "entry_point": "{{ entry_point }}",
  "inputs":  [{"name": "...", "file": "tensors/....bin", "dtype": "...", "shape": [...],
               "replicated": true}],
  "outputs": [{"name": "...", "file": "tensors/....bin", "dtype": "...", "shape": [...]}],
  "reassembly": {
    "op": "sum",
    "shards": ["goldens/rank0.bin", "goldens/rank1.bin", "goldens/rank2.bin", "goldens/rank3.bin"],
    "then_add": [],
    "dtype": "float32",
    "shape": [...]
  }
}
```

`reassembly` is a claim the gate **checks arithmetically**: combining those goldens the way it says
must reproduce the module's recorded output at the module's own bar. So dump all {{ factor }} ranks'
goldens into `goldens/`, not just rank 0's. Use `op: sum` for partials that rejoin by reduction or
`op: concat` with a `dim` for shards that rejoin by gathering, and put in `then_add` any term that
belongs after the ranks rejoin rather than inside a rank's partial.

The shard list needs **exactly {{ factor }} shards**, **all distinct**, and **none of them the
module's own recorded output** — reassembling the answer from the answer proves nothing. A shard's
shape may be a flattened `(tokens, dim)`; the check reshapes when the element count matches.

**`SUBMODULE.md`** — for a human: what one rank computes, which tensors replicate and which shard,
where each golden came from, how you verified it, and the five tolerance constants with their
derivation. Derive them from *this* rank's own recorded output using the rule in
`reference/reference_numerics.py`. The submodule's output is often a different dtype from the
module's, so the module's bar would be the wrong bar.

**`.autohelix/optimization/submodule.json`** — the manifest. **Edit the file already there**; its
current contents show the shape. The pipeline owns every field except two and restores the rest
after you finish, so an edit elsewhere in it is put back. Fill `tolerance` with the five constants,
and `tensors` with one entry per `.bin` your validator loads, each carrying `bytes` and a `sha256`.
The hash is optional, but it is what makes a golden edited later detectable, so record it.

## When you are done

Run `inference.py` and confirm it exits 0, prints all three markers, and leaves a fresh `.neff` and
`.ntff`. Then run the gate this repository will face:

    python -m optimization.submodule_checker --repo .

It runs seven checks — shape, self-containment, measurement, baseline passes, data provenance,
declared cut, single core — and names whatever is missing. A repository that cannot pass its own
validator and its own gate is not a starting point.
