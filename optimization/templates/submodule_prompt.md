You are cutting one module of a Trainium model down to the part a **single NeuronCore** runs, so
that a later loop can optimize that part in isolation and a stage after it can put the ranks back
together with a collective.

This is preparation, not optimization. Nothing here is measured against a target. What matters is
that the repository you leave behind is a correct, self-contained, *measurable* starting point — and
that you have written down how your cut reassembles into the whole module, because the stage after
this one reads that and cannot proceed without it.

## What you are given

`{{ repo }}` already holds, all frozen:

| | |
|---|---|
| `module/` | the bootstrapped module: its working NKI `source.py`, its `inference.py`, its `README.md`, and `tensors/` — every input, weight and the golden output, as recorded bytes |
| `FLOORPLAN.md` | how the ranked floorplan places this module, and the projection onto this one device. **Read this second, after `module/README.md`.** |
| `reference/` | the artifact's frozen references: `reference_torch.py` (what the module computes), `reference_inference.py` (how it was run), `reference_numerics.py` (what counts as matching), plus `vendor/` and `compat/` |

The bootstrapped module is the authority on *results*: whatever you build has to be consistent with
the bytes in `module/tensors/`. `reference_torch.py` is the authority on *semantics* — read it to
find out what a per-rank computation even is for this module. For a model whose reference already
carries a `rank`/`world_size` notion, the cut you need may be the one the reference itself describes.

{% if memory %}
{{ memory }}
{% endif %}

## The cut is yours to choose

There is no script that will tell you how to divide this module, because how to divide a module is a
judgement about what it computes. `FLOORPLAN.md` tells you the **width and the dimension**: that is
binding. Everything else — which tensors are replicated and which are sharded, what one rank's
output actually is, whether a term belongs inside a rank's partial or is added after the ranks
rejoin — is yours to work out from the reference.

Get this wrong and it is not caught here. It is caught two stages later, when your {{ factor }} ranks cannot
be reassembled into the module's recorded output, after a full optimization loop has been spent on
the wrong subproblem. So spend the time now: build a host-side torch version of your cut in a
scratch file, dump its intermediates, and check them against `module/tensors/` before you commit to
anything.

## What you must produce

**`source.py`** — the submodule kernel, exposing one top-level function named `{{ entry_point }}`
that takes its tensors as arguments and returns this rank's output. Seed it with the *bootstrapped
module's kernel narrowed to one rank*, not with a stub: iteration 0 of the optimization loop has to
capture a real latency, and a loop whose baseline does not run has no metric to improve.

**`inference.py`** — the validator, and then it is **frozen for the entire optimization loop**. Every
iteration will be judged by it and by nothing else, so it is worth more care than the kernel:

  - load every tensor from `tensors/` in this repo, using the dtypes and shapes `README.md` records;
  - trace `{{ entry_point }}` from `source.py` and run it on the device. Trace the *function*, so a
    later iteration may reimplement it in torch instead of NKI without the validator changing;
  - dump a `.neff` and a `.ntff`, read the latency out of `neuron-explorer`, and print it as
    `##autohelix[latency_ms=<number>]`;
  - compare against this rank's golden and print `##autohelix[passed=1]` on a match, exiting 0, and
    non-zero on a mismatch;
  - print `##autohelix[max_abs_err=<number>]` and fail when it exceeds `MAX_ABS_ERR`;
  - declare `RTOL`, `ATOL`, `MIN_COSINE`, `MIN_PASS_FRACTION` and `MAX_ABS_ERR` as module-level
    number literals with exactly the values in `SUBMODULE.md` under "The numerical bar";
  - import nothing beyond the standard library, torch, torch_neuronx, numpy, nki and `source`, and
    open nothing outside this repository.

It runs with `NEURON_RT_NUM_CORES=1` and must not try to change that.

**`tensors/`** — this rank's input, weights and golden, as raw little-endian bytes with no header,
**sliced from `module/tensors/` byte-for-byte**. Do not regenerate, round-trip through a framework,
or synthesize them. If your cut needs an intermediate the module's recorded tensors do not contain —
a router decision, a partial sum — derive it by *running* the frozen reference or the bootstrapped
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

`reassembly` is a claim that will be **checked arithmetically**: the {{ factor }} per-rank goldens it
names, combined the way it says, must reproduce the module's recorded output at the module's own bar.
So dump all {{ factor }} ranks' goldens into `goldens/`, not just rank 0's. `op` is `sum` (partials
that rejoin by reduction) or `concat` with a `dim` (shards that rejoin by gathering); `then_add`
names terms that belong *after* the ranks rejoin rather than inside any rank's partial.

Three things the shard list has to satisfy, because the check is only worth running if they hold:
**exactly {{ factor }} shards**, one per rank; **all distinct**; and **none of them the module's own
recorded output**. Reassembling the answer from the answer would reproduce the target perfectly and
prove nothing about the cut.

A shard's shape may be the flattened `(tokens, dim)` rather than the module's `(batch, tokens, dim)`
if that is what one rank actually produces — the comparison reshapes when the element count matches,
because the reference itself flattens and restores.

If that check fails, your cut does not close, and finding out here costs you minutes instead of
costing the pipeline two stages.

**`SUBMODULE.md`** — for a human: what one rank computes, which tensors are replicated and which are
sharded, where each golden came from, how you verified it, and the five tolerance constants with the
derivation that produced them. Derive them from *this* rank's own recorded output using the rule in
`reference/reference_numerics.py` — the submodule's output is often a different dtype from the
module's, so inheriting the module's bar would be the wrong bar.

**`.autohelix/optimization/submodule.json`** — the manifest. **Edit the file that is already
there**; do not rewrite it from scratch. It arrives carrying `module`, `entry_point`, `projection`,
`module_output`, `module_tolerance` and `module_tensors`, all of which the pipeline owns and restores
after you finish — anything you change there is put back. Two fields are yours to fill:

```json
{
  "tolerance": {"RTOL": 0.1, "ATOL": 0.1, "MIN_COSINE": 0.9999,
                "MIN_PASS_FRACTION": 0.999, "MAX_ABS_ERR": 0.0},
  "tensors": {
    "tensors/input.bin":  {"sha256": "...", "bytes": 1310720},
    "tensors/golden.bin": {"sha256": "...", "bytes": 1310720}
  }
}
```

`tensors` may be keyed by relative path as above, or keyed by tensor name with a `"file"` field in
each entry — either is read. List every `.bin` your validator loads. `sha256` is optional but is what
makes a golden edited later detectable, so record it.

## When you are done

Run `inference.py` yourself and confirm it exits 0, prints all three markers, and leaves a fresh
`.neff` and `.ntff`. A repository that does not pass its own validator is not a starting point.
