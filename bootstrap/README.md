# Bootstrapping Trainium NKI kernels

`autohelix bootstrap` takes one module of a published partition artifact, turns it into a
self-contained git repo, and loops an agent on it until the repo holds a working NKI kernel
with a validator that proves it reproduces the recorded reference.

It is the AutoHelix loop with the optimization removed. There is no metric, because until a
kernel exists there is nothing to measure; the only thing to satisfy is a fixed constraint,
and the constraint is red at the start by construction.

```bash
autohelix bootstrap init <repo> --artifact <artifact-dir> --module layers.0.attention
autohelix bootstrap run  --path <repo>
autohelix bootstrap check --path <repo>    # run the gate once, by hand
autohelix bootstrap report --path <repo>   # per-iteration verdicts
```

## How it differs from `autohelix run`

| | `autohelix run` | `autohelix bootstrap run` |
|---|---|---|
| metric | required, ranks the run | none |
| baseline | constraints pass | constraint fails, by design |
| failing iteration | worktree discarded | **merged anyway** |
| reviewer | only after a constraint passes | every iteration |
| stops when | budget, or a metric gate | the gate passes, or budget |

The merge is the load-bearing difference. Upstream, a rejected iteration leaves only its
notes behind — which is right when a working codebase is being improved and wrong here,
because every early iteration fails and the loop would restart from the stub forever.
Bootstrapping ratchets instead: `git log` is the record of how the kernel arrived, one
commit per iteration, each recording which checks were green at the time.

## The gate

`bootstrap/nki_checker.py`, run as a single constraint with a 1200s timeout. Six checks,
all of which must pass:

| | check | what it means |
|---|---|---|
| a | kernel formalization | `source.py` defines a top-level `kernel` under `@nki.jit`; `inference.py` reaches it through `torch_neuronx.trace` |
| b | self-containment | neither file imports or opens anything beyond the other, the standard library, NKI, and the repo's own `.bin` tensors |
| c | nki-only | `source.py` never names torch, numpy or scipy |
| d | metric measurement | the run leaves a `.neff` and a `.ntff` and reports a latency read from `neuron-explorer`'s `total_exec_time` |
| e | pass-test | `inference.py` exits 0 and matches the reference, at the pinned tolerance and not a looser one |
| f | data provenance | the tensors fed to the kernel are the recorded ones, byte for byte |

**The agent never sees the checker.** `bootstrap/preset.yaml` states all six requirements in
prose as its `goal`, and that file stays in the package — the loop reads it directly and no
copy is written into the module repo, so the constraint command that names the checker is
never in the worktree. That makes `preset.yaml` and `nki_checker.py` a pair: anything the
checker enforces and the goal does not say is a trap rather than a requirement, and they
must be changed together. `tests/test_bootstrap_driver.py` fails if they drift on the
tolerances, the markers or the banned constructs.

## The config is a fixed file

`bootstrap/preset.yaml` is the config, for every module repo, unmodified. Nothing generates
it, nothing substitutes into it, and `init` writes no config at all — so the preset you read
in the diff is exactly the preset that runs, and editing it changes the next run with no
re-init. `--config` on `bootstrap run` points at an alternative if you want to try a variant
without touching the reviewed one.

Two consequences of being fixed. The gate's command cannot name a per-repo path, so the
checker finds its own manifest by walking up from `--repo` — an iteration worktree lives at
`<repo>/.autohelix/worktrees/iter-N`, inside the repo, so `.autohelix/bootstrap/manifest.json`
is always above it. And the command says `python` rather than an absolute interpreter, so
`bootstrap run` probes up front that `python` on PATH can import `bootstrap`, `torch` and
`torch_neuronx`, and refuses to start rather than failing mid-iteration.

Latency is measured but not targeted. Check (d) exists to prove the profiling path works;
no gate and no ranking depends on the number. (Declaring it as a metric would also make
baseline capture fail at iteration 0, before a kernel exists to measure.)

### What it refuses

The interesting half of a hidden constraint is its refusals, and
`tests/test_bootstrap_checker.py` is mostly negative tests. Among them: a kernel that opens
`tensors/reference.bin` and hands the answer back (the kernel may do no file I/O at all,
precisely because that file is legitimately in the manifest); any of the four tolerance
constants raised, lowered, deleted, or computed from an expression rather than written as a
literal; a `.neff` left over from an earlier run rather than produced by this one; tensors
fabricated with `randn`/`ones`/`full`/`arange`/`fill_`; an edited `.bin`.

What static analysis cannot catch — a kernel that is an identity with the real work done on
the host, a comparison that is technically performed but meaningless — is the reviewer's
job. It runs every iteration and writes two sections: what is left to satisfy the gate, and
an adversarial read of whether this iteration is circumventing it.

## The repo `init` builds

Materialized, not copied: a published module directory imports the harness runtime and the
vendor's modeling package and reads its weights from a 475 GiB checkpoint, none of which
survives check (b).

| | |
|---|---|
| `source.py` | **editable** — the kernel. Starts as a `@nki.jit` stub that raises. |
| `inference.py` | **editable** — the validator. Starts as a skeleton with the tolerance constants and the tensor table filled in. |
| `tensors/*.bin` | frozen — input, reference output, and every weight, as raw little-endian bytes. No header and no sidecar: the dtype and shape are in `README.md`. |
| `reference_torch.py` | frozen — the original PyTorch implementation, verbatim. The specification, unimportable. |
| `README.md`, `MODULE.md`, `config.json` | frozen — the computation, the tensor table, the module's own pre- and post-conditions. |
| `.autohelix/` | gitignored — the tensor manifest (the only thing `init` writes here), plus notes, reviews and per-iteration verdicts once the loop runs. |

`scope.editable` is `[source.py, inference.py]`, so an edit to anything else is reverted
before the gate runs.

Weights are materialized at their recorded dtype, fp8 included — `layers.0.attention` has
`float8_e4m3fn` weights with `float8_e8m0fnu` block scales. A `complex64` buffer
(`freqs_cis`) is written as `float32` with a trailing `[real, imag]` dimension, since NKI
has no complex dtype; `README.md` says so per tensor.

## Choosing a target

`--module` takes any module id in the artifact and `--group` is found from it. Only
`--sample`/`--step` combinations the run actually recorded are selectable: `bootstrap init`
fails with the available keys otherwise.

The default, `layers.0.attention`, is the smallest correct kernel in group `00-Attention`:
layer 0 is the one member at `compress_ratios[0] == 0`, so it is pure sliding-window
attention with no compressor, no indexer and no compressed-KV concatenation. The other
thirty members of that group need the compressed branch, so a kernel bootstrapped here
covers the SWA path only.

Note that the DeepSeek V4.1 Flash artifact records **prefill only** for attention — step
suffixes across the whole artifact are `{0: 255, 1: 33}`, and every `#1` belongs to
`42-DSparkMarkovHead`. There is no recorded decode call to validate an attention decode
kernel against, and check (f) forbids synthesizing one.
