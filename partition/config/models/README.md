# Model specs

One YAML file per model. Everything model-specific lives here; the loop itself
([`../../README.md`](../../README.md)) knows nothing about any particular model.

```bash
autohelix partition list                # what is bundled
autohelix partition inspect qwen3.5-0.8b
autohelix partition run qwen3.5-0.8b
```

A spec argument can also be a path to a YAML file anywhere, a HuggingFace repo id
(`Qwen/Qwen3.5-0.8B`), or a local directory — a bare repo id needs no spec file at
all, since the loader and entry module are detected.

## Bundled specs

| Spec | Checkpoint | Fits one 46 GiB GPU? | Purpose |
|---|---|---|---|
| `qwen3.5-0.8b` | 1.6 GiB | yes | smoke test; hybrid attention, MTP, tied embeddings |
| `qwen3.8-27b` | 51.7 GiB | no | 64 layers, 2 signatures, hidden 5120 |
| `qwen3.5-35b-a3b` | 71.9 GiB | no | MoE: expert groups and routing capture |
| `deepseek-v4-flash` | 159.6 GB | no | fp8 weights with fp4-packed experts, 43 layers, 256 experts |
| `deepseek-v4.1-flash` | 510.3 GB | no | 40 layers + vision, 384 + 128 experts, sparse attention |

"Fits" means the whole checkpoint can be resident on the GPU at once. A model that
does not fit is still partitioned, traced and verified: modules are verified one at a
time on the GPU, and whole-model stages spread layers across GPU and host.

## Writing a spec

```yaml
name: my-model                 # artifact directory name
source: hf:org/Model           # HF repo id, or a local directory path
revision: <sha>                # optional; resolved and pinned automatically
loader: auto                   # auto | transformers | repo_code
dtype: bfloat16
trust_remote_code: false       # required to import a repo's own inference code

scope:                         # structural parts this run owns
  vision: false                # excluded parts are recorded in the graph as
  mtp: false                   # unpartitioned nodes, not traced or emulated
  engram: false

partition:                     # how you want it cut up; see the main README
  split_attention_ffn: true
  prompt: |
    ...

inputs:                        # see ../../inputs/README.md
  short: ../../inputs/short.jsonl
  long: ../../inputs/long.jsonl
  max_short: 4
  max_long: 3

overrides:                     # per-model loop defaults, e.g. cache_dequant: false
  cache_dequant: false

enabled: true                  # false makes `run` refuse, with `notes` as the reason
notes: |
  Anything a reader needs to know before running this.
```

With `loader: auto` a repo that ships its own inference code gets `repo_code` and
its entry module is detected; otherwise `transformers`. Vendor code is preferred
because it defines the model's numerics and may cover architectures transformers does
not know — importing it requires `trust_remote_code: true`.

Relative input paths resolve against the spec file, which is why the bundled specs
say `../../inputs/...` and why an installed wheel keeps `config/` and `inputs/` in
the same relative position.

## Notes on the DeepSeek specs

Both are the intended deployment target and both are larger than this machine can
hold, so what is reachable differs by stage:

| | V4-Flash | V4.1-Flash |
|---|---|---|
| transformers support | yes (`deepseek_v4`) | **no** — `deepseek_v41` is not a registered architecture |
| vendor inference code | `inference/model.py` | `inference/model.py` |
| vendor code runnable as-is | no: needs `convert.py` to a per-rank checkpoint | no: same |
| `inspect` / `plan` | yes, metadata only | yes, metadata only |
| `trace` onward | needs the whole model resident | needs the whole model resident |

Both use fp8 blockwise weights (V4: 128×128 blocks; V4.1: 32×32, both with `ue8m0`
scales) and fp4-packed experts, so `cache_dequant: false` is set on both — a bf16
mirror of those weights would be several times the checkpoint.

`encoding/tests/test_input_*.json` → `test_output_*.txt` in both repos give free
tokenizer ground truth and are worth checking before anything else.
