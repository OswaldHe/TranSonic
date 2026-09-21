# Sample input sets

The prompts the loop traces, verifies and emulates against. Plain JSONL — one JSON
object per line, no comments — so any reader can parse them.

| file | what it is |
|---|---|
| `short.jsonl` | complete instructions, roughly 130–190 tokens (committed) |
| `long.jsonl` | long-context prompts at 2k / 8k / 16k tokens (committed) |
| `long.provenance.json` | how `long.jsonl` was generated, written beside it |
| `fetch_long_inputs.py` | regenerates `long.jsonl` reproducibly |

## Fields

| field | meaning |
|---|---|
| `id` | stable identifier; names the sample's artifacts under `trace/` |
| `prompt` | the text |
| `role` | `user` applies the tokenizer's chat template if it has one; `raw` is a plain continuation |
| `target_tokens` | optional; pins the tokenized length, truncating or repeating to hit it |
| `source` | provenance, for the report |
| `expected_substring` | optional; what a correct answer should contain, for a human reading the sampled tokens |

`short.jsonl` deliberately sets no `target_tokens`. Pinning an exact length
truncates a short instruction mid-sentence, and the judge would then be assessing a
continuation of a broken prompt. The long set does pin lengths, because there the
tail is filler and safe to cut.

## Which samples a run uses

A spec's `inputs` section points at the two files and caps how many of each are
used:

```yaml
inputs:
  short: ../../inputs/short.jsonl
  long: ../../inputs/long.jsonl
  long_token_budgets: [2048, 8192, 16384]
  max_short: 4     # use at most 4 short prompts
  max_long: 3      # use at most 3 long prompts
```

`max_short` and `max_long` exist because cost scales with the number of samples:
every sample means another full forward during tracing, another set of dumped
feature maps for every module, and another generation pass during emulation. Lower
them to make a run cheaper; raise them for broader coverage. Omitted means no cap —
every prompt in the file is used.

## Regenerating the long set

```bash
python fetch_long_inputs.py --out long.jsonl                    # synthetic, reproducible
python fetch_long_inputs.py --out long.jsonl --source longbench # from the Hub
```

Synthetic prompts are needle-in-a-haystack and multi-hop retrieval, built from a
seed so the same command reproduces the same file. `--source longbench` pulls real
long-context samples instead and needs network access.
