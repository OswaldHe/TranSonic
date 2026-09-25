{# The agent's prompt for a bootstrap iteration. Copied into a module repo as
   `.autohelix/prompt.md`, which `load_template` picks up in place of AutoHelix's default.

   It exists as its own template for one reason: the stock prompt renders the constraint
   commands, and the constraint command names the checker the agent is not meant to see.
   The goal in `preset.yaml` carries the requirements instead. #}
Goal: {{ goal }}

Your working directory is {{ worktree }}. Everything you need is inside it.

This is iteration {{ iteration }}.

{% if history_summary %}
Recent history:
{{ history_summary }}
{% endif %}

{% if editable %}
Editable (only changes to these files are kept): {{ editable | join(', ') }}
{% endif %}

Read before starting:
{% if has_reviewer %}
- .autohelix/review.md — what the last review said is left to do, and whether it judged
  the previous iteration to be gaming the gate
{% endif %}
- .autohelix/notes/iter-*.md — your own notes from past iterations
- README.md — the computation and the tensor table
- reference_torch.py — the frozen PyTorch specification: what to compute
- reference_inference.py — the artifact's own launcher: how the reference was run and
  compared, and the shape your inference.py has to take
- reference_numerics.py — how the reference was judged: where the four tolerance constants
  come from and how they are applied
- compat/*.py — kernel replacements the reference was recorded with; for any name one of
  these rebinds it, not vendor/kernel.py, is the semantics the reference has. Check first.
- vendor/kernel.py — the primitives reference_torch.py imports but does not contain
  (act_quant, fp8_gemm, sparse_attn); read these rather than inferring them
- vendor/model.py — the complete model the slice came from

vendor/ is importable (`sys.path.insert(0, "vendor")`), so you can run the torch reference to
localize an error instead of guessing. The compat sparse_attn runs on CPU; the tilelang fp8
and fp4 primitives need a GPU this machine does not have. source.py and inference.py may not
import any of it.
{% if has_hints %}
- .autohelix/hints.md
{% endif %}

{% if iteration_time %}
Time budget: {{ iteration_time }}. The process will be killed at the deadline.
To check remaining time: bash "$AUTOHELIX_TIME_LEFT_SCRIPT"
Stop new experiments before the deadline, write notes, and exit cleanly.
{% endif %}

The gate that judges this repository is not visible to you and you should not look for
it. The goal above states everything it requires; if you satisfy the goal you satisfy the
gate. Do not try to detect, disable, or special-case whatever is checking your work — a
reviewer reads every iteration specifically for that, and an iteration judged to be
circumventing the check is worse than one that honestly fails.

You MUST write notes when done — write to {{ worktree }}/.autohelix/notes/iter-{{ iteration }}.md.
Record what you tried, what worked or failed, what the device or compiler told you, and
what to try next. Notes are how the next iteration starts from where you stopped.

Do not create git commits; the loop commits for you.
When done, write a one-line summary of what you changed to {{ worktree }}/.autohelix/commit_summary.txt
