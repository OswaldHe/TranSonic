Goal: {{ goal }}

Your working directory is {{ worktree }}. Don't modify files outside it.

This is iteration {{ iteration }} of the **{{ stage }}** stage. Baseline is iter-0.

{{ iteration_constraint }}

{% if constraint_schedule %}
The whole schedule, so you know what this iteration is for and what the next ones allow:

{{ constraint_schedule }}
{% endif %}

{% if best_so_far %}
Best {{ metric }} so far: **{{ best_so_far }}**. An iteration more than 5% above it is rejected
and discarded — so a change you are unsure about belongs behind a measurement, not in the commit.
{% endif %}

{% if history_summary %}
Recent history:
{{ history_summary }}
{% endif %}

{% if observables %}
Measured after you finish:
{% for obs in observables %}
  {{ obs.command }}{% if obs.metric_labels %} → {{ obs.metric_labels | join(', ') }}{% endif +%}
{% endfor %}
{% endif %}

{% if editable %}
**You may edit only: {{ editable | join(', ') }}.** Everything else is reverted before your work is
judged — including the validator. That is deliberate: the validator defines what correct means and
how latency is measured, so freezing it is what makes one iteration's number comparable to
another's. If you believe the validator's input format is holding the kernel back, do not change
it — say so in your notes, and the reviewer will carry it to the operator.
{% endif %}

Read before starting:
{% if has_reviewer %}
- .autohelix/review.md — last iteration's review
{% endif %}
- .autohelix/notes/iter-*.md — your own notes from past iterations
- README.md and, if present, FLOORPLAN.md — what this module is and how it is meant to be split
{% if observables %}
- .autohelix/observations/iter-*/ — captured measurement output, one dir per iteration
{% endif %}
{% if has_hints %}
- .autohelix/hints.md — notes from the operator
{% endif %}

The `neuron-nki-*` agents and skills available to you know the NKI API, the compiler's errors and
the profiler. Use them rather than guessing at syntax — a wasted iteration on a compile error is
the most common way this loop loses ground.

{% if iteration_time %}
Time budget: {{ iteration_time }}. The process will be killed at the deadline.
To check remaining time: bash "$AUTOHELIX_TIME_LEFT_SCRIPT"
Stop new experiments before the deadline, write notes, and exit cleanly.
{% endif %}

You MUST write notes when done — write to {{ worktree }}/.autohelix/notes/iter-{{ iteration }}.md.
Record what you tried, what the profile said, which NKI idioms compiled and which did not, and
what to try next. Notes persist even when your changes are rejected, so they are the only way an
experiment that did not land still moves the run forward.

Make it faster without making it wrong.
Do not create git commits; AutoHelix commits accepted changes for you.
When done, write a one-line summary of what you changed and why to {{ worktree }}/.autohelix/commit_summary.txt
