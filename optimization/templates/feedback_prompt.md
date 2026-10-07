You are writing the feedback this optimization run owes the people who build the toolchain.

Two loops have finished on `{{ module }}`. Every measurement the optimizing agents took and every
opinion the reviewers formed is on disk, and nobody has read all of it. Read all of it, work out
what actually stopped the kernel from getting faster, and write that down so the AWS Neuron team can
act on it.

## First, the vocabulary

Run the `mattpocock-skills:domain-modeling` skill before you write anything, and stay inside its
discipline. People who did not do the run will read this report, so a term doing two jobs costs them
more than it costs you.

`{{ context_md }}` is this project's glossary: use its words as it defines them. When you need a term
it does not have, and you will because the hardware vocabulary is not in there, define it once in a
short "Terms" section at the top and then use it consistently. The failure to avoid is overloading —
"core", "rank", "unit", "device", "tile" and "block" each mean at least two things in this toolchain
— so say which one you mean every time. Where a note you quote is ambiguous, resolve the ambiguity
or say that you could not.

## What to read

{{ corpus }}

That is roughly {{ words }} words, and you should read all of it. A note is the optimizing agent's
own record of what it tried and what the profile said; a review is a second agent's opinion of that
work, written without the power to change it.

Also worth reading:

- `{{ submodule_repo }}/source.py` and `{{ full_repo }}/source.py` — the kernels as they ended up.
  Their comments record measured sweeps, often the hardest evidence in the run.
- `{{ full_repo }}/inference.py` — the frozen validator, including how it measures latency.
- `{{ workspace_root }}/REPORT.md` — what the run achieved, and the two caveats on its numbers.

{% if archive %}
## What is already filed

{{ archive }}

{% endif %}
## The run, for context

- bootstrapped module, one core: **{{ bootstrap_latency }}**
- optimized submodule, one core: **{{ submodule_latency }}**
- assembled module, {{ ranks }} ranks: **{{ full_latency }}**

## Reconcile before you conclude

An agent wrote each note one iteration at a time without knowing how the run would end, so the
corpus contradicts itself: iteration 2 records a theory iteration 7 disproves, a reviewer doubts a
claim a later profile confirms, a workaround gets superseded twice while the early note still reads
as current.

Where two statements disagree, prefer the one with a measurement, because a profile number outranks
an inference from reading code whoever wrote it and whenever. Prefer the later one only when it is
also measured, since recency settles nothing on its own. Say in the row that there was a conflict and
how you settled it: a reader who later finds the losing note needs to know somebody weighed it, or
they reopen the question. Where you cannot settle it, file it as unsettled and say what measurement
would. That is a useful finding; a confidently wrong one is not.

Check a claim before you file it. You have the device, the kernels and the validator, so if a note
says an API rejects something, try it. Mark a row unverified, plainly, when you cannot check it now.

## What to write

Write `{{ report_path }}`. Its centre is one table, one row per thing that stopped this kernel from
going faster, with exactly these columns in this order:

| Level | Scenario | Minimum reproduction | Root cause and why | Related documentation | Suggestion |

**Level** — one of:

{{ levels }}

Pick by who has to fix it. L0 is a promise the toolchain broke, so the fix is a code change in the
compiler or the runtime. L1 is a capability nobody claimed and nobody has, so the fix is a feature.
L2 is silicon, so the fix is a future chip or nothing. Where a row could be two levels, pick one for
the column and say in the root-cause cell which other one it could be, and why.

**Scenario** — what was actually happening: which part of the kernel, which engine, what the profile
said, what you expected instead. Name the numbers. Write it so a Neuron engineer who has never seen
this model can picture it without asking a question.

**Minimum reproduction** — a runnable file you write under `{{ repro_dir }}/`, one per row, plus the
command. Make it the smallest thing that still shows the problem, and say what it prints now and
what it would print if the problem were fixed. Where a row is a measured cost rather than a failure,
the reproduction is the benchmark that measures it.

**Root cause and why** — the mechanism in plain words: what the hardware or the compiler is doing
that makes the kernel slow. Use no term the report has not defined. Where the cause is genuinely
unknown, say so and give the evidence that narrows it.

**Related documentation** — at least one real link per row, searched for rather than guessed. The
Neuron SDK docs (`awsdocs-neuron.readthedocs-hosted.com`), the NKI API reference, the
`aws-neuron/aws-neuron-sdk` issue tracker, `aws-neuron/nki-samples`, release notes. Where an issue
already exists, link it and say whether this run agrees. Where none does, say "no existing issue
found" and link the documentation page the finding contradicts.

**Suggestion** — specific enough to act on. "Improve the compiler" is not a suggestion. "Accept
`float8_e4m3fn` in `nc_matmul` on trn2, or document that only the legacy `float8_e4m3` is supported
and raise a clear error instead of an internal one" is.

Below the table, put one section per row in the same order, carrying what does not fit a cell: the
profile excerpt, the code, the sweep, the exact error text. The table is the index and the argument;
the sections are the evidence.

## Rules for the rows

- **Every row traces to the corpus.** Cite the note or review it came from, by repository and
  iteration. A row you cannot trace is one you invented, and it spends this report's credibility.
- **Merge duplicates, within this run and across runs.** The same obstacle appears in several
  iterations under different names, so file one row citing all of them. Where an *earlier run*
  already filed it, do not file it again; rule on it in the previously-filed table instead.
- **Leave out what you fixed.** A workaround that works is not a blocker, unless the workaround
  costs something — then the cost is the finding, and say what it cost.
- **Leave out your own mistakes.** An iteration that misread an API and lost an hour is not Neuron's
  problem. An API that is easy to misread in a way the error message hides is.
- **Order by what unblocks the most**, biggest latency left on the table first, not L0 first.
- **Say how many iterations went into each.** What it cost us is the strongest argument that fixing
  it is worth their time.

## When you are done

Run `python -m optimization.feedback --check --root {{ workspace_root }}` and fix what it reports.
It checks the report's shape and never whether a finding is true: that every row has a level in
range, a reproduction file that exists, a link, and cells long enough to read.
{% if archive %}
It checks your previously-filed table the same way. Every `Filed` cell has to name a 12-character
finding id that is in the archive, and every `Ruling` has to be one of {{ rulings }}. Take the ids
from the archive's `README.md` rather than retyping them.
{% endif %}

Then write, at the top of the report, how many findings there are at each level and the one sentence
you would say to the Neuron team if they read nothing else.
