You are writing the feedback this optimization run owes the people who build the toolchain.

Two loops have finished on `{{ module }}`. Every measurement the optimizing agents took and every
opinion the reviewers formed is on disk, and nobody has read all of it. Your job is to read all of
it, work out what actually stopped the kernel from getting faster, and write that down in a form the
AWS Neuron team can act on.

## First, the vocabulary

Run the `mattpocock-skills:domain-modeling` skill before you write anything, and work inside its
discipline for the whole task. This report will be read by people who did not do the run, so a term
doing two jobs costs them more than it costs you.

- `{{ context_md }}` is this project's glossary. Use its words as it defines them, and prefer them
  to your own coinages.
- When you need a term the glossary does not have — and you will, because the hardware vocabulary is
  not in there — define it once, in a short "Terms" section at the top of the report, and then use
  it consistently. One name per concept, one concept per name.
- The failure to avoid is term overloading. "Core", "rank", "unit", "device", "tile" and "block" all
  mean at least two things in this toolchain. Say which one you mean every time, and if a note you
  are quoting is ambiguous, resolve the ambiguity or say that you could not.

## What to read

{{ corpus }}

That is roughly {{ words }} words. Read all of it. Notes are the optimizing agent's own record of
what it tried and what the profile said; reviews are a second agent's opinion of that work, written
without the power to change it.

Also available, and worth reading:

- `{{ submodule_repo }}/source.py` and `{{ full_repo }}/source.py` — the kernels as they ended up.
  Their comments record measured sweeps, which is often the hardest evidence in the run.
- `{{ full_repo }}/inference.py` — the frozen validator, including how latency is measured.
- `{{ workspace_root }}/REPORT.md` — what the run achieved, and the two caveats on its numbers.

## The run, for context

- bootstrapped module, one core: **{{ bootstrap_latency }}**
- optimized submodule, one core: **{{ submodule_latency }}**
- assembled module, {{ ranks }} ranks: **{{ full_latency }}**

## Reconcile before you conclude

The notes were written one iteration at a time by an agent that did not know how the run would end,
so the corpus contradicts itself. Iteration 2 records a theory iteration 7 disproves. A reviewer
doubts a claim a later profile confirms. A workaround found early is superseded twice and the early
note still reads as current.

Where two statements disagree:

1. **Prefer the one with a measurement.** A number from a profile outranks a inference from reading
   the code, whoever wrote it and whenever.
2. **Prefer the later one only when it is also measured.** Recency alone settles nothing.
3. **Say that there was a conflict**, in the row's root-cause cell or in a footnote, and say how you
   settled it. A reader who later finds the losing note needs to know it was considered and why it
   lost, or they will reopen the question.
4. **When you cannot settle it, file it as unsettled** and say what measurement would settle it.
   That is a useful finding. A confidently wrong one is not.

Check a claim before you file it. You have the device, the kernels and the validator — if a note
says an API rejects something, try it. If a claim cannot be checked now, mark the row as
unverified and say so plainly.

## What to write

Write `{{ report_path }}`. Its centre is one table, and every row is one thing that stopped this
kernel from going faster. Use exactly these columns, in this order:

| Level | Scenario | Minimum reproduction | Root cause and why | Related documentation | Suggestion |

**Level** — one of:

{{ levels }}

Pick by who has to fix it. L0 is a promise the toolchain broke, and the fix is a code change in the
compiler or the runtime. L1 is a capability nobody claimed and nobody has, and the fix is a feature.
L2 is silicon, and the fix is a future chip or nothing. If a row could be two levels, say which and
why in the root-cause cell — but pick one for the column.

**Scenario** — what was actually happening. Which part of the kernel, which engine, what the profile
said, what the utilization was, what you expected instead. Written so a Neuron engineer who has
never seen this model can picture it without asking a question. Name the numbers.

**Minimum reproduction** — a path to a runnable file you write under `{{ repro_dir }}/`, plus the
command to run it. One file per row, the smallest thing that still shows the problem: strip the MoE
down to the two or three operations that matter. Say what it prints when the problem is present and
what it would print if the problem were fixed. If a row is a measured cost rather than a failure, the
reproduction is the benchmark that measures it.

**Root cause and why** — the mechanism, in plain words. Why does this make the kernel slow, in terms
of what the hardware or the compiler is doing. No jargon that the report has not defined. If the
cause is genuinely unknown, say that and give the evidence that narrows it.

**Related documentation** — real links, at least one per row. Search for them; do not guess a URL.
The Neuron SDK documentation (`awsdocs-neuron.readthedocs-hosted.com`), the NKI API reference, the
`aws-neuron/aws-neuron-sdk` issue tracker, `aws-neuron/nki-samples`, release notes. If an issue
already exists for a finding, link it and say whether this run agrees with it. If nothing exists,
say "no existing issue found" and link the documentation page the finding contradicts or exposes.

**Suggestion** — what the Neuron team could do, specifically enough to be actionable. "Improve the
compiler" is not a suggestion. "Accept `float8_e4m3fn` in `nc_matmul` on trn2, or document that only
the legacy `float8_e4m3` is supported and raise a clear error instead of an internal one" is.

Below the table, one section per row, in the same order, with the detail that does not fit a cell:
the profile excerpt, the code, the sweep, the exact error text. The table is the index and the
argument; the sections are the evidence.

## Rules for the rows

- **Every row traces to the corpus.** Cite the note or review it came from, by repository and
  iteration. A row you cannot trace is one you invented, and it will waste the reader's time and
  spend this report's credibility.
- **Merge duplicates.** The same obstacle shows up in several iterations under different names. One
  row, citing all of them.
- **Leave out what you fixed.** If the run found a workaround and the workaround works, that is not
  a blocker — unless the workaround costs something, in which case the cost is the finding and you
  should say what it cost.
- **Leave out your own mistakes.** An iteration that misread an API and lost an hour is not Neuron's
  problem. An API that is easy to misread in a way the error message does not reveal is.
- **Order by what unblocks the most.** Biggest latency left on the table first, not L0 first.
- **Say how many iterations were spent on each.** The cost to us is the strongest argument for the
  fix being worth their time.

## When you are done

Run `python -m optimization.feedback --check --root {{ workspace_root }}` and fix anything it
reports. It checks the shape of the report, never whether a finding is true: that every row has a
level in range, a reproduction file that exists, a link, and cells long enough to be read.

Then write, at the top of the report, how many findings there are at each level and the one sentence
you would say to the Neuron team if they read nothing else.
