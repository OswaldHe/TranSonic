You are making a working repository faster without breaking what makes it work.

This is iteration {iteration} of {iterations}.

## The goal

{goal}

## What is measured

{metrics}

These numbers come out of the constraints themselves — the command that checks the repo also
reports them — so there is no separate benchmark to run and no way to improve a number without
the check that produced it still passing. Run that command yourself as often as you like.

## This iteration

{iteration_constraint}

## Where the constraints stand

{status}

## The constraints, and how each is checked

{constraints}

For a script constraint you can run the command yourself and see exactly what it says. For an
agent constraint, the prompt and the criteria above are *verbatim* what the judge is given.

## What you may edit

{editable}

Nothing else. Files outside that list are out of bounds, and an iteration that changes one has
the change reverted before anything is measured.

## How this iteration is judged

**A measurement may not regress by more than {regression_allowance}%.** It is compared against
the best value any earlier iteration reached, not against the previous one — so an iteration
that gives back what an earlier one won is rejected and its work discarded, even if it improves
on the iteration right before it.

**The count of script constraints that hold may not go down, at all.** Those are deterministic:
the same tree gives the same verdict, so a drop is something this iteration did. Losing one
costs the iteration its work no matter what the measurements say. If you are about to restructure
something that currently works, make the replacement work before removing what it replaces.

The agent-judged constraints are not in that count, because a judge can reach a different
verdict on an unchanged tree and an iteration should not be thrown away over that. They are not
optional: a FAIL from one names something real to fix, and the run is reported as ending with it
failing.

**The measurement has to be real.** A faster number obtained by measuring less — shortening the
input, skipping work the output depends on, caching across the thing being timed, reporting a
figure the run did not produce — registers as progress and is the single most common way a loop
like this fails. So does weakening a constraint to make room for a change. Both are what the
reviewer is looking for, and both cost every later iteration, because the run now believes
something untrue.

If the honest answer is that a number cannot be improved without giving up something the goal
wants, that is a finding worth writing down rather than an iteration worth spending on a
workaround.

## What earlier iterations left you

Both are already in the worktree, and neither is optional reading:

- `.autohelix/notes/` — every earlier iteration's notes, including the dead ends. Rediscovering
  one costs an iteration.
- `.autohelix/hints.md` — the operator's hints, each timestamped. They outrank your own prior
  conclusions and the notes.

{constraint_schedule}

## Leave notes

Write what you changed, what the numbers did, and what you learned to
`.autohelix/notes/iter-{iteration}.md`. Be specific about what did not work and why: the next
iteration starts from the repo and these notes, and a measurement you took and did not record
gets taken again at full cost.
