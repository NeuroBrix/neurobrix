# A census says "not this time". A door says "never".

When you are unsure whether something can do harm, do not measure that it did
not. Put it in a state where it cannot.

## The rule

Given a doubt of the form *"could this touch the cards / the live tree / the
measured state?"*, there are two answers available, and they are not of the same
kind.

**The census** runs the thing and looks at what happened around it — process
lists, memory before and after, a log line. It answers for that run, on that
machine, in that moment. Every future run re-opens the question, and a census
that has passed a hundred times says nothing about the hundred and first.

**The door** makes the harmful state unreachable, then runs. If the thing
succeeds behind the door, you have proved something stronger than you asked
for: not that it did not, but that it **cannot**. If it fails, you have your
answer too, at once and at no cost, and the thing waits.

A door is not more cautious than a census. It is a different claim — universal
rather than particular — and it is usually cheaper to install.

## The instance that produced this page

2026-09-10. A kernel-IR harness compiles Triton kernels for an explicit target.
Compiling needs no device of that kind (the same mechanism had reported shared
memory for sm_86 on a rig of V100s), but whether the stack underneath opens a
CUDA context anyway was not known — and a certified-directory campaign was in
flight on all four cards, where a context would have cost real memory on a timed
measurement.

The proposal was a census: print `nvidia-smi --query-compute-apps` before and
after and compare. The owner refused it and named the door:

```
CUDA_VISIBLE_DEVICES= python tools/kernel_boolean_ir_equality.py
```

With no device visible, no context can be created on a real card and no byte of
its memory can be taken, whatever the stack decides to do. The harness now
**refuses to start** unless `CUDA_VISIBLE_DEVICES` is set and empty.

It compiled fourteen kernels behind that door. The question is closed for every
future run, not for that one.

## The corollary: a door does not only make safe, it makes RUNNABLE NOW

This is the half that was missing when the page was first written, and it is the
half that pays.

A census forces you to wait. It can only be taken while the thing runs, so if the
machine is busy — a campaign in flight, a locked-clock bench, a customer's
cluster — the honest thing to do with a census is to postpone. The instrument
then ships **unexecuted**, and an unexecuted instrument proves nothing, however
well written it is.

This repository has the living example. The `budget-unified` gate exists,
reads well, and its own report says of itself, under *"What the gate owes"*,
that it is **armed and has never run**. The numbers attributed to it — 42/42,
+5 %/+18 % — were never measured. A gate that has not run is a claim. It is entered in
`docs/reference/vacuous-gates-register.md` for the same reason every other
member of that class is: its failure mode is a green.

The IR harness was one afternoon from the same fate. Behind the census it would
have waited for the campaign to close, would have been reported as "written, not
launched", and would have been believed on its prose. Behind the door it ran
immediately — and running it is what found its two defects:

* it compared the raw IR, whose debug records carry the temporary file path, so
  every kernel read DIFFERS and, worse, **its own control cell was green for
  that same wrong reason**;
* and once that was fixed, it found a real one: `cond != 0` promotes the literal
  to i32 and buys a sign-extension on every element, on the kernel 39 of 56
  containers reach.

Neither would have been found by reading. Both were found in the first minute of
running, and the door is the only reason there was a first minute.

So the rule has two halves, and the second is not a bonus:

> **A door removes a risk AND removes a wait.** An instrument that cannot be run
> until some other work finishes is an instrument that will be reported before
> it is exercised. Ask of every "I will run this later": is there a state in
> which it could run *now* without being able to do harm?

## Doors this repository already has, recognised late as the same thing

Each of these was written as a local fix. They are one rule:

| doubt | the door |
|---|---|
| is the campaign measuring the live tree under an editable install? | it refuses to start without `--src`, or with a `--src` inside the live repository |
| is this worktree missing the ignored pointers that change Prism's decisions? | it refuses to start without `.nbx_registry` and `forge` |
| will this cell burn hours on a verdict already known impossible? | a cell whose known cost exceeds the budget is refused at the door |
| did a killed job fire the next waiter? | the DONE marker is written only on success, and carries a version so it cannot pre-exist |
| is a kernel reading past its tensor? | an out-of-range gather index traps, rather than being audited afterwards |

The pattern each time: a check that used to run *after*, moved to *before*, and
turned from a report into a refusal.

## How to write one

1. **Name the harmful state**, not the harmful outcome. "A CUDA context on a
   real card", not "slowing the campaign down".
2. **Find the cheapest way to make that state unreachable** from where the code
   runs — an environment variable, a missing pointer, an absent capability, a
   read-only mount. Prefer something the operating system or the driver
   enforces over something your own code checks.
3. **Refuse at entry**, before the first byte of work, with a message that says
   what is missing and gives the exact command that satisfies it.
4. **Leave a deliberate opening** with a name that reads as one — `--allow-…`,
   never a silent bypass — so the door can be opened knowingly on a machine
   where the doubt does not apply.
5. **Keep the census.** It costs nothing and it confirms the door. It just is
   not the proof any more.

## Where a door is the wrong tool

A door removes a capability. When the capability is the thing under test, a door
tests nothing: you cannot prove a kernel refuses an out-of-range index by making
indices impossible. There, the instrument is an injected fault and a gate seen
failing on it — the other half of this repository's discipline, and the reason
`cell 6` of the fault-channel proof runs the fault rather than forbidding it.

Doors for what must not happen. Injections for what must.
