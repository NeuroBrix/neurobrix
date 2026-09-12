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

## Writing and repointing are two acts, and the check goes between them

A door refuses at entry. There is a second shape that saved this project on
2026-09-12 and that had never been named: **when an act replaces something that
works, write the replacement somewhere else, verify it, and only then repoint.**

An incomplete container — a video model built without its backbone because the
snapshot's weights had been purged — was being uploaded onto a working hub slug.
The hub was not harmed, and not because anyone checked in time. `replace` writes
to a distinct storage key, verifies the checksum, and repoints the record only
after; a mismatch rolls the upload back and leaves the previous artifact serving.
The upload was stopped at 0% of 6.73 GB and there was nothing to undo.

The same command's LOCAL half had the opposite shape. `forge local --overwrite`
removed the installed container and then extracted into the same path, so the
complete local container was gone before anything could object — and an
extraction that stops midway (a truncated archive, an NFS stall, a mains cut on a
rack with no UPS) leaves a partial installation in the canonical path. A partial
installation looks installed.

**What protected us was not the care of whoever ran the command. It was that the
write and the repoint were two acts with a check between them.** That is a
property of the design, available on every run, to everyone, including the person
who is tired. Care is not.

How to recognise the missing form: look for `rmtree`, truncate, or an in-place
overwrite of a path that something else reads by name. The three questions are —
what is readable at that path while the write is in progress; what is readable if
the write stops halfway; and what compares the new thing against the OLD thing's
declaration rather than against itself. A tree compared with itself agrees with
itself.

Shipped as `forge.verify_extraction` plus a staging tree and an atomic rename.
Seen both ways before it was trusted: a clean install repointed and left no
staging tree, and an archive whose manifest declared a component the archive did
not contain was refused with the live installation unchanged — same mtime, same
components.


## A wait condition that lists process names expires the day you write a tool

A timed bench must run on a quiet host — this project has measured what happens
otherwise, and the bench's own script says so. On 2026-09-12 one started anyway,
in the middle of an NFS export that had not completed a 100 MB read in fifteen
minutes, and began timing five paired couples whose every weight comes from that
export.

Its wait condition was three process names. The queue that had replaced those
jobs was a new script with a new name, so the condition looked at a machine with
none of its three and concluded quiet.

**Enumerating what must not be running is a guess about the future.** It is right
until someone writes a tool, and it fails SILENTLY and in the direction that
costs: it starts work rather than blocking it, and the work it starts produces
numbers that look like every other number.

Two forms that do not expire:

* **Wait for the thing you actually depend on to say it is done** — the queue's
  own end marker, not the absence of its parts.
* **Measure the condition instead of inferring it.** `tools/export_quiet.py`
  reads bytes off the export and refuses under a floor. Not `df`, which answers
  from cached metadata while bulk I/O is dead, and not the load average, which
  is a decaying mean that stayed above 30 for ten minutes after every cause had
  been killed. Both of those were consulted on the day and both said the machine
  was fine.

The general form: **a precondition stated as a list is a precondition that only
its author can maintain.** State it as a measurement of the thing itself, or as a
signal the producer emits, and it survives the next tool.
