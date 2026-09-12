# What "certified" means, and what it does not

Read this before writing a sentence about kernel selection anywhere — a release
note, a README, a benchmark page, an answer to a user.

## The short form

**The certified directory is not a speed feature. It is the only place where a
proof exists.**

Everywhere else, the guarantee is a **consensus**.

## The long form, because the distinction is not decorative

Kernel settings reach a run by one of two roads.

**The certified road.** `src/neurobrix/config/autotune/<vendor>/<profile>/`
holds a setting per kernel, per dtype, per shape key, and every entry carries
its own proof: the date, the engine and backend versions, the shape, the
deviation measured **against the fp64 oracle**, the profile's tolerance, and the
machine that measured it — plus the settings that were excluded and by how much
they deviated. `neurobrix autotune check` refuses a file without a proof, or one
whose proof does not re-read. A shape served from there has been compared to
something true.

**The swept road.** A shape the directory does not hold is swept at runtime, and
what stands between the user and a wrong kernel is `screen_configs` — the
consensus screen. It runs every candidate once from the same starting state,
clusters them by agreement, and keeps the largest cluster.

That screen was the right answer to a real incident: on 2026-09-07, anchoring
the comparison on a nominated reference config inverted, because `matmul`'s
first config was one of three that wrote half their output, and screening
against it excluded the seven correct ones. There is no way to know in advance
which candidate is right — that is the whole problem — so the screen asks which
answer the candidates agree on.

**But agreement is a vote, and a vote consults no oracle.** It has two failure
modes it cannot see:

* the majority cluster is wrong **in the same way**, so the minority that is
  right is excluded and the wrong kernel is seated;
* every candidate agrees and all are wrong — the code path is
  `if len(clusters) == 1: return everything`, and it says nothing at all.

## Therefore, the sentence we are allowed to write

Outside `nvidia/volta`, **"screened" has never meant "proved". It has meant "the
candidates agreed with each other."**

Our certified directory covers `nvidia/volta` and nothing else. On every other
card the runtime sweeps, so every other card has been running on a consensus,
not on a proof — and the configuration space that consensus votes over **differs
by target**. That is measured, not supposed: the same flash-attention tile needs
98 304 bytes of shared memory on sm_70 and 164 352 on sm_86, because the cost is
a function of the target as much as of the tile. A different space means
different candidates and a majority nobody here has ever seen.

## A plausible mechanism, offered as such

A user on sm_86 reported greedy decode returning 98 tokens against 31 in
compiled mode, collapsing into repetition. At temperature zero there is no
sampling, so the divergence is in the forward pass and not in the draw.

A wrong kernel seated by a wrong majority, in a space we never explore, on a
card whose settings nobody has certified, is a **plausible mechanism** for that.

**It is not proved and must not be reported as proved.** It has not been
reproduced; that hardware is not here. What is established is the structural
gap — the screen consults no oracle — and that the gap is widest exactly where
that report comes from. Anyone writing about it says both halves or neither.

## The two numbers that prove it, measured on the same day

This was doctrine before it was evidence. On 2026-09-11 it became evidence, and
the two halves came from two machines at once.

**On `nvidia/volta`, the oracle refused nothing.** A certification pass ran every
candidate on every uncertified shape against the fp64 oracle: **535 shapes, 17
candidates each, 0 configurations excluded.** The consensus vote and the oracle
never once disagreed. On the one target this project has certified, the screen
was right every time.

**On Metal, the oracle refused four shapes out of four.** The same instrument,
the same tolerance, on `addmm` — every one of them wrong, and wrong by a factor
of about **one billion**, from a defect upstream of the kernel.

    nvidia/volta    0 of 535 shapes refused by the oracle
    metal           4 of 4 shapes refused by the oracle

Put side by side, those two lines are the whole argument of this page, and they
say exactly what it says: **the gap is widest where nobody has looked.** Volta is
the target with a certified directory, and there the vote had nothing to catch.
Metal is a target being brought up, and there the vote would have seated an
answer wrong by nine orders of magnitude — silently, because a vote consults no
oracle and unanimity passes everything.

**Nobody can argue the oracle is a luxury after this.** It cost 56 minutes on
Volta and found nothing; on Metal it was the only thing standing between a
bring-up and a number that is not merely imprecise but meaningless. An
instrument that finds nothing on a healthy target and everything on a sick one
is not overhead. It is the definition of a working instrument.

And it settles which of the two results is the surprising one. Zero exclusions
on Volta is not a non-event to be mentioned in passing — it is the control that
makes the Metal four legible. Without it, four refusals could be the oracle
being too strict; with it, they are four real defects.

## Consequences that follow immediately

* A benchmark on non-Volta hardware may not be described as running "validated"
  or "certified" kernels. It ran screened ones.
* "Certified for `<profile>`" is a statement about one directory and one
  machine, never about the engine.
* Extending the certified directory to a new target is not an optimisation
  task. It is the act that turns a consensus into a proof, and it is the only
  one that does.
* **The overrule is now wired** (2026-09-10). `screen_configs` asks an installed
  oracle provider (`set_screen_oracle`) before it clusters; where one answers,
  the vote is not consulted for that key, a candidate the oracle contradicts is
  never seated, and a space the oracle contradicts ENTIRELY raises rather than
  returning silently. When the vote was about to seat a config the oracle
  refuses, that is printed as a FINDING — a majority wrong in the same way is
  the evidence that a target needs looking at, and correcting it quietly would
  destroy the only trace of it.
* **No provider is installed by default**, so the shipped behaviour is unchanged
  and the hot path pays one `is None`. Producing an oracle costs an fp64
  reference per key, which costs a card; the certification runner is its first
  client. Until a provider is installed on a target, that target still runs on
  a consensus, and this page still describes it as it is.

---

## What the directory BUYS, and the law that predicts it (measured 2026-09-12)

The preceding sections are about correctness. This one is about cost, and it is
here because the two get confused: the directory's reason to exist is the proof,
but people ask what it saves, and the honest answer has a shape worth stating.

**The saving is not a function of the sweeping machinery. It is a function of how
many distinct kernel shapes a request meets per second of the time it runs.**

Measured over eleven paired cells — cold, three repetitions, one lever
(`NBX_AUTOTUNE_CERTIFIED` on/off), a frozen tree, every card at 1290/877 MHz.
Sorted by sweep-cost over base-time, the distribution is **bimodal with an empty
interval**:

| regime | cells | cost/base | sweep cost |
|---|---:|---|---|
| sweep-dominated | 5 | 4.32× – 14.38× | 127 s – 2 924 s |
| sweep-negligible | 6 | 0.01× – 1.30× | 26 s – 403 s |

**Nothing falls between 1.30× and 4.32×.** A median over the eleven reads 1.14×
and describes no model in the upper group, understating every one of them by an
order of magnitude — so this page reports the two regimes and their boundary, and
never a median. A distribution with a hole in it does not have a middle.

**Neither obvious explanation survives the data.** The key count has members on
both sides (37 keys at 4.32×, 33 keys at 0.73×). So does the base time (35 s at
1.30×, 3 468 s at 0.01×). Their **ratio** does not: distinct shapes met per second
of base run is 1.26–2.78 in the upper regime and 0.00–0.25 in the lower — the same
partition, with the same empty gap. And the **per-shape sweep cost is flat across
both regimes**, about 3–12 s a shape, which is what rules the machinery out: the
sweep costs the same everywhere; what differs is how many sweeps a request buys
per second it runs.

So the rule predicts before measuring:

> **A short run meeting many distinct shapes pays the sweep many times over. A
> long run meeting a few dozen pays it once and amortises it.**

Multimodal and MoE text models sit in the first regime; diffusion and video in the
second. A model's family is a weak proxy; its shape density is the thing.

**Scope, and it binds every number above.** These are measurements of ONE rack, at
ONE clock, cold, against a frozen tree. A machine whose replay cache is already
warm pays a different price, and a card whose shapes were never certified pays all
of it. The table with its full per-model rows and its caveats lives with the
campaign, not here.
