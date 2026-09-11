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
