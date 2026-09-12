# Bringing `metal-first-light` home: what needs a decision

**Prepared 2026-09-12 on the Dell. Nothing is merged. Nothing may be merged from
this file** — the other machine is mid-chantier, and this list exists so that when
the merge happens it is an arbitration and not a discovery.

**State**: `origin/metal-first-light` is **105 commits ahead** of `main` and **20
behind**, from a merge base at `d5b64d6` (2026-09-11). It is the largest structural
debt in the project: the trunk — the thing a user installs, and the thing this
machine certifies — carries none of it.

---

## The first surprise: there are no textual conflicts

A trial merge (`git merge-tree`, which writes nothing) reports **four files
changed on both sides and ZERO conflict markers**. Git will merge all of it
silently.

| file | trunk since the base | Mac since the base |
|---|---|---|
| `src/neurobrix/core/runtime/executor.py` | +27 −1 | +58 −11 |
| `src/neurobrix/kernels/autotune_certify.py` | +227 −4 | +98 −5 |
| `tests/unit/kernels/test_autotune_certified_directory.py` | +48 −0 | +7 −1 |
| `tools/precision_zoo_campaign.py` | +146 −14 | +17 −1 |

**That is the reason to read this list rather than trust the merge.** A clean
merge is a statement about text, and every item below is a statement about
meaning. Two mechanisms that do the same job in two different files never
conflict; they both arrive.

---

## Item 1 — the two oracles are NOT rivals. They are two halves of one thing.

This was the item feared most, and the reading dissolves it.

* **Trunk** (`kernels/launcher.py`) holds the **primitive**
  `configs_agreeing_with_oracle(results, oracle, dtype_name)`, the **adapter**
  `_oracle_keeps(...)`, and the **seam** `set_screen_oracle(provider)`. Its own
  comment names what it lacked: *"a helper whose every test passes can still have
  no seam."*
* **Mac** (`kernels/screen_oracle.py`, 189 lines) holds the **provider**:
  `provider(tuner, key, buffers)` recomputing the kernel's mathematics in float64
  from the live operands, `announce_no_oracle(...)` for the kernels it cannot
  cover, and `install()` — whose body is `L.set_screen_oracle(provider)`.

**They compose exactly.** The Mac wrote the thing the trunk's seam was cut for.
No arbitration is needed on the mechanism.

**ANSWERED 2026-09-12, and it moves the decision** — `docs/reference/owed-proofs.md`
item 3. The table names THREE kernels, not two (`matmul_kernel`, `addmm_kernel`,
`baddbmm_kernel`), so the Mac's own `addmm` blind-spot case is one the provider
covers. Counted against this rack's 7 158 certified keys: **6 336 covered (88.5%),
822 not (11.5%), and the uncovered set is exactly the convolution family** —
`conv2d_forward_kernel` 770 and `depthwise_conv2d_kernel` 52.

The answer is therefore not "most kernels" but one family, which makes
`announce_no_oracle` a smaller decision than it looked: make it REFUSE to seat a
configuration rather than fall through to the bare consensus screen, since the
bare screen is the instrument the Mac measured seating a wrong configuration
unanimously. A float64 direct convolution then moves those 822 keys from a vote to
a measurement.

**Why it matters beyond tidiness**: the Mac's own measurement says the consensus
screen has a blind spot it cannot see — on four `addmm` shapes, the emitted MSL
declared `alpha`/`beta` as `int`, so every candidate was wrong in the same way,
the vote was unanimous, and the bare screen **seated a wrong configuration every
time** while the same screen with the oracle refused every time. This machine
certifies against an fp64 oracle already; the screen does not, yet.

## Item 2 — the NeuroBrix launcher, and it is R33's structural piece

The Mac's `launcher.py` also adds a backend-launcher interface — `launch()`,
`block_for()`, `target()` — which is the replacement for Triton's `kernel[grid]`
that R33 requires (*"Triton's launcher, which imports torch in its binder and its
driver on every backend, is replaced by a vendor-agnostic NeuroBrix launcher"*).
The trunk has none of it.

**This is not a merge question, it is a sequencing one.** The Dell measured a
throughput regression from exactly this peel once already (the decode replay
watched a call site the engine had replaced, `project_replay_regression_2026_09_08`).
Landing the launcher on the trunk means re-running that measurement, not
re-reading the argument.

## Item 3 — four files both sides grew, and what each grew them for

None conflicts textually; each needs one read.

* **`autotune_certify.py`** — the trunk added the clock door
  (`rig_protocol_refusal`, `4043d39`) and the unreachable-key classification
  (`392b663`); the Mac added +98 lines of its own. **Read for: does the Mac's
  work also touch `certify()`'s entry or its failure accounting?** Two entry
  conditions written independently will both be present and neither will know
  about the other.
* **`precision_zoo_campaign.py`** — the trunk rewrote `cell_cost_estimate`
  (`arms`, `narrowest`, ended-vs-killed arms) and added `flightrec_refusal`
  (`f29ff39`, `392b663`). **Read for: whether the Mac's +17 touch the same
  estimate**, because a cost function that two machines changed for two reasons
  is the kind of thing that merges cleanly and then reports a number neither
  machine intended.
* **`executor.py`** — the trunk added the VAE-scale declaration read (`f0e573b`,
  which changes no model's behaviour today). The Mac's +58 are unread here.
* **`test_autotune_certified_directory.py`** — both added tests. Additive by
  nature; verify both suites run green together rather than assuming it.

## Item 4 — what the trunk is missing, named so the merge has a checklist

From the owner's own account, and not yet verified line by line on this machine:

* the CUDA fault channel — `f769f2e`. **Its proof is returned**; see
  `docs/reference/owed-proofs.md`, one item outstanding (gate the buffer on a
  non-zero code).
* the Prism fix that plans against what the machine actually has;
* exact bf16 synthesis;
* the oracle wired into the live screen (item 1).

---

## How this should be done, and by whom

**Both agents and an owner's arbitration**, on a quiet rig, with the other
machine's chantier finished. Not today.

The order that follows from the above: item 1 first (it is composition, not
conflict, and it is the cheapest real gain), then item 3's four reads, then item 2
last — because the launcher is the one that needs a measurement re-run and not a
reading.
