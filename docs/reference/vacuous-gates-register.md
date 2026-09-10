# Register of vacuous gates — instrumentation that lies by construction

**One instance per entry. A count, not an impression.**

This class is the most productive shared finding of this project, and until now
it had no home: two machines were counting it independently, each reaching "the
ninth instance" within an hour of the other on 2026-09-10. A class that everyone
cites and nobody records is a class whose cost is a feeling.

## What belongs here

An instrument — a gate, a recorder, a metric, a guard, a clamp — that **cannot
report the thing it was built to report**, and whose failure mode is therefore
SILENCE or a GREEN. Silence is indistinguishable from correctness, which is why
this class is expensive out of all proportion to the size of each instance.

It belongs here whether the cause is a swallowed exception, a list nothing
flushes, a key the producer never writes, a path never taken, a scan that scans
nothing, a table that exists for no target, a comparison that compares the wrong
thing, or a claim in prose that nothing checks.

It does NOT belong here if the instrument reported correctly and the code was
wrong. That is an ordinary defect. The mark of this class is that **the
instrument was never in a position to speak**.

Doctrine: `docs/reference/what-a-green-test-proves.md` (the three spaces a test
can be empty in) and `docs/reference/proving-by-doors.md` (why an unexecuted
instrument is a member of the same family).

## How to add an entry

Append. Never renumber — a number that moves cannot be cited. Fill every field
you can evidence and write `?` in the ones you cannot, rather than a plausible
guess: a register that contains one invented line is a register nobody can
quote. If you find the missing field later, fill it in place.

| field | meaning |
|---|---|
| date | when it was FOUND, not when it was introduced |
| site | `file:line` where the lie lived, or the commit if the line has moved |
| machine | which rig found it — this class is found by whoever runs, and that matters |
| what it could not say | one sentence, in the negative: what the instrument was unable to report |
| how it surfaced | almost never a test. Usually a user, a contradiction, or another instrument |
| closed by | the commit that fixed it, and the injection that was seen turning it red |

---

## Instances

### 1–5 — the founding five (Dell, dates before 2026-09-09)

Recorded as a group in the memory entry *"instrumentation that lies by
construction is one family"*, which counted **five vacuous gates and 235 blind
sites** before this register existed. Their individual sites were not written
down at the time.

**Deliberately left as placeholders rather than reconstructed from memory.**
Known shapes among them: a recorder swallowing an exception; a list nothing
flushes; a proxy computing nothing; a metric reading an absent key. Anyone who
can attach a date, a site and a commit to one of these should split it into its
own numbered entry below and strike it from this group.

| # | date | site | machine | what it could not say |
|---|---|---|---|---|
| 1 | ? | ? | Dell | ? |
| 2 | ? | ? | Dell | ? |
| 3 | ? | ? | Dell | ? |
| 4 | ? | ? | Dell | ? |
| 5 | ? | ? | Dell | ? |

### 6 — the image-fidelity gate that could never say AGREES

* **date** 2026-09-09 · **machine** Dell · **site** the vendor-correctness cell's metric read `psnr` while the brick emitted `psnr_db`
* **what it could not say** that an image AGREED with the vendor — no image cell could ever return anything but DIVERGES, so four harness defects hid behind a verdict that looked like a finding.
* **how it surfaced** a cell that had never once passed was fed a known-good input.
* **closed by** ? (the fix landed in the vendor-cell work of 09-09) · **rule it produced** *a gate that never passed was never tested*: feed it something you know is right and see PASS before trusting any failure.

### 7 — the per-op recorder blinded by a replaced call site

* **date** 2026-09-09 · **machine** Dell · **site** the per-op recorder's single-output guard; audit tool `tools/observability_gap_audit.py`
* **what it could not say** what 2 of 3 Wan VAE upsamples produced — a fusion proxy is not a Tensor, so the recorder skipped them on the guard meant for tuple-returning ops.
* **how it surfaced** a cross-engine walk showed a gap where siblings of the same op type had recorded.
* **closed by** `270da70` *a brick that replaces a call site reports what it replaced* and `e56b397` *the gate becomes decidable — the runtime's planned ops*.

### 8 — `--skip-done` read a crash as a verdict

* **date** 2026-09-10 · **machine** Dell · **site** `tools/precision_zoo_campaign.py`, the done test read only that `result.json` exists
* **what it could not say** that a model had never actually been measured. Three models sat frozen out of every campaign on records carrying `gate.ran: false`, and each hid a distinct further defect.
* **how it surfaced** a campaign that kept skipping models whose numbers nobody had ever seen.
* **closed by** `098ce03` · **injection** a record with `gate.ran: false` is no longer "done".

### 9 — the flight recorder that did not write the flight

* **date** 2026-09-10 · **machine** Dell · **site** `tools/flightrec.py`, `--log`
* **what it could not say** anything at all: the flag recorded a path into the record and nothing ever wrote to it. Under a rack with no UPS, this is the instrument whose whole purpose is to survive a cut.
* **how it surfaced** a resumed job whose log was empty.
* **closed by** `3fa2984` — the child's output is now teed to the path.

### 10 — the flash-attention clamp that clamped nothing off Volta

* **date** 2026-09-10 · **machine** Dell, from a user's A40 report · **site** `src/neurobrix/kernels/wrappers.py`, the `sdpa_thresholds` read (the reason is now written at `wrappers.py:8508`)
* **what it could not say** that a tile did not fit. `sdpa_thresholds` exists only in the seven profiles this repository ships, so for **any** card whose profile is not shipped the clamp returned `(None, None)` and pruned nothing. Every language model refused to start on an sm_86.
* **how it surfaced** a user, not a test. No card here could reproduce it.
* **closed by** `f9893a6` (limit from the driver) and `372e1f5` (cost from the compiler) · **the number that settled the design** the same tile needs 98 304 bytes on sm_70 and 164 352 on sm_86 — the cost is a function of the target, so an analytic estimate cannot exist.

### 11 — the paired campaign whose control arm measured an empty sweep

* **date** 2026-09-10 · **machine** Dell · **site** `tools/precision_zoo_campaign.py`, one replay directory shared across N repetitions
* **what it could not say** that an arm had swept nothing. Repetition 1 swept and wrote; 2 and 3 read their own leftovers; the median landed on a repetition that did no work and reported `speedup 1.0288` on a cell whose first repetition read 67.01 against 95.94. **15 of 15 cells would have been measured empty**, whatever the disk held.
* **how it surfaced** the owner reading `B.swept = 0` beside a speedup of 1.0287.
* **closed by** `0614946` — replay isolated per repetition, plus a guard that gives NO ratio (rather than a ratio of one) to a lever arm that served nothing and swept nothing.

### 12 — the test suite a fresh clone could not collect

* **date** 2026-09-10 · **machine** Dell · **site** `tests/__init__.py`, untracked
* **what it could not say** anything: a clone of the repository collected zero tests, so every gate in it was vacuous by construction for anyone who was not this working tree.
* **how it surfaced** a clone into an empty directory, which is the only method that proves it.
* **closed by** `910c6d2` · **proof** the clone now collects 1,529 tests, identical to the live tree.

### 13 — the device assert the backend computed and discarded

* **date** 2026-09-10 · **machine** **Mac** · **site** the three gather/scatter kernels armed on 2026-09-02; `tl.device_assert` lowered into the IR under `debug=True` and elided at the Metal dispatch point
* **what it could not say** that an index was out of range. Measured: the generated MSL carried `mask_5 = idx < 2` and never read it — the guard paid for in registers and not delivered. `index_select` wrote nothing and left the pool's residue; `embedding` read 4 floats past the weight; `index_put` **wrote 8 floats past the tensor**.
* **how it surfaced** reading the generated code, on a machine that cannot execute the path.
* **closed by** `f769f2e` (a capability-armed fault channel) · **CUDA proof owed and prepared**, `campaigns/prepared/03_cuda_fault_channel_proof.sh`, whose deciding cell runs FIRST: inert must never mean disarmed.

### 14 — the out-of-range test that passed for the wrong reason

* **date** 2026-09-10 · **machine** **Mac** · **site** the permanent gather/scatter test's out-of-range case
* **what it could not say** whether the KERNEL refused the index. Its numpy oracle `rows[idx]` raised on the out-of-range index before the kernel was ever questioned, so `returncode != 0` was satisfied by the oracle line.
* **how it surfaced** the author of `f769f2e` re-reading the case it was about to rely on.
* **closed by** `f769f2e` — the out-of-range case now takes no oracle at all.

### 15 — my own control cell, green for the wrong reason

* **date** 2026-09-10 · **machine** Dell · **site** `tools/kernel_boolean_ir_equality.py`, first version, comparing raw TTIR
* **what it could not say** whether two kernels compile to the same instructions. The two variants are written to different temporary files, so every `#loc` record carried a different path: all fourteen kernels reported DIFFERS, **and the control cell announced "the gate bites" for that same wrong reason** — it would have said so even if the precedence trap compiled identically.
* **how it surfaced** running it. The door (`CUDA_VISIBLE_DEVICES=`) is the only reason it could be run at all before the campaign closed.
* **closed by** `ad1916d` — comparison of the `tt.*` body with every location stripped; columns cannot be kept either, since the two forms put their operators at different columns.
* **worth recording plainly** this was a vacuous gate written inside the instrument built to catch vacuous gates, by the person who had spent the day cataloguing them.

### 16 — a comment that asserted a cost nobody had measured

* **date** 2026-09-10 · **machine** Dell · **site** `src/neurobrix/kernels/ops/where.py`, the comment *"costs nothing"*
* **what it could not say** — nothing, because it was not an instrument at all. That is the point: a claim of cost, neutrality or equivalence written in prose is an assertion with no gate behind it. `cond != 0` promotes the literal to i32 and buys an `arith.extsi` i8→i32 on every element, on the kernel **39 of the 56 installed containers** reach.
* **how it surfaced** an IR harness written for a different purpose.
* **closed by** `ad1916d` — `tl.where(cond.to(tl.int1), …)` emits exactly the cast the compiler already performed, TTIR identical to the pre-edit kernel, and the form is pinned by a test that says why two forms silence the warning and only one is free.

---

## The Mac's entries

Entries 13 and 14 are the Mac's, transcribed from `f769f2e` because they are
part of the count and the count had to be one list. **The Mac completes its own**
— any instance found there that is not above should be appended here with the
same fields, and any `?` in 13 or 14 corrected in place by the machine that has
the evidence.

The rule for both machines is the same: append, never renumber, and write `?`
rather than a plausible reconstruction.

## What the count is worth

Sixteen entries, of which five are placeholders and eleven carry a site. Two
machines, two weeks of concentrated looking. Every one of them produced silence
or a green rather than an error, and **not one was found by a test** — they were
found by users, by contradictions between two numbers, by reading generated
code, and twice by another instrument built for something else.

That is the argument for the register: a class this large, whose members are
invisible to the suite by definition, is only tractable if its instances are
written down where the next person will look.
