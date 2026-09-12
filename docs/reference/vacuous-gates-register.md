# Register of vacuous gates — instrumentation that lies by construction

> **A helper whose every test passes can still have no seam.**
>
> Entry 17, and the whole class fits in it. `configs_agreeing_with_oracle`
> shipped one day with six green tests — correct, tested, and unreachable. The
> first call from the real data raised `TypeError`. Nothing in the suite could
> have said so, because nothing in the suite was the caller.

**A class whose failure mode is silence is invisible to the suite that contains
it. So this register is not bookkeeping — it is the only instrument the class
admits.**

One instance per entry. A count, not an impression.

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

### 17 — an overrule with every test green and no seam

* **date** 2026-09-10 · **machine** Dell · **site** `src/neurobrix/kernels/launcher.py`, `configs_agreeing_with_oracle` shipped in `137d78c`
* **what it could not say** that the consensus screen had seated a wrong kernel — which is the entire reason it was written, and the subject of `docs/reference/what-certified-means.md`. The predicate takes ONE buffer; a screened result is a SNAPSHOT, a list of buffers with a dtype each. Nothing carried a snapshot to it, so it could not be called from its only caller. Six tests, all passing, all against the primitive.
* **how it surfaced** trying to wire it: the first call from the screen's data raised `TypeError: a bytes-like object is required, not 'list'`.
* **closed by** the snapshot adapter `_oracle_keeps` plus the provider hook `set_screen_oracle`, with a test that drives `screen_configs` itself rather than the predicate, and pins both failure modes the vote cannot see — a majority wrong in the same way, and a unanimous wrong space — plus the control that with no provider the default path is bit-for-bit the behaviour shipped since 2026-09-07.
* **the distinction worth keeping** the predicate was not defective. It was correct, tested, and unreachable. **A helper whose every test passes can still have no seam**, and the class this register catalogues includes the instrument that exists but is not connected — not only the one that speaks falsely.

### 18 — a door that refused a correct invocation

* **date** 2026-09-11 · **machine** Dell · **site** `tools/precision_zoo_campaign.py`, `frozen_src_refusal` called unconditionally
* **what it could not say** that `--trees label=path,label=path` names a frozen worktree PER ARM. The door knew exactly one way of naming a frozen tree and turned away the invocation that satisfies its own rule by construction.
* **why it belongs here** it is the mirror of the class rather than an exception to it: an instrument that cannot recognise the state it is checking for. The damage is not a wrong green, it is that the next person adds `--src` beside `--trees` to get past it — and a rule survives as a formality while the habit it was built for dies.
* **closed by** `d5b64d6` — `frozen_trees_refusal`, same three checks per arm, naming the ARM in its refusal · **injection** an arm pointed into the live repo, and an unlabelled entry.

### 19 — a gate that printed a table with no rows and was called finished

* **date** 2026-09-11 · **machine** Dell · **site** the byte matrix of `01_budget_unified_gate.sh`
* **what it could not say** anything: invoked with no `--models`, the campaign filtered on `--family`, whose default is `None`, and selected zero models. It printed column headers, zero rows, and the script echoed `gate termine`.
* **how it surfaced** reading the output for a number and finding a header.
* **closed by** re-running over a NAMED five-family sample, with the claim it can support written beside it — `7de1560` touches one source file, so a model whose plan does not change is inert by construction.
* **the sharp end** this is the gate that was supposed to produce the `42/42` a delivered report had claimed without measuring. It would have reproduced the same empty claim with a different provenance.

### 20 — `_rig_busy` measured a moment, not the rig

* **date** 2026-09-11 · **machine** Dell · **site** `tools/certify_the_catalogue.py`, `_rig_busy`
* **what it could not say** that a campaign was in flight. A campaign BETWEEN two of its runs holds no compute process; `nvidia-smi` returned an empty list and the check said "free" while a gate held GPU0 five seconds either side.
* **how it surfaced** the tool answered 0 and the very next command showed a process on the card.
* **closed by** `08fd48c` — free means no compute process AND no driver alive that is about to start one; a `--plan` invocation does not count itself.
* **the precedent it repeated** 2026-09-10, a second instance launched onto a live measurement declared dead from a buffered log. The rule was "check the PID, never a log's size". This check obeyed the letter and missed the gap between two runs.

---

## The shape all three of 18-20 share

They are in the INSTRUMENTS, never in the engine — a door, a gate's scope, a
readiness check. The other machine reports the same distribution on its side.

The reason is structural and worth stating once: **an instrument has no suite
behind it.** A kernel has tests; a test has a gate; a gate has nothing. So
nothing forces an instrument to distinguish *"I looked and found nothing"* from
*"there was nothing to look at"*, and both come out as the same silence.

That is where the next one is, on both machines. Ask of every instrument: what
does it print when it was given nothing to measure? If the answer is the same
thing it prints on success, it is already an entry here and nobody has noticed.

### 21 — a readiness check that refused its own run

* **date** 2026-09-11 · **machine** Dell · **site** `tools/certify_the_catalogue.py`, `_rig_busy`
* **what it could not say** that the driver it had found was ITSELF. Entry 20 taught it to see a driver between two runs; it then counted its own launcher, which carries the same script name on its command line. The MEET phase refused to start three times before the pass ran.
* **how it surfaced** calling `_rig_busy()` directly returned 0 while the same tool, launched, refused with "1 compute process(es)".
* **closed by** exclusion of the whole **session** (`os.getsid(0)`) rather than the pid (defeated by the shell wrapper) or the process group (defeated by a launcher that puts the wrapper in another group). The session is the widest thing still unambiguously "this run". The control test pins the other direction: a SECOND instance in another session still counts.
* **and the refusal now says what it saw** — the driver's command line, up to three of them, with the line *"a refusal that does not say what it saw cannot be acted on"*. That is entry 18's lesson applied to entry 20's fix: the first two attempts printed a number and no subject, and the number was wrong.

### 22 — the pass died on the forty-fourth model and lost the forty-three

* **date** 2026-09-11 · **machine** Dell · **site** `tools/certify_the_catalogue.py`, the MEET loop
* **what it could not say** anything about the models it had already met: an exception composing one model's request ended the whole pass, and `meet.json` was never written.
* **the cause underneath** the family came from the HUB LISTING, whose category column reads `CODE`, and the engine has no `code` family. A shelf label was used where the container's own declaration was the authority.
* **closed by** reading `family` from the container's `manifest.json`, and by making a request that cannot be composed a **named skip** rather than a crash. Forty-three models must not be lost because the forty-fourth has no stimulus.
* **why it is in this register** the run exited 1 and printed a traceback, so it was not silent — but the ARTEFACT was: no record, no partial result, nothing to read afterwards. An instrument that produces nothing when interrupted has the same failure mode as one that produces a green: there is nothing to disagree with.

### 23 — the clock protocol had no check, and the natural one would have been green forever

* **date** 2026-09-12 · **machine** Dell · **site** the measurement protocol itself; closed in `tools/rig_clock.py` + `tools/rig_protocol.json`
* **what it could not say** that the rig was not at the protocol clock. Application clocks do not survive a reboot, this rack lost mains twice in nine minutes on 2026-09-11, and it came back with cards 0 and 1 at **1312 MHz** and cards 2 and 3 at **1290 MHz** — each pair at its OWN factory default. The protocol value is 1290. No harness read the clocks at entry; the value was typed by hand on a `--lock-clock` flag, or forgotten.
* **why it belongs here rather than in a bug list** the trap is in the SHAPE a check would naturally take. 1290 is the factory default of this rack's 32 GB cards, so a check sampling card 2 or card 3 reads the protocol value on a rig that is half wrong — and would have gone green on every run since the machine was built, having never once fired. The two SKUs advertise byte-identical supported-clock lists, so no capability query reveals the disagreement either. Only reading every card does.
* **how it surfaced** not by a test. The owner read `nvidia-smi` across all four cards while reviewing a post-outage state report that had checked git integrity, remotes, worktrees, GPU occupancy and disk — and not the clocks.
* **closed by** a door rather than a census: `require_protocol_clock()` reads **every** card the driver reports, refuses unless all agree with the authority in `rig_protocol.json`, names each diverging card with its own value alongside the conforming ones, and prints the exact restore command. Reading zero cards is itself a refusal. There is no built-in fallback value — a missing authority refuses, because a harness that invents the number stops citing the protocol. One deliberate opening, `--allow-off-protocol-clock`, which says so in the run's own output. Wired into `precision_zoo_campaign.py` and `certify_the_catalogue.py` beside the frozen-tree refusal, at the same moment and in the same class.
* **seen failing** twice, on a real card: card 1 set back to its 1312 default, the brick refused and exited 1 naming it; then the campaign refused end-to-end for a job pinned to card **2** — a card that was itself at protocol, which is the point. The unit suite injects the exact 09-11 reading, and was itself watched turning red under a deliberate one-card sampler planted in the brick.
* **what is NOT established, and cannot be** whether the 971 shapes certified on 2026-09-11 were measured on protocol. The certification benches on the default CUDA device — card 0, one of the two that sit at 1312 by factory default — and its proof recorded hostname, platform and hardware profile but **not the clock**, so the conditions are unrecoverable from the record. The choices themselves are internally consistent (every candidate for one shape is timed on one card at one frequency, so the ranking holds); it is the `best_ms` values whose regime is unknown. Closed forward only: `autotune_certify._machine()` now writes `clocks_mhz` into every proof, read once per run and memoised — `_machine()` is called once per certified key, and a driver query per key is the anti-pattern this project has already paid for.

---

### 24 — the door was placed on the harnesses beside the command, not on the command

* **date** 2026-09-12 · **machine** Dell · **site** `neurobrix autotune certify` — `src/neurobrix/kernels/autotune_certify.py:certify()`
* **what it could not say** that the rig was off the protocol clock. Entry 23 closed that the day before — and closed it on `precision_zoo_campaign.py` and `certify_the_catalogue.py`, the two `tools/` harnesses that were in view at the time. The engine's OWN certification, which the autotune doctrine names as the only way to fill the certified directory, kept no entry condition at all. It had just been taught to RECORD the clocks into every proof, which is what made the gap easy to miss: the field was there, so the subject looked handled.
* **why it would have stayed green forever** nothing about the command would ever have complained. It writes a directory of timings, each proof carrying a clock reading that no code compares to anything, and the run reports `N shape(s) certified` whatever the frequency was. The 09-11 entries are exactly that artefact: internally consistent, and of an unrecoverable regime.
* **the shape, which is the general lesson** a defect gets closed at the sites the closer was LOOKING at. Two harnesses were open on the screen; the shipped command was one import away and stayed open. Recording a condition is not checking it, and a door on the paths beside an entry point is not a door on the entry point. **When closing a class, enumerate its call sites from the code rather than from the session** — the grep that found this took one line and ran after the fix had already been committed and pushed.
* **how it surfaced** not by a test, and not by the tests written for entry 23, which all passed. By grepping for the door's call sites before *using* the command — the certification of the remaining shapes was the next task in the queue, and it would have run through an open door.
* **closed by** the refusal moved INTO `certify()` rather than into the CLI wrapper, so it holds for every caller and not only for the one that types the documented command, and placed before the profile is even resolved, so it cannot spend what it exists to protect. The protocol is DISCOVERED, never shipped — an env pointer, then the workshop's `tools/rig_protocol.json` in a source checkout, then the machine's dotfile — because a protocol is a property of a rack and an installed NeuroBrix carries no rack's decision. There is no built-in value: a named authority that does not read is a refusal, since an engine that invents the number stops citing the protocol. A machine declaring no protocol is not refused but is TOLD so in the run's own output, a silence there being indistinguishable from a door that held.
* **and the asymmetry it exposed** the engine's clock reader returned only the graphics clock while the workshop's door checks graphics AND memory, which `nvidia-smi -ac` sets together. Two doors for one protocol that disagree are a false green waiting for its occasion, so the reader was widened to both and a test now runs BOTH doors against the same injected readings and fails if their verdicts differ. They stay separate implementations deliberately — a campaign measures a frozen `--src`, so the workshop's door must not depend on whichever engine tree is under measurement — and they read the same authority file, so only the logic could drift.
* **seen failing** on a real card and in the suite. Card 1 set back to its 1312 default: the documented command refused, exit 1, naming card 1 with its value beside the three at protocol, before anything was timed and with nothing written; then the same invocation with `--allow-off-protocol-clock` proceeded and said so. In the suite, a one-card sampler planted in the brick turned 5 of 12 red, and a graphics-only comparison turned 2 red — including the cross-door agreement test, which named the disagreement it exists for (`engine passed, workshop refused`).

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

24 entries, of which five are placeholders and 19 carry a site. Two
machines, two weeks of concentrated looking. Every one of them produced silence
or a green rather than an error, and **not one was found by a test** — they were
found by users, by contradictions between two numbers, by reading generated
code, and twice by another instrument built for something else.

That is the argument for the register: a class this large, whose members are
invisible to the suite by definition, is only tractable if its instances are
written down where the next person will look.
