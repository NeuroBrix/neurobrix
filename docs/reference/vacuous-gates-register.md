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

### 25 — the plan printed a cost, and nothing compared it to the clock

* **date** 2026-09-12 · **machine** Dell · **site** `tools/certify_the_catalogue.py` (the MEET runner) + `tools/precision_zoo_campaign.py:cell_cost_estimate`
* **what it could not say** that a run was doomed before it started. The planner reads each model's recorded cost, prints it, and decrements it from the campaign budget. The runner then kills every run at `--timeout`. **Nothing compared the two numbers.** On 2026-09-11 the plan accepted `Wan2.1-T2V-1.3B` at 6 962 s and the runner killed it at 2 700 s; `Allegro`, on the same clock, went the same way. Two kills, ninety minutes of rig, and not one shape collected between them.
* **why the green read as an answer** the plan's output is a table of costs that all fit the budget, which reads as "this campaign is affordable". It was — against the budget. Against the clock each run would actually be given, two of its rows were arithmetically impossible, and the table never mentioned the clock.
* **and the number itself was mis-founded, twice** it was the SUM of a paired A/B cell's two arms, while the MEET phase — in its own docstring — is "ONE run per model, not a paired A/B": every model with a record was charged about double. And the two arms it was summed from had both exited `rc=1`. The function correctly treats a *killed* arm (`rc < 0`) as a lower bound, but counted a *failed* arm as a measurement — so the figure that read as "what this model costs" was the cost of failing.
* **closed by** `cell_cost_estimate(..., arms=1)` prices one run rather than a cell; only arms that ENDED answer, a kill being a floor and not a cost; a non-zero arm is still counted but its basis now says in words that this is the cost of failing, not of finishing. And a second door, `timeout_refusal()`, refuses at entry any run whose own recorded cost exceeds the clock it is about to be given, naming both numbers and the `--timeout` that would let it run. The refusal strictly dominates the kill: today's behaviour spends the whole timeout to reach the same outcome and loses the timeout too — a killed run produces nothing, so nothing is lost by refusing. Nothing is guessed: a model with no record still runs, as the budget guard already promised.
* **the second defect, found inside the fix, and the one worth keeping** the first version read the WIDEST arm for both questions, justified in a comment that said *"a guard that under-states a known cost is the one that lets a doomed run start"*. That sentence was written before the measurement. Run against the real records it refused `Qwen3-VL-30B` — recorded arms 211 s and 3 135 s — on a 2 700 s clock, for a model that had **met in 365 s the previous day**. Reserving and doom-testing are two questions and take two statistics: the widest arm for *how much must I reserve*, the narrowest for *is this doomed*, because a model that has finished once under the clock is not doomed by it whatever a slower arm did. The table that settles it, MEET actual against the cells: Qwen3-VL 365 s (arms 211 / 3 135), DeepSeek-Coder-V2 140 s (93 / 1 430), Qwen3-Omni 257 s (98 / 1 294), CogVideoX-2b 614 s (548 / 951) — every MEET run lands near the NARROWEST arm and never near the widest, because by then the replay cache is warm.
* **how it surfaced** the first version was green on sixteen tests before it was ever pointed at a real record. It was the confrontation with the 2026-09-10 cells that produced the false refusal, in the same hour and in the same shape as the defect being fixed — **a plan accepted on a number nobody had confronted with the real thing.**
* **seen failing** three injections, each red on its own test: pricing `arms=1` as the cell sum again (2 red), silencing the failed-arm note (1 red), and a `timeout_refusal` that never refuses (1 red). The final form is pinned against the live 2026-09-11 records: one refusal (`Wan2.1-T2V-1.3B`, the 45 minutes that bought nothing) and zero false refusals across the eight models that met.

---

### 26 — an invariant that holds, and is wrong

* **date** 2026-09-12 · **machine** Dell · **site** `src/neurobrix/core/prism/profiler.py:_resolve_shape`, the TRUST GATE
* **the form, which is new to this register** the four gates above it are gates that check NOTHING. This one checks something, the check passes, and the thing it certifies is false. It is the more dangerous shape, for the same reason the clock field was: **a vacuous gate leaves no trace, and this one leaves a reassuring one.**
* **what it certifies** a tensor's symbolic dimension may be evaluated at runtime only if its own expression tree reproduces the concrete dim witnessed at trace. The gate was written against a real defect — image graphs carrying mis-associated product expressions, one Sana `_unsafe_view` dim whose expression traced at 8.4e14 against a concrete dim of 1 — and against corrupted integer annotations (98 component graphs of the zoo carrying another tensor's extent in an output slot; Kokoro's decoder conv put its input length, 15361, in the batch slot and was sized at 19.2 GB).
* **what it cannot see, by construction** the gate compares the expression against a **recorded number**, not against what the expression MEANS. An expression whose every node's `trace` reproduces its children's reproduces the traced dim exactly — and passes — while naming a symbol that has nothing to do with the axis it describes. **The corruption is in the identity of the symbol, and arithmetic self-consistency is blind to identity.** The class it was built for is the one where the annotation DISAGREES with the trace; the class where it AGREES and is still wrong was invisible to it from the first line.
* **the instance** `CogVideoX-5b-I2V/vae_encoder`, tensor `aten.slice::36::out_0`, traced `[1, 128, 3, 112, 176]`. Its temporal dim, traced **3**, is a chain of additions over `s1` — the SPATIAL latent-height symbol — and resolves to **129 036**. Its H, traced 112, resolves to **90**, which is `latent_w`: the name of another dimension. Its W, traced 176, resolves to **5 400**, which is `latent_h × latent_w`: a product. Every one of those expressions is self-consistent at trace, so every one passes the gate.
* **what it cost** Prism evaluated them faithfully and asked for **898 238 MB of activations for a component whose weights are 822 MB** — 944 GB for one component against a 93 GB rack — and refused the model before it ran, advising the operator to obtain more host RAM. The estimator was not wrong. It was reading a corrupted annotation that had been certified as sound.
* **why it was read as an estimator defect for a day** because the number is absurd and the estimator is where absurd numbers come out. The discriminator was to profile the graph at its OWN trace binding, where the peak is 40–55 MB: the estimator reproduces the trace exactly, so the fault is upstream of it. **An instrument that reproduces its calibration is not the thing that is broken.**
* **the discriminator that must replace consistency** plausibility, not self-consistency — a dim resolving to 129 036 for an ordinary request, an H resolving to the value of another axis, a W resolving to a product. None of these is detectable by asking an expression to agree with itself.
* **the census — RETRACTED, then redone: an accident, not a family.** The first census resolved shapes with `InputConfig.to_symbol_map()` and reported 22 components across 12 containers — every video container in the zoo. **That census was wrong, and wrong because of the instrument.** `to_symbol_map()` is a POSITIONAL map: it binds `s0`→batch, `s1`→latent_h, `s2`→latent_w by position. These containers DECLARE their symbols (`s1: time`, `s2: height`, `s3: width`, in `symbolic_context`), and the profiler's real entry point, `build_symbol_map()`, applies those names over the positional base. Prism uses the named one. I used the positional one, bound `time` to a spatial extent, and read the consequences as corruption.
* **what the correct map says** redone with `build_symbol_map` across the same 182 graphs at the same three bindings: **3 components, 2 containers.** And two of those three are a false positive of my own threshold — `Wan2.2-I2V/transformer` and `transformer_2` carry `dim1 = mul[time, height, width]`, trace 150 = 5×10×12, which is **exactly right**: a DiT flattens the latent volume into a sequence. They cross a >100×-of-trace threshold only because the trace is FRUGAL (latent 10×12) against a 512² request — the very trap named two entries below, unapplied to my own census. Their peak estimate is 2 104 MB, entirely sane.
* **what actually remains: one container, one component, one dimension.** `CogVideoX-5b-I2V/vae_encoder`, `aten.cat::6`, `dim2 = add[s1]` with `s1` declared `time` at trace 1: it traces to **3** and resolves to **8 751** when `time` binds to 5. No causal padding does that. Everything else in the zoo resolves as its trace does — `Wan2.1-VACE/vae_encoder` resolves `[1, 192, 1, 56, 88]` to itself, exactly.
* **and the H/W half of this entry was also my artefact** "H resolves to the value of W, W resolves to their product" was the positional map binding spatial symbols to the wrong axes. Under the declared map, H resolves to 112 and W to 176 — their traced values. **What survives is the temporal dimension alone.** The 944 GB refusal stands: Prism computed it with the CORRECT map, which is why its number was right while mine were noise.
* **the lesson, and it is the day's own** the one measurement made from the start with the right instrument — a peak estimate exceeding the whole rack, 1 component of 182 — was right all along and never moved. Four further instruments built on top of `to_symbol_map()` produced four confident, wrong answers. **A map named `to_symbol_map` on a class named `InputConfig` looks like the symbol map; it is the positional base the real one is built from.** Two functions, one obvious name, and the obvious one is the wrong one for this question.
* **is the missing `latent_height` the SAME fault? No, and the data says so** the two looked like one tracer defect on one family — one losing an annotation, the other corrupting it. They are different recording steps. The containers that fail on `latent_height` have an **empty topology summary** (`components.<n>.shapes: EMPTY` on Allegro-TI2V, Open-Sora-v2, Wan2.2-I2V) while their GRAPHS carry the latent input perfectly well (Allegro-TI2V declares `[1, 12, 5, 14, 22]`, Wan2.2 `[1, 36, 5, 10, 12]`). And the corruption counts do not track the loss: the three that lose the summary score 2175, 912 and 369, the eight that keep it score 121 to 1543. **B is an unpopulated summary; C is a mis-named symbol.** Two fixes, not one.
* **what the runtime can and cannot do about C** four structural invariants were tested against the corrupted annotations and ALL FOUR are satisfied by them: each expression reproduces its own traced dim; each symbol carries exactly one trace value per graph (182 of 182); every shape-preserving op resolves exactly as its input does (0 violations in 182 graphs); a convolution's output channel against `weight.shape[0]` is defeated by gated convolutions, whose weight legitimately carries twice the output channels. The runtime has **no independent source for what axis a dimension describes** — the graph's only statement about that IS the annotation under test — so the identity check cannot be built here. It belongs where the tracer holds the module and the real tensor.
* **and the capability that blocks the runtime's fallback** the one check the runtime COULD make is on its own conclusion: a peak that grows far more than the request's own input does. That needs the container's **trace binding** as an anchor, and there is no way to ask for one: `estimate_peak_memory(input_config=None)` does NOT profile at the trace — it silently falls back to `InputConfig()`'s defaults, a 1024x1024 batch-2 request. Every threshold measured against that anchor is measured against a second request. **Named as the missing capability — a trace-binding profile — rather than worked around**; a gate whose threshold was calibrated on a false anchor was built, measured, and reverted rather than shipped.
* **where the fix is born** at build time, not here. A channel count is a weight property and must never carry a symbol; a temporal axis must not be expressed over a spatial one. Compensating in the estimator would make the runtime paper over a build-side limit, which this engine does not do.

---

### 27 — a gate that never passed was never tested

* **date** 2026-09-12 · **machine** Dell · **site** my own invocation of `tools/r33_sys_modules_probe.py`
* **what it reported** `torch in sys.modules at exit: False`, and `0 torch modules loaded`, for **both** Orpheus builds — including the one whose successful run had just been observed importing `site-packages/snac/snac.py` and reaching the HF Hub mid-request. A clean green over a claim that was false.
* **what it actually measured** nothing. Its own preceding line reads `the run exited 2` — an argparse error from the way I passed the arguments. The run never reached the decode path where the vendor package is imported, so `sys.modules` at exit is a statement about a process that did nothing. **The probe exited 0 while reporting on a run that exited 2**, which is why the green looked like an answer.
* **the form, and it is the cousin of entry 24** entry 24 was *recording a condition is not checking it*; this is *a green from a run that did not happen is not a result*. Both produce an artefact that reassures. Together they explain why four prepared campaign scripts were found this same day with no clock door while everyone believed them guarded: an instrument that emits something is read as an instrument that checked something.
* **why it did not cost anything this time** only because the direct evidence existed independently — the plain build's own run log, from the run that produced its WAV, carries `Warning: You are sending unauthenticated requests to the HF Hub` and the `snac` frame. Had the probe been the only instrument, it would have certified the opposite of the truth, and the R34 violation would have been recorded as absent.
* **closed by** not citing it. The verdict in `validation_outputs/orpheus_slug_decision_20260912/VERDICT.md` names the probe run void, says why, and rests the conclusion on the direct log evidence. **The repair the probe needs is its own: a probe must refuse to report on a run that did not exit 0**, rather than describing the `sys.modules` of a process that died at argument parsing. Named here; not yet written.
* **the rule it re-earns** feed a known-good input and see the instrument PASS before trusting any of its failures — and equally, see it FAIL on a known-bad one before trusting any of its passes. This probe had never been watched doing either in this session.

---

### 28 — a refutation made with a broken instrument is not a refutation

* **date** 2026-09-12 · **machine** Dell · **site** `InputConfig.to_symbol_map`, and four instruments built on it in one day
* **what happened** the owner prescribed an identity check on symbolic dimensions — verify that the symbol a dimension names belongs to the axis it claims to describe. I answered that it **cannot be built in the runtime**, and I answered it with a measurement: four exact structural invariants, all four satisfied by the corrupted annotations, and the conclusion that *the runtime has no independent source for what axis a dimension describes — the graph's only statement about that IS the annotation under test.* The owner accepted it. **It was false.** Every graph carries `symbolic_context.symbols` with `{name, trace_value, source}` per symbol — `s1: {"name": "time", "source": "input::args::dim_2"}` — and the tracer has written it from the beginning. The independent source exists and is complete.
* **why the measurement did not show it** because it was taken through `InputConfig.to_symbol_map()`, the POSITIONAL base (`s0` batch, `s1` latent height, `s2` latent width). Bound that way, a video container's `time` axis takes a spatial extent, the resolved shapes explode, and the explosion reads as corruption. The same instrument produced a census of "22 components across 12 containers, every video container in the zoo"; redone through `build_symbol_map`, which lays the declared names over that base, the true count is **one container, one component, one dimension**.
* **why this is worse than an ordinary wrong answer, and is its own entry** a false ASSERTION invites checking. A false REFUTATION closes the question: it says *there is nothing here*, and nobody looks again. This one closed a line of work that was correct, and it closed it with the authority of a measurement — four invariants, exact, reproducible, and all of them measured through the same broken lens. **The cost of a refutation is the enquiry it ends, and that cost is paid silently.**
* **the signal that was there and was not read** one number never moved. Prism's own estimate — 944 GB for the offending component — was computed with the CORRECT map from the first minute and stayed put across every instrument I built, while all of mine moved. **A measurement that holds still while yours move is not agreement; it is the control.** Neither of us read it as one.
* **and the knowledge was already in the file** `build_symbol_map`'s docstring documents the 2026-08-10 root case in full: Qwen3-Omni's mel-frame axis, named `seq_len`, bound to the global text config instead of its 441-frame trace, the estimate collapsing, `block_scatter` packing a 16 GB card to 15.77 GiB with no headroom. It is written at the exact place it needed to be read — **inside the function that was not called.** A note cannot reach the caller who calls the other one, and this same note had already failed once.
* **closed by a refusal, not a note** `to_symbol_map` now RAISES, naming what it used to return and both ways forward; the positional base survives as `positional_symbol_map`, whose name carries its own hypothesis; `build_symbol_map` is the single caller. Measured first: the whole repository contained **exactly one** call site, `build_symbol_map` itself — every other caller was a throwaway script of mine, which is precisely the population a rename would not have reached and a refusal does.
* **seen failing** restoring the old behaviour (`to_symbol_map` returning the base again) turns the refusal test red; the declared-vs-positional test shows the two maps a factor of 12 apart on the same axis of the same graph.
* **the rule** a census is not a result until its instrument has been confronted with a measurement taken another way. And a refutation is an assertion with a longer shadow: it deserves MORE corroboration than the claim it kills, not less.

---

### 29 — a rule checked only at the trace point is blind to every error that cancels there

* **date** 2026-09-12 · **machine** Dell · **site** `forge/tracer/symbolic/rules.py`, the `cat` and convolution rules; visible in `CogVideoX-5b-I2V/vae_encoder`
* **the general form, and it is why none of our gates could catch it** a symbolic rule is validated by ONE comparison: does the expression it produces reproduce the extent witnessed at trace? That comparison is made at a single point of the symbol's domain — the trace value. **Any error that happens to vanish at that point is invisible to it, permanently, by construction.** Not because the check is weak, but because a check at one point cannot separate two functions that agree at that point.
* **the instance** the causal temporal pad concatenates three slices of the SAME tensor to turn 1 frame into 3. The rule sums its inputs' symbolic dims — correct in general — and records `3·s1`. **The truth is `s1 + 2`**: two of the three inputs are fixed-size padding, not independent extents. At trace `s1 = 1`, and `3·1 = 1+2 = 3`. The two functions agree at exactly one point of the domain, and that is the only point anyone looks at.
* **what turned an error into a catastrophe** the causal convolution that follows brings the concrete extent back to 1 while KEEPING the 3-reference expression — its temporal reduction is never applied, so the error does not settle. Each of the seven resnet blocks triples it again: `3 → 9 → 27 → 81 → 243 → 729 → 2 187`. The recorded function is `f(t) = 2187·t − 2184`, and `f(1) = 3` still. **A factor-3 error compounds to a factor-2187 one while remaining exactly correct at the point of validation.**
* **what it cost** Prism evaluates it faithfully and asks for **944 GB of activations for a component whose weights are 822 MB**, refuses the model, and advises the operator to obtain more host RAM. Every downstream instrument inherited the absurdity and none could contradict it, because the annotation passes the only test there is.
* **why no gate of ours could have caught it** the shape resolver's TRUST GATE (register 26) asks precisely this question and nothing else: does the expression reproduce its traced dim? It does. The symbol-consistency check passes (one trace value per symbol, 182 graphs of 182). The shape-preserving-op check passes (0 violations in 182 graphs). **Four independent structural invariants, all satisfied, all evaluated at the same single point.**
* **the remedy, in form** a rule must be checked at a point where the error CANNOT cancel — which for a symbolic rule means **at least one binding other than the trace**. A test whose stimulus has `s = 1` will be green for the very reason that blinded the gate. Where a second binding is not available, the rule's structure must be asserted directly (a causal pad is `s + k` and the `k` is a literal), not inferred from an agreement of values.
* **and the corollary for tests** this generalises the shape rule already in `what-a-green-test-proves.md` — *a test proves nothing at a shape that makes all its branches equivalent*. Here it is sharper: **`1` is the value at which multiplication and addition stop being distinguishable**, and a frugal trace tends to produce exactly that value on the axis being symbolized. The most economical stimulus is the one that hides the most rules.
* **the census — a population, and a small one** the question the instance raises is not *is this container broken* but *where else could a defect of this class not be seen*. `tools/symbol_collision_census.py` asks it of every declared symbol in the catalogue: 182 component graphs, every axis, its trace value, and whether that value is one at which distinct rules agree. **25 components of 182, in 20 containers of 56.**
  * **5 arithmetic collisions**, all at trace 1 — the worst value, where `k*s`, `s+(k-1)`, `s**n` and `s` are one rule. **One of the five is the defect above.** The other four are `Allegro-TI2V/transformer` (`seq_len` from an attention mask), `Janus-Pro-7B/gen_embed`, `Qwen3-Omni/talker.code_predictor…codec_embedding` and `openaudio-s1-mini/codec.decoder`, all `seq_len`. None is known to be defective; none can be CLEARED by the trace-point check either.
  * **34 weight-extent collisions** — an axis whose trace value equals a parameter extent of its own component, so a dim bound to the wrong quantity reproduces the trace anyway (`real-esrgan` height and width at 64; `parakeet/joint` seq_len at 1024).
  * **the batch axis is excluded and counted, never dropped.** It is 58 of the 293 raw flags and nearly all of the 57 at trace 2, and it is a DELIBERATE requirement at those values — a batch symbol is never a literal 1, and 2 is the CFG batch. Listing it buries the rest; `--all` shows it for anyone re-examining the decision rather than its consequences.
* **filed** `D-CAUSAL-PAD-SYMBOL-COMPOUNDS` in `DETTE.md`, with the blast radius measured (2 187 references for the offender, 5 for the next across every 5-D component of 56 containers) and the remedy named as a tracer chantier requiring a re-trace of one container.

---

### 30 — an artefact of the right size is not a success

* **date** 2026-09-12 · **machine** Dell · **site** my own reading of the Allegro-TI2V rebuild chain
* **what happened** the chain built the container, verified the corrected topology inside it, and then **`forge replace` was refused by argparse in under a second** for a missing positional argument. I looked at `models/video/Allegro-TI2V/model.nbx.building`, saw 26.1 GB — the right size, the right name, the right minute — concluded "it is at the rename", and reported the chain as in flight. It had already failed. **The flight recorder held the word `failed` the whole time and I never read it.**
* **the cost** two hours in which the queue was believed to be advancing and was not. The owner found it from OUTSIDE, by querying the hub and seeing a date of 6 September against a local artefact of 14:06 — which is a good catch and also the wrong place to have to catch it from.
* **the form** a file of the expected size is evidence that a process WROTE, not that it SUCCEEDED. Size, name and timestamp are all downstream of the work; the exit code is the only witness that reports on the whole of it. **The witness that adjudicates is the exit code, never the trace left on disk.**
* **its cousin, and the reason this is a form and not an anecdote** the other machine wrote the same shape from the other end: a 10 MB PNG whose file size, dimensions and name were all correct, and 99.7 % of whose pixels were 0 or 255. A plausible artefact is the most expensive kind of silence, because it satisfies every cheap check.
* **why the instrument was already there** `tools/flightrec.py` exists precisely to make an interrupted job legible, writes `failed` into a record before anything else, and the session hook prints it. It was armed, it was correct, and it was not consulted. **Having the instrument and looking elsewhere is a distinct failure from not having it**, and the remedy is not another instrument.
* **closed by** the rule, not a tool: a chained step reports on the STEP'S exit status, and a chain that ends anywhere but its last line is reported as failed until its exit code says otherwise. On this machine the exit code is one command away at every point — `flightrec status`, the `|| { echo FAILED; exit 1; }` the chain already carries, and the record itself.
* **the third of a family in one day** a snapshot path read from a truncated usage line, backticks executed inside an unquoted heredoc, and a positional argument never seen because the same `grep` filtered it out. All three are *a composition I did not re-read*, and all three were caught by something refusing correctly — argparse twice, the builder once (*"a component with no weights is a build failure, never a container to ship"*). The refusals worked. The reading did not.
* **five in one day, and the fifth names the remedy.** After the four below came a
  request built by `request_args` — the single source of truth — then **joined into a
  shell string and re-split on spaces**, so the multi-word prompt became separate
  arguments and argparse refused: `unrecognized arguments: red apple rolling slowly
  across a wooden table`. **A list of arguments passed through a shell string stops
  being a list.**

  All five are one class: *a composition whose quoting I did not think through*. A
  truncated usage line; backticks expanded in an unquoted heredoc; a positional
  filtered out by my own grep; a `-m` message the shell read as code; an argument
  list flattened to a string. **The remedy is not better quoting — four of the five
  happened AFTER I had diagnosed the mechanism and written down the fix.** It is to
  stop generating shell from nested heredocs at all: the launcher is written in
  Python, where an argument list stays a list and `subprocess` takes it without a
  shell. A rule you must remember at every call site is a rule you will not apply;
  a shape that has no quoting to get wrong needs no remembering.

  And the whole family was caught by something refusing correctly — argparse three
  times, the builder once (*"a component with no weights is a build failure, never a
  container to ship"*), bash once. **The refusals worked every time. The reading
  did not.**
* **and the fourth arrived while this entry was being committed**, which is the most useful thing in it. The commit carrying this text used `git commit -m "…"` with DOUBLE quotes, so the shell expanded the backticks and parentheses inside the message body — `command not found`, `syntax error near unexpected token '('`, and git then read fragments of the prose as pathspecs. **I had already diagnosed this exact mechanism earlier the same day and written that the fix was `<<'MSG'` with quotes, and then did not use it.** Knowing a rule and applying it are different acts, and the gap between them is not closed by knowing the rule harder. **A commit message is a file, and it goes in through a quoted heredoc — never through an argument the shell reads.**

---

### 31 — two decisions, each correct alone, wrong together

* **date** 2026-09-12 · **machine** Dell · **site** `forge/tracer/worker.py`, two branches ~800 lines apart
* **the form, and it is the hardest of this register to prevent** every other entry has something to correct: a check that checks nothing, a name that lies, a green over a run that did not happen. **Here both decisions are right.** There is no one to correct and no line to call wrong — only a COMBINATION to forbid, and nothing in the code says the two are related.
* **decision one, and it is load-bearing** the single-frame I2V encoder is traced at `T=1`, deliberately, with its reason stated where it is done: *"the vendor causal frame-chunk loop unrolls per trace chunk (3 chunks at T=9 broke replay at T=1); T=1 trace == T=1 runtime, single chunk, exact."* Raising it breaks replay. The comment even ends *"H/W stay symbolic"* — its author did not expect T to be.
* **decision two, and it is reasonable** a pixel-space video input `[B, 3, T, H, W]` gets an explicit per-dim role declaration, because *"the naming convention has no rule for the positional 'args' input, so T/H/W froze at the trace values."* Declaring the layout is exactly right; it was added to FIX a freezing defect.
* **what they do together** a **constant axis receives a symbol, at the value where every rule agrees.** `k*s == s+(k-1) == s**n == s` at 1, so the causal pad's `s + 2` was recorded as `3*s`; seven resnet blocks compounded it to `2187*s - 2184`, still exactly 3 at the only point the trace check looks at; and the engine asked for **944 GB of activations for a component whose weights are 822 MB**.
* **why no review catches it** the two branches are eight hundred lines apart, each carries a correct justification, and neither mentions the other. A reviewer reading either one agrees with it. There is no diff in which they appear together, and the defect exists only in their conjunction.
* **closed by** naming the conjunction where one of the halves already stands: the role declaration now asks whether the axis is constant, and gives it a role OUTSIDE the symbolizing set when it is. Measured on the live container, `2187 -> 1` references and `aten.cat::6` returning to the `literal 3` it always was, with `batch`/`height`/`width` keeping their own names.
* **and the near-miss that is worth as much** the first attempt used `dynamic_dims_spec`, which symbolizes the chosen dims — and, **by its own documented contract, renames every one of them `seq_len`** ("a variable SEQUENCE axis"). All three surviving symbols came back as `seq_len`, and since the engine's `build_symbol_map` binds BY NAME, a height would have bound to a sequence length at runtime. **A worse defect than the one being fixed, shipped as its fix.** It was caught by re-reading the differential's own numbers instead of its verdict line — the discipline that failed seven other times the same day and held here.
* **the rule** when a value is chosen for one reason and interpreted for another, the two places must name each other. A justification that is complete in itself is exactly the one that will not mention the constraint living elsewhere.

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

31 entries, of which five are placeholders and 26 carry a site. Two
machines, two weeks of concentrated looking. Every one of them produced silence
or a green rather than an error, and **not one was found by a test** — they were
found by users, by contradictions between two numbers, by reading generated
code, and twice by another instrument built for something else.

That is the argument for the register: a class this large, whose members are
invisible to the suite by definition, is only tractable if its instances are
written down where the next person will look.

### 32 — a door placed at two of the three save sites, on a path the work does not take

`forge trace --component vae_encoder` exited 0 and printed `Saved: ... (3305 ops)`
on a graph the depth-collision door refuses when run against the same file by
hand. The door had been wired at the two `graph.json` save sites found by
grepping for the R19 prune call in `tracer/orchestrator.py`. The video trace
takes neither of them: it runs in the `trace-worker` SUBPROCESS and saves at
`tracer/worker.py`, the third site.

`tracer/patterns/dead_subgraph.py` says **"the three graph.json save sites"** in
its own docstring, as placement guidance for exactly this kind of pass. That
docstring was read while placing the door, quoted in the commit that placed it,
and two sites were wired.

What makes it this class and not an ordinary miss: the trace **passed**. A gate
at a call site the work does not take is green for the reason that blinds it, and
its green was read as "25 clears the graph" for the twenty minutes before the
file was checked by hand.

Seen failing: with the third site wired, the same command refuses and names the
symbol, its trace value and the candidate.

**The rule that generalises**: when a pass must run at every exit of a stage,
count the exits from the code, not from the search that found the first ones —
and a docstring that names the number is an assertion to CHECK, not a note to
nod at.

### 33 — a flag accepted upstream, dropped downstream by `parse_known_args`

`--trace-time 33` was declared on `forge trace`, declared on `trace-worker`, and
propagated into the worker command line by the orchestrator. `run_worker()`
re-parses `sys.argv` with its own parser and ends with `parse_known_args()`, so
the option was discarded without a word and the worker traced at its seed of 25.

The failure did not look like a dropped flag. The door then recommended 33 on
every iteration, and the fixed-point driver reported eight steps of `25 -> 33 ->
33 -> 33 ...`. Read as data it says "this axis does not converge"; the truth was
"this axis was never moved". Two minutes of GPU and a wrong conclusion about the
model.

**The rule**: `parse_known_args` converts an unknown option into silence. Any
option that crosses a process boundary is declared on BOTH sides or it does not
exist — and an option accepted at the top and discarded at the bottom is a silent
bypass wearing the name of a decision, which is precisely what
`docs/reference/proving-by-doors.md` forbids when it says the opening must be
named and never silent.

### 34 — a census that re-counted the class it had excluded four hours earlier

`tools/symbol_collision_census.py` excludes the batch axis by NAME, with a
comment explaining that a batch symbol at 1 is a deliberate project decision and
that listing it buries the actionable axes — of 293 flagged axes it was 58.

`tools/depth_collision_census.py`, written the same day to supersede it at depth,
did not carry the exclusion. It reported **117,426 dims at extent 1** and a list
of "surprises" headed by LLM and audio components. Every one of them was a batch
axis at 1 on every tensor of a large graph. The instrument's own predecessor
contained the fix, in a comment written to prevent this exact reading.

It also produced a false negative in the other direction: attributing to `time`
every dim owned by another symbol turned a residue into a claimed "172-deep
structural plateau, no stimulus escapes them", which was written into
`forge/tracer/worker.py` as the justification for a stimulus value. With the
three-way split (this symbol / another symbol / no symbol at all) the structural
count across all 182 local graphs is **zero**.

**The rule**: a deliberate exclusion is part of the measurement's definition, not
a detail of one implementation. When an instrument is superseded, the exclusions
are the first thing ported and the first thing tested — otherwise the successor
is a regression wearing a larger number.

`tools/depth_collision_census.py` was WITHDRAWN rather than repaired: repairing it
would have produced a second copy of `tools/stimulus_from_depth.py`, which does
the same reading with the three-way split and answers with an actionable value
per symbol instead of a total. It never reached a commit that claimed its
numbers; the numbers it produced are recorded here so the retraction outlives the
file.

### 35 — an instrument that answered with a number obtained by the extrapolation it exists to distrust

The depth-collision door refuses a graph and names "re-trace with `time` at 25".
That value is computed by evaluating the graph's recorded expressions at a point
OTHER than the one that produced them — which is the exact operation the door
exists because nobody can trust.

Where it is simply invalid: when the traced program unrolls a loop whose trip
count depends on the symbol. The Wan VAE encoder emits 1448 ops at T=9, 2271 at
17, 3305 at 25, ~100-130 ops per frame, and every module group is exactly linear
in the chunk count `(T-1)//4+1` — 3 chunks to 5 chunks moves the four groups
8 -> 14, 21 -> 35, 6 -> 10, 3 -> 5, fitting `3k-1, 7k, 2k, k`. Each re-trace
re-unrolls, and the blind count stayed at **135 through T=17, 25, 33 and 41**
while the recommendation walked 25 -> 33 -> 41.

The refusal was right every time. The number beside it was an invitation to keep
paying for traces.

**The remedy shipped**: the value is now printed as a CANDIDATE that says what it
is, the authority is the loop that re-traces
(`nbx/campaigns/prepared/stimulus_fixed_point.py`), and a blind count that does
not FALL between two iterations is reported as a plateau — measured, not assumed.

**The rule**: an instrument may refuse without knowing the remedy. When it offers
one, the offer carries the same burden of proof as the refusal, and "I evaluated
my model outside the range where it was fitted" does not meet it.

### 36 — the statistic, across eight instruments in one day

Of the eight instruments that carried a claim on 2026-09-12, **every one that was
seen saying NO told the truth; the ones never seen refusing did not.**

Seen refusing, and correct:
1. the clock door — named the two diverging cards before any campaign started
2. the timeout refusal — refused a run whose own recorded cost exceeded its clock
3. the flight-recorder refusal — refused an unrecorded long GPU run
4. `to_symbol_map`'s refusal — the one call site it had was the right one
5. the artefact witness — refused the stale cached graph on its first production run

Never seen refusing, and wrong:
6. the depth-collision door at two of three save sites (entry 32) — passed a graph
   it refuses by hand
7. the differential script that printed `before max 2187, after max 1` and
   concluded NO CHANGE — it diffed by common tid across a 389 -> 265 op rewrite
8. the build-progress reading — a `model.nbx.building` at 26.1 GB reported as a
   chain in flight while the recorder said `failed` (entry 30)

The asymmetry is not luck and it is not about care. An instrument is exercised by
the act of refusing: you cannot watch it say no without learning what it read. An
instrument that only ever agrees is never exercised at all, and its agreement is
the same shape whether it is correct or absent.

**Operational consequence**: the cost of a gate is not its code, it is the
injection that turns it red. Landing a gate without one buys a green whose
provenance is unknown — and the register now has thirty-six entries saying what
that costs.

## The executable form of entry 30's rule

Entry 30 ended in a sentence: *a verification names the artefact it read, and
refuses if that artefact predates the change it verifies.* A sentence in a
register is read by whoever is already suspicious, and both instances of entry 30
happened to someone who was not.

`tools/artefact_witness.py` is that sentence as a refusal:

```python
from artefact_witness import witness, StaleArtefact
w = witness(cached_graph, ("git", forge_repo, "tracer/symbolic/depth_gate.py"))
print(w.line)   # names path, size, sha12, write time, and the reference commit
```

It prints the artefact's identity whether it passes or fails, and raises
`StaleArtefact` when the file is older than the commit that made the change. Seen
both ways before it was trusted: it passed the re-traced Wan encoder written at
15:05:51 against a commit at 14:48:40, refused the CogVideoX encoder written on
2026-06-25 against the same commit, and then refused a stale cached graph on its
first production run inside the class-E chain — which is the only reason that run
did not report on a state the change had never reached.

The reference may be a commit (`("git", repo, pathspec)`), another artefact
(`("file", path)`), or a raw timestamp with a label. A witness with no reference
point is refused at the CLI: that is a timestamp, not a verification.

**What its conservatism costs, measured the same day.** The rule dates a change by
the last commit touching the referenced file, so an UNRELATED edit to that file
makes every earlier artefact stale. It happened within the hour: a commit that
only extended a CLI flag's reach touched `tracer/worker.py`, and the witness then
refused a graph traced twenty-four minutes earlier. The graph was re-traced — and
came back with the SAME sha, `7216b9fb1478`, bit for bit.

That is the right trade and the reason is R27/R28: because traces are
bit-reproducible, a false alarm costs sixteen seconds and returns a proof that the
change was inert. A rule that refused to refuse until it could tell relevant edits
from irrelevant ones would need to model what every change touches, which is the
reasoning it exists to replace. **A conservative refusal whose false positives are
cheap to clear is better than a precise one that has to be right.**

### 37 — a build that packaged a video model without its backbone, and said COMPLETE

`forge build` produced a 5.45 GB CogVideoX-5b-I2V container against a 21.6 GB
hub artifact and exited 0. The snapshot's `transformer/` held `config.json` and a
103 KB weights index and nothing else — its 11 GB had been purged on 2026-09-07
under R38 — so the backbone was copied as a generic MODULE rather than a
component.

Everything the build printed was true. `BUILD COMPLETE`, `Graphs: 3/2 copied from
cache`, and a container write audit that listed every removed transformer shard
by name. The one line that carried the verdict read `Components: vae,
text_encoder`, in the middle of a successful-looking summary, and the chain that
called it checked `NBX.exists()` — an existence test, which is the weaker half of
entry 30's lesson applied to an artefact of the wrong size instead of the right
one.

The upload of that container onto the working hub slug had already started. It
was caught by comparing 6.3 GB against the 21.6 GB the hub listing showed — by
eye, which is not a gate. Nothing was lost on the hub: `replace` uploads to a
distinct key and repoints only after checksum verification, and the upload was
stopped at 0% of 6.73 GB. The complete LOCAL container was lost, because
`--overwrite` had already replaced it before anything could object.

**Two placements, and only one of them is a door.** The first version of the
refusal ran after the build and reported over the wreckage. It now runs at entry,
on the snapshot, before the builder is called.

**The predicate is the cause, not the symptom.** Comparing the topology's
declared components against the manifest flagged `vae_encoder`, a registry alias
built from `vae`'s weights and deliberately absent from the manifest of every
healthy video container. A door that fires on a correct build is uninstalled
within the week. The shipped test is: a directory holding a `*.index.json` and no
weight file declares weights it does not have.

**The rule**: when a build degrades a component instead of failing on it, the
degradation is the defect. A pipeline stage may not silently change WHAT it is
producing — and the summary line that records the change is not a warning, it is
an artefact-integrity claim that nothing checked.

### 38 — a census predicate that was a tautology at the value it was applied at

Asked how many catalogue components unroll a temporal loop, the first instrument
scored each graph by how many of its module groups repeat a number of times `r`
explained by the chunk count `k`: `r % k == 0 or r % k == k-1`. It reported
**seven components at a perfect 1.00** across five containers.

Every one of them had `k = 2`. At `k = 2` the predicate reads
`r % 2 in {0, 1}` — **every integer passes**. The score was 1.00 by construction
and said nothing about those graphs.

The tell was in the same table and was legible before the verdict: the ONE
component with two trace points and a fitted slope — `Wan2.1-VACE/vae_encoder`,
517 ops per chunk, the case the whole question came from — scored **0.79 and fell
below the threshold**. An instrument that ranks the proven case beneath seven it
invented is not mis-tuned, it is measuring something else.

Read on the two fitted points, the real signature is far sharper and needs both
of them: between k=5 and k=7 the twelve module groups move as `27k-40`, `19k-26`,
`16k-17`, `13k`, `13k-12`, `7k`, `4k`, `3k-1`, `2k`, `2k-2`, `k`, and one
constant. **Eleven of twelve affine in k.** A divisibility test on one graph
passes `2/k` of integers by chance and cannot separate "r copies because k
chunks" from "r copies because the architecture has r of them".

**The rule**: state the null before the score. A predicate that accepts a known
fraction of random inputs must be reported against that fraction, and one whose
null reaches 1.0 at the values actually present in the data has no power there
and must REFUSE rather than return a number. The replacement measures two trace
points from the same tracer and fits the slope; its positive and negative
controls sit on the same model — encoder 517 ops per chunk, decoder flat at 384.

### 39 — a heterogeneous rack, and a card chosen without reading its size

Twice within ten minutes, on a rig whose four cards are two 16 GB V100s and two
32 GB V100s:

* `Allegro-TI2V` was run pinned to `CUDA_VISIBLE_DEVICES=0` and died at
  `aten.convolution::0` asking 9 GB with 11.4 GB already live. Card 0 is a 16 GB
  card and the container is 19 GB.
* The unroll measurement campaign was pinned to card 1 — the other 16 GB card —
  and reported `no graph` on trace after trace. On card 3 the same command
  succeeded in fifteen seconds.

Neither is a memory bug and neither says anything about the artefacts under test.
Both were about to be written down as one: the first as "the re-uploaded Allegro
container does not run", the second as "these components cannot be traced".

**And the pinning itself was the error, not just the choice of card.** Prism's
job is placement across the rack; `CUDA_VISIBLE_DEVICES=0` removes the rack and
then reports that the model does not fit. The retry on `"2,3"` was worse — it
remapped the ordinals and produced a sticky `cudaErrorIllegalAddress`, which is
the class the memory already records: anything resolving a real device index
fails under pinning, so an unpinned run comes BEFORE calling a failure a
regression.

**The rule**: on a heterogeneous rack, a card index is not a resource. Read
`memory.total` per card before choosing, and prefer no pinning at all so the
placement engine sees what it was written to see. A failure from a pinned run is
a fact about the pinning until an unpinned run says otherwise.

### 40 — a door that caught the extreme case and passed the partial one

The starved-snapshot door landed the same morning (entry 37) tested a component
directory for "an index and NO weight file at all". `CogVideoX-5b-I2V`'s text
encoder holds `model-00001-of-00002.safetensors` and not
`model-00002-of-00002.safetensors` — both casualties of the same 2026-09-07
purge that took the transformer. The directory has a weight file. The door passed
it.

The build then wrote a 17.32 GB container carrying **half a text encoder**,
reported `BUILD COMPLETE`, and its upload onto the working hub slug ran for five
minutes.

What caught it was not a gate. It was a component-by-component size comparison
against the installed container, run by hand because 17.32 GB did not match the
21.6 GB the hub listing showed:

| component | installed | new archive | delta |
|---|---:|---:|---:|
| text_encoder | 8.87 GB | 4.66 GB | **−4.22 GB** |
| transformer | 11.05 GB | 11.05 GB | 0 |
| vae | 0.81 GB | 0.81 GB | 0 |

A halving in exactly one component, with the others byte-identical, is a missing
shard and not a dtype change — and the shard count was one `index.json` away.

**The sharper predicate needed no new information.** The index NAMES its shards.
It now reads `weight_map` and refuses on any declared file that is absent,
naming it: *"text_encoder/ model.safetensors.index.json -> 1 of 2 shard(s)
absent: model-00002-of-00002.safetensors"*.

**The rule, and it is the second time today**: a predicate written for the case
that motivated it will catch that case. Ask what the ALMOST-right input looks
like — the directory with some of its weights, the container with most of its
components, the graph with one axis still blind — because that is the input that
gets past and the one that is expensive. The extreme case announces itself; the
partial case is indistinguishable from success until something compares it
against what it should have been.

**And the comparison that caught it belongs in the tool, not in a person.** The
previous artefact — the installed container, the hub's size, the last build — is
a declaration of what this thing should contain, and every one of those three was
available to the code that wrote the container.
