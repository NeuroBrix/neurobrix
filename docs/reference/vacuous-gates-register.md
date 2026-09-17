# Register of vacuous gates — instrumentation that lies by construction

> **Each of the week's errors was seen by a measurement, none by rereading.**
> (Volta certificate roadmap, 2026-09-14: five errors, five measurements —
> the owner's line, kept at the head of this register on 2026-09-16. The
> latest instance the same day: a dtype rule wired into the wrong map,
> caught by a 56-container sweep, not by the eyes that wrote it.)

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

72 entries, of which five are placeholders and 67 carry a site. Two
machines, two weeks of concentrated looking. Almost every one produced silence
or a green rather than an error — and two do the opposite, which is why they are
here rather than elsewhere: **65** (a door that held a COPY of its authority's
format list and refused every upscaler) and **67** (a gate whose classifier
assumed both graphs shared a stimulus, so it scored a correct repair FAIL and
could not see the transition it is named for). A false refusal costs what a
false pass costs — a card idle, a repair postponed — and its cause is the same:
an instrument whose model of the world excludes the case in front of it.

**"Not one was found by a test" was true until 2026-09-12 and is no longer.**
They are still overwhelmingly found by users, by contradictions between two
numbers, by reading generated code, and by another instrument built for
something else — entry 40 was a component-by-component size comparison run by
hand because 17.32 GB did not match the 21.6 GB a listing showed. But entry 38
was found by its own instrument's output table, where the one case with a fitted
slope sat BELOW seven the predicate had invented, and the sentence you are
reading was corrected because the test that counts this register caught the nine
entries added without updating it. A test can find one of these. It has to be a
test of the thing's OWN consistency — a count against its entries, a proven case
against its ranking — rather than a test of the code the instrument watches.

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

### 41 — `ast.parse` said "parse ok" on a file Python refuses to run

A timeout was added to two hub requests in `forge/forge.py`. Both calls already
carried a `timeout=10` further down, past the end of the six-line window the edit
had been read in, so the result was a duplicate keyword argument.

The check run before landing it was:

```python
python -c "import ast; ast.parse(open('forge.py').read()); print('parse ok')"
```

It printed **parse ok**. Duplicate keyword arguments are rejected when the AST is
COMPILED, not when it is parsed — `ast.parse` builds the tree and stops, and the
duplicate-name check lives in the symbol table pass that `compile()` runs after
it. So the instrument answered a narrower question than the one it was asked, and
its answer was true.

`forge.py` was unrunnable for twelve minutes. The Open-Sora conversion chain
started in that window, called `forge build`, and reported a `SyntaxError` in its
own log — where it reads as a build failure of the model rather than as a broken
tool.

**The rule**: a syntax check is `compile(source, name, "exec")`, never
`ast.parse`. More generally — when an instrument and the thing it stands for are
two different phases of the same pipeline, the instrument is only as good as the
phase it runs. Asking "does this parse" and reading it as "will this run" is the
same substitution as asking "is this file the right size" and reading it as "is
this file complete" (entry 30), and it fails for the same reason: the cheap
question was answered honestly.

Second observation, recorded because it recurred twice in an hour: **both defects
came from an edit window too small to see the whole call.** A `time` import at
line 27 was missed by `head -25` and briefly read as a missing import; this
`timeout=10` at the end of a call was missed by a six-line grep window. Reading
six lines around a match is reading a fragment, and a fragment of a function call
is not a function call.

### 42 — an internal listing read as a different instance, and a sentence written before the act

Two from one evening, recorded together because they are the same shape.

**The listing.** With the public name blocked, the hub's internal entry point
was probed: `http://10.0.0.39:3000/api/models` answered 200 with **12 models**
and no `CogVideoX-5b-I2V`. For several minutes that read as a *different
instance* — a staging copy — and publishing to it would have been silent and
wrong. It was the page size. `?limit=200` returned 47 with the same record ids
as the public hub. A first page compared with a whole catalogue is a fragment
compared with a thing (entry 41's rule, again): the number was true and the
comparison was not.

Now every publish command asks the registry for its listing at entry and
refuses one that does not answer in the hub's shape. The case it exists for is
an address that returns HTTP 200 with *something*.

**The sentence.** A report said *"I have disarmed the three chains that
publish"* — past tense — while all four were still running. It was composed
before the act, in exactly the form the house rule forbids for numbers: written
before the measurement. Nothing was harmed, because the interception fails the
handshake before any data is sent; the report was still false when it was read.

**The correction that matters for whoever reads the debt in six months.** The
interception was the LaLiga/Cloudflare judicial block applied by Spanish
operators during matches, not an attack. Its certificate is **self-signed by a
blocking device**, not forged: nobody impersonated the hub, an appliance signed
its own name. The refusal was right either way — *bypassing an interception is
deciding alone that it is benign* — and the remedy was not to wait it out but to
stop taking the road it watches: the rack now publishes through its own
network (`docs/reference/workshop-layout.md`).

### 43 — a MAC prefix read as a machine

Diagnosing `SlowDownWrite` from the object store, the ARP table was read for
three addresses: the store (`bc:24:11:…`), the hub (`bc:24:11:…`) and the
export host (`b8:59:9f:…`). Two shared a prefix and the third did not, and that
became a topology in the report: *"three distinct machines — the store and the
hub are two VMs, the NAS is a separate Mellanox box."*

`b8:59:9f` is Mellanox's OUI. It identifies the maker of a network card. The card
is the hypervisor's own 100 Gbps link; the "separate box" was the same host the
two VMs run on, and all three sit on one ZFS pool — which is the fact the whole
diagnosis needed and the inference pointed away from. The owner established it on
the host. The conclusion drawn from the wrong topology (*"the store's trouble
is not the export's"*) was the opposite of true: they are the same spindles.

**The shape**: a number whose subject was never established, again. A MAC prefix
answers *who made this interface*; it was read as *what machine is this*, and
the answer to the cheap question was true. It is the same substitution as an
`ast.parse` read as *will this run* (entry 41) and a size read as *is this
complete* (entry 30).

**The rule**: an identity comes from the thing's own declaration — the host's
pool layout, the VM's config, a `hostname` — never from a property that merely
correlates with it. Where the declaration is on a machine you cannot read,
the cell says *not established*, and the report carries the topology as a
question rather than as a finding.

### 44 — a rule inferred from one incident whose cause was confounded

*2026-09-12, this machine.* After a double mains cut, the object store began
refusing three writes in four with `SlowDownWrite`, and a 118 GB upload crawled
at 398 kB/s. The diagnosis reached the pool: one rotational ZFS pool carries the
exports and every VM's disk, and a download, a build and an upload were running
at once. A rule was written the same hour — *"never a download, a build and an
upload at the same time; one pool writer at a time"* — turned into a serial
queue, endorsed by the supervisor, and then applied to everything heavy,
including the GPUs: four V100s sat at zero for hours behind the upload.

The rule was refuted on three facts, two of which were already in the report
that wrote it. (1) During the incident the host's I/O was near zero and the load
was falling: the pool was not saturated. (2) The store's `/minio/health/cluster`
answered 503 — it had marked its own drive as hung, and stayed in that state
until the owner rebooted the host, after which the first probe read 9.78 MB/s
and zero refusals. A wedged state, not a capacity limit. (3) The owner's
history: this machine had run three or four simultaneous copies for months
without an incident. An inference made in a panic does not hold against months
of practice.

**The shape**: a cause placed on a single point where two different rules
produced the same symptom. "Three writers on one pool" and "a store that has
marked its drive hung" both present as *every write is slow*; only one was
tested, by the serial queue "resolving" an incident that the reboot had
resolved an hour earlier. The queue was the vacuous gate: green because the
harmful state was already gone, read as proof that it had removed it.

**The rule**: a rule inferred from one incident is a hypothesis until a second
instance or a controlled measurement confirms it, and it is written as one. The
concurrency policy is withdrawn entirely; what remains is a **sensor** —
`tools/export_quiet.py` reads bytes off the export and refuses a heavy write
under 40 MB/s — and the one measured fact from a separate incident (2026-09-07:
`SlowDownWrite` on a 548 MB/s burst, settled by adaptive pacing from 40 MB/s).
A measurement in place of a policy. The withdrawal carries its reason so that
the rule does not return in six months under another name.

### 45 — a mention recorded under a key nobody stores

The ruling of 2026-09-12 says an unscreened seat *carries the mention where it
is recorded*. The launcher recorded each seat under the screen's own
de-duplication key — the constexpr kwargs of the launch — while Triton stores
the chosen configuration, and the replay cache writes it, under the autotuner's
SHAPE key (the `keys` arguments' values, then every argument's dtype). The
stamping in `capture()` matched `(kernel, repr(key))` across the two spaces and
never matched anything. Found by the production demonstration on 2026-09-13,
which is the only reason it was found: **three announcements, zero records
stamped**, the certified directory untouched — and a unit suite green around
it, because every unit test handed both sides the same key.

**The shape**: the weight dict's two key spaces (feedback of 2026-09-08), again
— a fact recorded in one key space and read back in another, with nothing to
say the spaces differ. **The rule**: a record and its reader share ONE key,
computed by ONE function; here `autotune_shape_key` rebuilds Triton's key from
the live arguments and every seat is recorded under it. The demonstration is
the gate: `unscreened_in_production.py` reads *demonstrated* only when an
announcement, a stamped record and an untouched directory all hold together.

### 46 — an oracle delivered, tested, and never joined to the provider it was written for

`kernels/oracles/conv2d_fp64.py` landed on 2026-09-12 with its own `ORACLES`
table and nineteen tests against torch at 1e-12, its docstring naming the
screen's uncovered 11.5 % as the reason it exists. The live screen's provider
(`kernels/screen_oracle.py`, merged from the other machine the next night) has
its own `ORACLES` table — GEMM only — and reads no other. Every convolution key
kept being announced *"no oracle for this kernel"* while the reason text said
the convolution family was the uncovered set and the oracle module said it was
the cover. Found by a test that failed on the merge for a different reason,
then by reading both tables side by side.

**The shape**: entry 17's — a helper whose every test passes can still have no
seam. **The rule**: an oracle is delivered when the provider CALLS it, and the
proof is a live screen line adjudicating a key of that family, not the module's
own suite. The two tables are now one at import (`ORACLES.update(...)`), and
the reason text names what is covered rather than what is not.

And the live proof found the second half the same hour: joined, the provider
was CALLED on every conv key and still returned None — it built its operand
dictionary from `tuner.nargs`, the POSITIONAL arguments, while `kernel_height`,
`stride_*`, `padding_*`, `groups` and `fp16` are constexpr launch KWARGS and
live in `meta`. A `KeyError` caught as "no reference", on every key, with the
provider's own three GEMM oracles unaffected because they need no constexpr.
The provider now takes the launch kwargs (`_call_screen_oracle` passes them to
a provider whose signature accepts them), and the test hands it a conv key
whose constexprs come only through `meta`.

### 47 — a screen de-duplicated by a key that ten shapes share

The correctness screen runs at `prune_configs`, once per autotune key, and kept
a `seen` set so a launch is never screened twice. The set was keyed by the
screen's own key — the constexpr kwargs of the launch. `real-esrgan-x4` meets
ten `conv2d_forward_kernel` shapes with one and the same constexpr tuple
(3×3, stride 1, padding 1, groups 1, `fp16=False`): the screen ran on the first
and returned the other nine unscreened without a line, and the run's log read
*"screening at key …"* exactly once for ten sweeps. Found on 2026-09-13 by
counting announcements against sweeps in a live log — one against ten — after
entries 45 and 46 had put the shape key and the oracle in front of the screen.

**The shape**: entry 45's, one step earlier — the same wrong key, used this
time not to record but to decide whether to look at all. **The rule**: a
de-duplication key is the identity of the thing de-duplicated; here that is
the shape key Triton stores the choice under, and the screen now de-duplicates
by it. And the converse record: a seat the screen DID adjudicate is now
written `screened: true` with its adjudicator's name, so a silent entry can no
longer be read as a verified one.

### 48 — a load filter applied in the graph's key space to the loader's keys

`consumed_weight_names()` (2026-09-09) hands the triton loader the set of
parameter names some op in the graph reads, so that a MoE build does not load
the experts its trace never routed to — a 38 % saving, measured. It speaks the
GRAPH's names. The loader filters the INDEX's keys by exact membership, and the
two spaces are joined only after loading, by `_reconcile_weight_keys`'s unique
suffix rule. Every key whose two names agree passed; the one whose names differ
by a prefix — `token_embed.weight` in the index, `encoder.token_embed.weight`
in the graph — was skipped before the reconcile could see it, and its first
consumer met `None`. Found on 2026-09-13 by Wan2.2-I2V-A14B's proof by run:
compiled rendered nine frames, triton died at `aten.embedding::0`, and
triton-sequential named the operand.

**The shape**: entry 45's, one layer down — a fact stated in one key space and
tested in another, with a count that agreed (242 consumed, 242 in the index)
and hid the one name that did not. **The rule**: a filter is applied in the key
space of the thing it filters; the consumed set is expanded into the loader's
space with the reconcile's own rule (`consumed_in_loader_space`) before the
loader sees it, single-part suffixes decide nothing, and the direction of doubt
is to load. The gate: the same proof, on the same container, in triton.

### 49 — a control arm that inherited the directory its isolation depended on

The paired campaign gives every arm of every repetition its own replay
directory, so a control arm sweeps instead of reading back what an earlier arm
wrote (entry 36's lesson, 2026-09-11). The directory is named from the output
path, and a second run into the same `--out` found the first run's
`B_replay_r0..2` already there: its control arm read every key back, swept
nothing, and the cell reported 0.85× — a "gain" of the directory measured
against a control that had done no work. It overwrote a clean 3.20× cell taken
forty-nine minutes earlier, and the document rendered the new number until the
count `keys 48/0` was read.

**The shape**: an isolation that holds by construction on a fresh path and by
nothing at all on an inherited one — the same defect as a scratch cache the
previous run left warm. **The rule**: the thing an isolation depends on is
checked at entry, not assumed from the name; a cold arm now refuses a replay
directory that already holds an artifact, names the path, and the deliberate
opening is `NBX_ALLOW_REUSED_REPLAY=1`. The overwritten cell is marked in the
campaign's `PERTURBED.json` with the clean numbers quoted from the run log,
and the document says the record measured nothing.

### 50 — a filter that knew one of the two readers

The triton weight loader takes `only=`, the set of weights the graph consumes,
and skips the rest — the saving is real (11.8 GB of unrouted experts on a
30 GB MoE component). Entry 48 fixed its key space. The full suite on that
tree then failed ten triton cells with the same two sentences: *"requires
embed_tokens weight"* on four VLMs, two audio-LLMs, a TTS backbone and a warm
serve, and `'NoneType' object has no attribute 'nbx_dtype'` at `aten.mm::0` on
both int4 builds. The graph is not the only reader of the weight dict. A
language model whose graph takes `inputs_embeds` never consumes its token
embedding — the flow handler reads it by name to build the context and the
tied logits — and an int4 build stores a consumed `X.weight` as three keys the
graph never names. Both were loaded before the filter existed; both went
missing the day it arrived, and the 04:28 suite the same night already showed
them under the noise of thirty-seven failures.

**The shape**: a premise stated as a law — *"a parameter no op consumes cannot
be reached by execution"* — true of the replay loop and false of the handler
around it. The join between the two key spaces was also a second rule written
beside the reconcile's rule, and the two disagreed on ambiguous suffixes.
**The rule**: ONE function binds loader keys to graph names
(`GraphExecutor.bind_weight_keys`, the reconcile's own three passes), the
filter is computed from it, every non-block key is loaded whatever the graph
says (what the flows read is never inside a block; what the filter saves is
never outside one), and a storage triplet is wanted through its stem. Gate:
`tests/unit/runtime/test_consumed_weights_reach_the_loader_key_space.py`,
seen red three of five on the old code; the regression cells are the proof by
run.

### 51 — an empty set read as no answer

`neurobrix upscale` under `CUDA_VISIBLE_DEVICES=""` — the cell the regression
suite runs to stand for a machine without a graphics card — died with *"No
CUDA GPUs are available"* at the first constant. The per-environment profile
tag (2026-09-05) returns None when detection fails, and it returned None when
detection SUCCEEDED and found nothing; the caller serves the shared
`default.yml` for None, which on this rack describes four V100s, and Prism
planned `cuda:0` for a process that could see no card. The cell had passed on
2026-09-03 and was red from the day the tag landed.

**The shape**: two different facts — *I could not look* and *I looked and
there is nothing* — collapsed into one value, and the fallback written for the
first applied to the second. **The rule**: an empty visible set is an
environment of its own, tagged `cpu`, with its own detected profile; None is
reserved for the exception. Gate:
`tests/unit/prism/test_an_empty_visible_set_is_a_cpu_host.py`, seen red on
the old code; the proof by run is the same cell.

### 52 — a green over zero cells

The budget-unified gate's second phase — the byte matrix over every model whose
strategy does NOT change, the safety net under the five pinned pairs — called the
campaign with `--machine` and neither `--models` nor `--family`. The tool's default
selection keeps the cached models whose family equals `--family`; unset, that is
no model at all. It printed the table's header, an empty body, and the script
printed `gate termine`. RUN.md read as a gate that had found nothing to differ.
Found by asking why a "matrix over the rig" had taken zero seconds.

**The shape**: a selection that can be empty, and a gate whose success is the
absence of a bad row — so an empty selection is indistinguishable from a clean
one. **The rule**: a campaign that selects nothing refuses at entry and names the
flags that select; and the gate script names its families. Gate:
`tests/unit/tools/test_campaign_refuses_an_empty_selection.py`, seen red on the
old selection; the matrix re-armed per family behind the suite re-run.

### 53 — "the engine skips those", written for two engines, true of one

Prism sizes a component by the weights its graph consumes, and the solver's
comment (2026-09-09) said *the engine skips those* — the unrouted MoE experts,
11.8 GB of 30.6 on DeepSeek-Coder-V2-Lite. The triton loader did skip them.
The compiled loader read every key of every shard, so the plan that placed
Qwen3-Omni's thinker at 5.2 GB on one 32 GB card was executed by a loader
that put 57 GB there. Four native cells of the suite died that way on a
quiet rig from a frozen tree, and the 15:20 run had hidden them behind a
foreign process's memory ("environment, not engine").

**The shape**: an equivalence between the two engines stated in prose beside
the code of one of them, with no gate on the other (R30's silent asymmetry,
in a loader rather than a kernel). **The rule**: the compiled loader takes
the same `only=` set, computed by the same function, and a test reads a
shard through it. Gate:
`tests/unit/core/test_compiled_loader_loads_what_the_plan_budgeted.py`,
seen red on the old reader; the proof by run is the four native cells.

### 54 — a set computed before the rewrite that adds its readers

The consumed-weight filter reads the FINAL graph, "after any fusion pass",
and for the llm family that is true: the MoE fusion runs at graph load. For
a MoE LM packaged under another family the fusion waits for the flow's
declaration at execute time — after the weights were loaded from the
un-fused graph, in which no op consumes the experts the trace never routed
to. The fused kernel then read them all, and Ming-Lite-Omni's first MoE
block met `'NoneType' object has no attribute 'data_ptr'` under triton on a
quiet rig, from a frozen tree, with the filter of entries 50 and 53 in
place. Prism sized the same component on the same un-fused graph: 5.2 GB
planned for Qwen3-Omni's thinker, 57 GB executed.

**The shape**: a set that is exact for the graph it reads and stale for the
graph the engine runs, because a declared rewrite comes later. **The
rule**: a rewrite that adds readers loads what it now reads — the declared
fusion recomputes the set and loads the difference through the engine's
own loader, in both engines — and Prism sizes the fused copy of a routed
component. (A first version moved the fusion into the factory; the review
found it read a path no package has and would have applied the LM's norm
to the talker's own router — the register's own class, caught before the
commit.) Gate:
`tests/unit/runtime/test_a_declared_fusion_loads_what_it_now_reads.py`,
seen red before the fix; the proof by run is the Ming triton cell.


### 55 — a waiter that watches the result and never the producer

A follow-on chain polled the campaign record for the line `== rejeu 2
termine` before taking the rig. The producer of that line — the rerun-2
chain — was killed by PID at 17:12 on 2026-09-13, a few seconds AFTER its
pytest had finished and written its complete output, and before the chain's
own closing echo. The waiter kept polling for a line no process would ever
write: four V100s idle from 17:12 to 21:25, found by the session that
resumed after a connection cut. The record itself was complete (8 failed, 1
passed, in `suite_rerun2.log`); only the marker was missing.

**The shape**: a waiter whose only test is "is the result here yet?" cannot
distinguish *not yet* from *never*. It has no liveness on what it waits for,
so a producer that dies after succeeding — or before starting — starves it
silently and for ever. Entry 4-of-the-chain-rules ("a DONE marker is written
only on success") is right and does not cover this: the job succeeded, the
marker-writer was what died. **The rule**: a waiter checks that what it waits
for is still producing, not only that the result is absent — it holds the
producer's PID (or a heartbeat the producer refreshes) and turns into a
refusal, with the producer's name and last sign of life, the moment the
producer is gone without its marker. Gate: `tools/wait_for.py` (shared
brick, seen refusing on an injected kill of the producer), and every chain
armed from this entry on waits through it rather than through a bare
`until grep -q` (the chain already running at 21:26 keeps its bare loops;
the first user of the brick is the post-certification waiter of the same
night). Written 21:25 by the resumed session; the brick and its injection
land with the commit that carries this entry.

### 56 — a directory keyed by a profile two memory classes share

The certified autotune directory is keyed by `(vendor, profile, kernel,
dtype, shape)`, and the profile is the vendor profile FILE in force —
`nvidia/volta.yml`, one file for this rack's two V100 SKUs (16 GB cards 0
and 1, 32 GB cards 2 and 3). The lookup read nothing about the executing
card (`kernels/autotune_certified.py:lookup`, before this entry), so an
entry proven on one memory class was served as-is to the other. The proof
recorded only `machine.hardware_profile`, a name that says the memory for a
pinned card (`auto-v100-16gb-16g`) and, for a rig-wide run, the SUM and the
first card's model (`auto-4xv100-16gb-96.0g`) — the card that ran is
unknown there. Counted on the 7 191 entries at 21:45: 5 644 proven on a
16 GB card, 17 on a 32 GB card, 1 530 on the rig with the card unknown.

**The shape**: an instrument whose key omits an axis on which its claim is
made, so two different claims read as one entry and the record cannot say
which was proven. **The rule**: a proof says which card's memory it was made
on (`proof.machine.device = {ordinal, visible_devices, name, memory_mb}`, read
from the card the certifier's inputs were placed on — the ordinal is the CUDA
ordinal in the visible set, the visible set is written beside it, and the
memory is what the class reads), an entry carries one certification
per memory class (`variants` by `<N>g`), the lookup receives the executing
card's class (its tensor's device, read in the Prism profile) and serves only
what covers it; an unknown class — the proof's or the card's — is served
nothing and said in clear (`certified for 16 GB, this card is 32 GB, not
served`). Legacy proofs are read by their profile name where it is a single
card and stay `?` where it is the rig: those 1 530 serve no card until
re-proven. Gate:
`tests/unit/kernels/test_an_entry_serves_only_the_memory_class_it_covered.py`
(12 tests, seen red before the code existed). The measurement this entry
does NOT contain: whether a config proven on 16 GB differs from one proven
on 32 GB for the same shape — same GV100 die, same locked clock, only the
HBM differs; expected identical, to be measured on a 32 GB card when one is
free, and the coverage certified regardless because the rule is coverage,
not expectation.

### 57 — a measurement that exists only where it was made

The certifier writes its directory entry by entry, atomically per file, and
nothing carried those files anywhere until the pass ended and a person
committed them. Three mains cuts in three days (2026-09-11, 09-12, 09-13
23:45 UTC) each found hundreds of certified entries on disk and nowhere else
— 971 on the Friday, 1 222 on the Saturday night, every one verified whole
after the cut (`88c8af0`). They survived by the file system's journal. A
truncated write at the wrong moment would have cost a file; a dead disk, the
pass; and each cut cost the hours it took a person to come back, verify, and
commit. The clock lock had the same shape: it was restored by the memory of
whoever woke up after the cut, three times, and the door refused every timed
run in between.

**The shape**: a result whose only copy is on the machine that produced it,
in a state a cut can reach at any moment, while the process that could have
carried it elsewhere waits for the end of a pass that takes hours. Not a
vacuous gate but its neighbour — a gate (`autotune check`, the door
`rig_clock.py`) that was right and could not act, because the act was left
to a person. **The rule**: what a long run produces is carried off the
machine WHILE it runs, by a process that holds the producers and runs the
gate before each carry — `tools/certified_checkpoint.py`, one per
repository, an interval or the producers' death as its trigger, every
remote pushed and READ BACK, a refused file named and left; a cut then costs
one interval, not a pass. And a state the system must be in at every boot is
put there BY THE SYSTEM at boot — `tools/systemd/nbx-rig-clock.service`
runs `rig_clock.py --restore` after the driver, applies the protocol to
every card, reads every card back, and refuses if one did not take. Gates:
`tests/unit/tools/test_the_certified_directory_is_checkpointed_while_written.py`
(7 tests; the gate-refusal and no-card-door injections seen RED 2026-09-14
00:03-00:05 — the door test's first form was itself vacuous and is recorded
in its docstring), `test_rig_clock_door.py` (restore: a driver that says
"All done" and changes nothing is refused, seen RED with the read-back
neutralised). Proof on the rig, 2026-09-14 00:04: card 1 set to 1312 by
hand, `systemctl restart nbx-rig-clock`, all four cards read 1290.

**Measured 2026-09-14 00:08-00:14 UTC, the measurement entry 56 said it did
not contain** (`tools/memory_class_sample.py`, campaign
`2026_09_13_certification_tail/sample_16_to_32`): the first 50 matmul census
keys certified on card 2 (32 GB, pinned, tree d1def45, clocks 1290/877 read
on every card, no throttle reason active) against the directory's 16 GB
entries for the same keys (certified 2026-09-07 00:07-00:26, same engine
0.5.3, same Triton 3.6.0): **37 keys choose the same configuration, 13 a
different one, 0 without a counterpart** — and the 32 GB card's best time is
**1.11x to 1.19x the 16 GB entry's on every one of the 50 keys (median
1.175)**. The expectation "identical" is contradicted twice, and the rule
(coverage per class, never expectation) was the right one for a reason the
entry did not know. What this does NOT yet say: whether the 17 % is the card
or the day — the 16 GB side was measured a week earlier under conditions this
record does not hold. The control is queued behind the certification on card
0 (`guard_after_card0.sh`: the same 50 keys on a 16 GB card today, against
the same 09-07 entries); until it reads, the 17 % is UNADJUDICATED between
"the 32 GB SKU is slower at the same clock" and "the rig was slower on 09-07".

**Adjudicated 2026-09-14 01:47 UTC — it was the day, not the card.** The
control (`sample_16_control_card0`: the same 50 keys certified on card 0,
16 GB, idle, at 1290/877): against the directory's 09-07 entries for the same
class the 16 GB card today reads **median 1.176× (1.113–1.194)** — the same
figure the 32 GB card read at 00:13. And the 32 GB draft against the 16 GB
draft of the same night: **median 0.998 (0.976–1.008), 39/50 same
configuration** — the two SKUs agree at the same locked clock, as the die
says, and the 11 configurations that differ are near-ties inside the timer's
noise (the 32 GB draft differed from the 09-07 entries on 13, the same
level). What differs is the 09-07 measurement itself: **5 628 of the
directory's 8 515 proofs are of 2026-09-07 and record no clock** — they were
made five days before the clock door (`71a0e7c`, 09-12) and the certifier of
that day did not write `clocks_mhz`. The ratio 1.176 is within 1 % of
1530/1290 = 1.186, the V100's boost over the protocol lock; that the rig was
unlocked on 09-07 is the likely reading and is NOT proven (nothing recorded
it). Their deviations stand (numerics do not depend on the clock); their
timings are not comparable with any proof made behind the door, and the
configuration each chose was chosen at an unknown frequency. The rule that
follows: a proof without a recorded clock is re-proven at the protocol clock
when a card of its class is idle — the two 16 GB cards are, tonight.

Second control, card 1 (16 GB, 02:03-02:09 UTC, same 50 keys): 1.174×
(1.110-1.187) against the 09-07 entries; 0.998 against card 0 the same
night; 1.000 against the 32 GB card. Three cards in one night agree to 1 %;
the 09-07 proofs alone read 17 % faster. And the configuration chosen agrees
between two 16 GB drafts of identical conditions on 34 of 50 keys, between the
32 GB and either 16 GB draft on 37-39 — so a certified choice is stable on
about seven keys in ten and a near-tie the timer decides on the other three,
which is what the catalogue's "certified choices contradicted by the runtime
sweep (near-ties)" column has been counting.


### 58 — a trace at a small shape, and an index that wraps at a large one

`mochi-1-preview` died on CUDA error 700 at `aten.mm::1` of its VAE at every
attempt since 2026-09-10 (D-MOCHI-CUDA-700-AT-MM). Two `compute-sanitizer`
runs — 7 200 s on 09-13, 18 000 s on 09-14, cards 2+3 — ended on their
budget with the run's log silent and no error summary: five hours of
instrument, nothing measured. `--triton-sequential` with
`CUDA_LAUNCH_BLOCKING=1` named the op in 19 minutes: the matmul at
M=1 068 480 × N=2048, fp32 out — 2 188 247 040 elements, past 2^31. The
kernel computed `stride_cm * offs_cm` with `offs_cm` an int32 arange; past
row 1 048 576 the product wraps negative and the store leaves the
allocation. The trace saw M=33 264. Upstream: triton-lang/triton#832
("index computations are done in int32 even for large tensors" — intended
C semantics, the maintainer's word), fixed the same week by three other
projects (comfy-kitchen#172 on Triton 3.6.0, Triton-distributed#209,
FlagGems#6083) with the same in-kernel `.to(tl.int64)`.

**The shape**: a kernel proven correct at every shape a trace or a census
ever presented, and wrong at the first shape whose element count crosses a
power of two no test had reached — the register's "shape it runs at"
class, at the scale of an integer type. **The rule**: an index computation
that can exceed 2^31 is done in 64-bit at the site that can exceed it, and
the gate is a shape chosen to cross the boundary
(`tests/unit/kernels/test_a_gemm_beyond_two_billion_elements.py`:
M=1 100 000 × N=2048, K=64, fp16 — 2 252 800 000 elements, C 4.5 GB, the
rows before, at and past 1 048 576 checked against a host product; seen
RED 05:37 UTC on the kernel as it was — the same error 700 — GREEN after).
The cost was measured, not assumed (ptillet's register-spill warning is
about matmul, and no Volta number existed upstream): the same 50 matmul
keys certified on card 2 before and after, same card, same locked clock —
best time after/before **median 0.996 (0.980–1.020)**, 38/50 same
configuration (the noise level of two identical drafts, entry 56), every
deviation within tolerance; and three models with the setting pinned
(`NBX_DISABLE_AUTOTUNE=1`, card 3) — TinyLlama, Kokoro, Sana MultiLing at
4 steps — **byte-identical** before and after. And the instrument lesson
beside it: a memcheck that returns nothing in five hours is not "still
running", and the engine's own op-by-op mode with launch blocking is the
first instrument for a fault, not the last.


**The class, the same night (06:08-07:10 UTC).** With the GEMM fixed, the
op-by-op run went past `mm::1` and died at `aten.add::14` — the broadcast
variant of `add`, whose sibling had been widened by hand in May (Sana 4K
VAE) while it had not: the same bug written twice is a missing brick. An
audit found the int32 form `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` at
**152 sites in 105 kernels**; all were widened at once, at the program id
(`pid.to(tl.int64) * BLOCK_SIZE + …`, FlagGems #6083's form). The first form
tried — casting AFTER the product — was wrong, and its source-reading gate
was green over it: the product `pid * BLOCK_SIZE` had already wrapped
before the cast, and the 2.25e9-element fill still faulted on card 2
(06:31). A gate that reads text proves the text; the GPU test beside it
(`test_a_flat_kernel_beyond_two_billion_elements.py`: ones, a transposed
materialisation, an add over 2 252 800 000 elements, the elements before,
at and past 2^31 read back — RED on the tree of 90fefd4, GREEN after)
proves the arithmetic, and the gate now refuses both forms. Measured: four
models byte-identical before and after with the setting pinned (TinyLlama,
Kokoro, Sana MultiLing at 4 steps, whisper-large-v3-turbo, card 2), and
their timings within 3 % over three warm runs of the after arm (the first
after run pays the recompilation of every changed kernel and is not a
measurement of the kernel).


**The tile forms, 07:33-07:45.** Past `add::14`, the op-by-op run died at
`aten.native_group_norm::26`: `chan_start * HW` and `batch_idx * C * HW`,
products of program-id-derived scalars with dimensions — the same wrap, in
the tile form no arange-regex could see. Rather than a third sweep for a
third spelling, every `tl.program_id(...)` in the kernels now reads
`.to(tl.int64)` at its source (293 sites in 157 files): every offset
derived from a program id is 64-bit by construction, and the gate refuses
an int32 program id. Beyond-2^31 tests: GEMM, the flat kernels,
group_norm (N=1, C=64, HW=35.2e6).

### 59 — an instrument that keeps measuring after its context died

The certifier that re-certified the speech leg's keys on card 1 (07:09-07:38
UTC) met, in the middle of its list, a matmul of 2 188 247 040 output
elements on a tree whose kernel still wrapped (`suite_53012f7`, before
90fefd4): CUDA error 700. The CUDA context is then dead — sticky — and
every later launch and malloc in the process fails with the same code. The
certifier counted the fault as one FAILED key and went on: **227 further
keys were reported FAILED, each with "GPU malloc failed (error 700) for
256 bytes"**, a fault that happened once written 228 times, and 227 shapes
that were never measured recorded as if they had been tried. The 453 keys
certified before the fault stand (their proofs re-read); nothing after it
was a measurement.

**The shape**: an instrument whose failure handling treats every failure
as local, when one class of failure ends the instrument's ability to
measure anything at all — the census after that point is a list of the same
sentence. **The rule**: a failure that poisons the process stops the run at
once, names the key it died at, and exits non-zero; what was measured
before it stands, and the summary says where to resume. Gate:
`tests/unit/kernels/test_the_certifier_stops_at_a_sticky_cuda_error.py`
(three tests; `sticky_cuda_error` forced false seen RED 07:44 UTC). The
lost 227 keys are re-certified on the fixed tree — the small ones on a 16 GB
card for the class the guard needs, the Mochi-size ones on a 32 GB card, the
only class that can hold them.

### 60 — four green tests beside a rule wired into the wrong map

The rule "an uncalibrated component's activations are estimated in fp32,
the conservative path's dtype" shipped on 2026-09-16 with four green tests
and one injection seen red. Every test called the helper that decides the
dtype; none called the solver that consumed it. The helper was wired into
`_resolve_component_dtypes` — the per-component map that sizes the WEIGHTS
and that the executor takes as the component's dtype — not into the
activation estimate. The 56-container `--explain-plan` sweep, run before the
commit because the owner's rule says a plan is budgeted under the model
that executes it, read **thirty weight bills doubled and nine strategies
moved** (Qwen3-30B's 57 GB of fp16 weights planned at 115 GB; Ming and
Qwen3-Omni pushed from block_scatter to lazy_sequential by weights that do
not exist). Rewired to the activation estimate alone; a fifth test now calls
the map and asserts the weights' dtype is untouched — it was RED on the
first wiring.

**The shape**: a helper tested in isolation is proven correct; where it is
CALLED FROM is a second claim nothing tested — the same class as entry 17
(a helper whose every test passes can still have no seam), from the other
side: the seam existed and led to the wrong consumer. **The rule**: a change
that adds a decision to a computation tests the computation's OUTPUT (here:
weights unchanged, activations moved), not only the decision. And the
measurement that caught it — the sweep — is not optional when the change
moves a plan: it is the gate the owner named.

### 61 — a comparison of two absent keys, read as "identical"

The CogVideoX-2b fingerprint walk (2026-09-16) compared the two runs' op
records on a key named `sha256`. The instrument writes `sha`. Every record's
`sha256` was None on both sides, None equalled None 39 116 times, and the
walk printed **"IDENTICAL op by op"** for two runs whose videos differed in
99 % of their pixels. Read on the right key an hour later: the first
differing op is `aten.view::2` [26, 16, 60, 90] — the initial latent — and
32 412 of 39 116 ops differ after it. The cause (a seedless request's Triton
stream ran unseeded: a present-None slot shadowed the default, and the
default was read from the container instead of the merged defaults) was
within reach of the first walk; the absent key hid it behind the most
reassuring word the walk could print.

**The shape**: entry 17's family from yet another side — an absent key is
silence, and silence compares equal to silence. **The rule**: a comparison
REFUSES when the field it compares is absent on either side (the re-read
script does; the walk did not), and a verdict of "identical" over N records
states what it compared — `hashed elements per op` beside the count. The
memory that names the key (`sha`) was written in April; the script was
written from the head in September.

### 62 — a capability probe that compiles is not a probe that executes

Found on the Mac (2026-09-16): the Metal fork chose `-std=metal4.1` because a
probe COMPILED under it — and the GPU runtime rejected the metallib it
produced. A probe that compiles proves the toolchain accepts the syntax; it
proves nothing about the device running the artefact. The same day this rack
measured Triton 3.8.0's bundled `ptxas` the other way round — compiled a
`.target sm_70` PTX, then EXECUTED the kernel suite (943 passed) and TinyLlama
on the V100s with bytes compared to the old stack — and that is why that
reading holds. The stack door (`tools/stack_door.py`) was extended in the same
spirit: the wheel's arch list says what it was built for; a cuDNN convolution
and a cuBLAS matmul RUN on every card say what serves it.

**The rule, general**: a capability probe — architecture, shared memory, a
dtype's native support, tf32, bf16, anything that decides a code path —
EXECUTES the path it decides and VERIFIES its result against a reference;
compiling, linking or loading are not evidence. **Census on the CUDA side of
this trunk (2026-09-16)**: the launcher reads compute capability and shared
memory from the DRIVER (attributes, not probes); native bf16 is a hardware
profile flag (`config/vendors`), not a probe; the reduction tile is a
backend-capability table; the certification screen executes every setting
against the fp64 oracle; a candidate configuration is timed by running it.
No compile-only probe found here; the one that fit the shape sits in the
Metal driver's standard selection and is being converted where it lives.

### 63 — four days of "proven by execution" that no one had looked at

Between 2026-09-12 and 2026-09-16 this project wrote *proven by run* on model
after model. Every one of those proofs was a NUMBER: a byte gate between the
Triton arm and the PyTorch arm, a PSNR against another arm, a pixel-dynamics
range, a wall clock. Not one artefact had been opened — no image looked at, no
text read, no sound heard. And a byte gate cannot see a defect present on both
sides: one broken graph upstream breaks both arms identically and the matrix
prints **IDENTICAL**, which reads exactly like success. The register already
held that shape under *the contaminated oracle*; what was missing was the
positive rule that would have forced the look.

The owner's form, and it is R29 hardened rather than a new rule: **a line is
PROVEN only when an artefact of a REAL request — never a trace stimulus — has
been judged by an instrument OUTSIDE the engine, and the verdict is written
beside the file.** Outside means: a text is read, and code it contains is
EXECUTED; a transcription is compared with a text known in advance and with a
third-party ASR; a synthesised voice is read back by that ASR and the words
compared; an image is looked at for the thing the request asked for, with the
degeneracy facts beside the look; a sequence of frames is watched as a
sequence. Agreement between two arms of this engine is an AGREEMENT — a
measurement, named as such in the document, never a proof. And the consequence
that was not being drawn: **a degenerate artefact is an open defect, ahead of
everything else in the queue**, not a table line.

The first application (`nbx/campaigns/2026_09_16_vitrine`) paid for itself in
nine seconds: the standing belief that `real-esrgan` had been "rendering white
on both arms for four days" was false. On a real photograph it returns the
input's scene at four times the size, correlation 0.998 with the bicubic
reference, 199 358 distinct colours. What the record actually held was a
different, closed defect on `swin2SR-x2`; the upscaler's byte-matrix cells had
been run on a 64×64, 138-byte fixture and reported *identical* — an agreement
between two arms about a toy. Nobody had opened the file.

**The rule's own trap, seen in the same hour**: the degeneracy facts for that
photograph read *417 uniform rows of 1792* — 23 % — because a studio photograph
has a white background, and the input carries the same 23 %. A threshold alone
would have called a correct artefact degenerate. The numbers bound the look;
they do not replace it, in either direction.

### 64 — a guard that scanned the wrong level, and had therefore never seen anything

`tools/skips_that_hide_a_red.py` exists to find the form where a red becomes a
skip: a catch-all `try` around `from x import a`, so the absence of `a` — the
function under test — reads as the absence of a machine. It scanned
`tree.body`, with the comment `# MODULE level only`. The form it hunts is
almost always written INSIDE a test or a fixture, so on 2026-09-16 the scan
reported **0 findings across 274 test files**, and a walk of the whole tree
reported **20, in 17 files** — every one of them invisible to it since the day
it was written. A clean report from a guard looking at the wrong level is
indistinguishable from a clean tree.

Found by applying, to our own tools, a rule the repository's new second reader
carries in its configuration: *any guard or predicate that inspects only
top-level structures when the thing it guards can be nested*. The same class as
entry 17 — a helper correct in isolation, wired where nothing reaches it —
except here the helper was reached, ran on every file, and answered zero.

**The rule**: a guard states the DEPTH it inspects, and the test that lands with
it injects the thing it guards at a depth greater than one. The detector now
walks (`risky_guards`, any depth), and its test injects the nested form inside a
function and inside a class body, seen red against the old scan.

**The twenty it now sees are a finding of their own**, filed as
`D-SKIPS-THAT-HIDE-A-RED-TWENTY-GUARDS`: each is a named import inside a
catch-all, and the fix is scope, not removal — guard the package import, import
the names inside the test where a missing one fails.

**CLOSED the same day.** All twenty scoped, the scan reads zero, and the unit
suite under the device door went from 56 failures to 50 — none new, six gone.
Two of the twenty were more than a scope fault and are recorded here because
they were found by doing the work, not by reading it:

* `test_autotune_correctness_screen.py` guarded five tests behind
  `_detect_gpu_backend() is not None`, which answers **which backend this build
  can address** — a fact about the install. It answered "cuda" on a machine with
  no visible device and the five tests failed at their first allocation with
  `cudaErrorNoDevice` instead of skipping. That is entry 62's class exactly, a
  probe that compiles standing in for a probe that executes; the probe now
  allocates one element and frees it.
* `test_gather_scatter_oob.py` declared its device probe BELOW one of the tests
  that opens a device, so that one ran unguarded while its two siblings skipped.
  Order of declaration decided which tests were protected.

And the detector itself was one entry short of correct: it walked the whole
`ast.Try` node, so an import in a HANDLER — `except Exception as exc: from x
import E; assert isinstance(exc, E)`, which NARROWS a broad catch — counted as
a swallowing guard. It walks `node.body` now; a nested handler inside that body
is still counted, because an exception there does reach the outer catch-all.
Both directions land with their injection.

### 65 — a door that copied its authority's list, and refused what the authority accepts

The re-trace door asks "is there a COMPLETE snapshot here?" before it spends a
card. Its format test listed four layouts: a diffusers pipeline, a transformers
model, a NeMo archive, a NeMo directory. The build toolchain's own detector
(`tracer/format_detector.py`) accepts a fifth — BARE_WEIGHTS, a `.pth`/`.pt`/
`.safetensors`/`.ckpt` with no config of any kind — which is how every upscaler
in the catalogue ships. So on 2026-09-16 the re-trace of `real-esrgan-x2`
stopped with **"no COMPLETE snapshot on the export or in the download
directory"** while the checkpoint sat in the directory it had just been handed,
and the same refusal had been waiting for every upscaler since the door was
written.

This is the mirror of the usual entry and belongs in the same register: a gate
whose green is empty here has a red that is empty, and a false refusal costs
what a false pass costs — a card idle, a repair postponed, and a message that
reads like a fact about the disk. What makes it the same class is not the
direction, it is the cause: **the door held a COPY of a rule whose authority
lives elsewhere.** Four of the five layouts were transcribed correctly. The
fifth had never existed at transcription time, and nothing re-reads a copy.

**The rule**: a door that reproduces another component's decision names that
component and re-reads its input, never its conclusion. Where the authority
cannot be imported (this one lives in the separate build toolchain), read the
DECLARATION the authority reads — here the registry entry, which names the
checkpoint file for exactly the models that ship as one, because several
variants share one upstream repository and only the entry says which is which.

The fix admits bare weights **only** when the registry names the checkpoint and
it is present, which is also what keeps the guard the door was built for: a
stopped diffusers download (Sana 4K, 6 GB of shards, no `model_index.json`,
2026-09-07) is bare weights too and declares no checkpoint, so it stays refused.
Landed with both injections seen red —
`tests/unit/tools/test_snapshot_bare_weights_is_a_format.py`: the old four-layout
form fails the upscaler case, and a naive "accept any bare weights" fails the
stopped-download case and the wrong-variant case.

### 66 — a census that followed symbol ids where the thing that can be lost is a dimension

`tools/where_the_symbol_chain_breaks.py` follows each declared input symbol
through a graph and names the operator where its expression became a literal.
Its second run reported **167 breaks and 82 symbols never carried**, with
`aten::view` at 78 — and among the twenty cleanest of those breaks stood
TinyLlama, DeepSeek, Voxtral, VibeVoice, canary-qwen, granite-speech and every
T5 text encoder in the catalogue, all with the same `[1, S, C] -> [S, C]`
flatten said to have frozen `seq_len` at 23.

Those models run at seq_len ≠ 23 every day. Opening ONE of the twenty against
its graph took a minute and dissolved all twenty: the view's arguments record
`{"type": "mul", "left": s0, "right": s1}` in full. The census was following
`s3` — `seq_len` declared a SECOND time, on `position_ids` instead of
`input_ids` — which no operation names, because the tracer bound the expression
to `s1`.

**The rule**: what can be lost is a DIMENSION, not a declaration. A tracer
declares one symbol per input that carries the dimension, so any instrument that
iterates over declarations counts a model's `seq_len` once per input and reports
the unreferenced copies as losses. Group by the dimension (name and trace value),
and a carrier of any member carries it. **102 of the 447 declarations are
duplicates** — and every gate that reads the declaration has been counting them.

Corrected: 345 dimensions, **111 breaks** (not 167), **48 never carried** (not
82), `aten::view` **36 in 20 components** (not 78 in 38). The ranking survived;
the magnitudes did not, and a report had already been written with them.

The general form, and the reason this entry sits beside 64 rather than
elsewhere: **the first count that looks like an answer is the moment to open one
row against the source.** Both errors of this census — axis indices read as lost
expressions, then aliases read as lost dimensions — inflated the number and left
the ranking intact, which is exactly the shape that survives a sanity check.

### 67 — the gate that could not see the repair it exists to recognise

The re-trace gate's whole vocabulary — `witnessed`, `symbolized`, `re-expressed`,
`slice-end-symbolized` — reads one transition: a shape argument that was a
literal is now an expression. It reads it by checking that **the expression's
trace value equals the old literal**, which is exact and correct as long as both
graphs were traced at the same stimulus.

On 2026-09-16 the repair WAS the stimulus. `real-esrgan-x2`'s pixel-unshuffle
went from the literal `32` (64//2 at a 64x64 trace) to `floordiv(s1, 2)` of trace
`56` (112//2 at 112x80). 32 ≠ 56, so the classifier scored the transition it is
named for **zero symbolized**, counted every recorded shape in the graph as a
change it could not classify — **2193 beyond annotation** — and the gate answered
**FAIL** on a repair that is correct and proven by artefact at three sizes.

The sibling case makes the shape of the blindness plain: `real-esrgan-x8`, the
same re-trace, came back **byte-identical on both arms** — the strongest evidence
a gate can be handed — and was also refused, by the same count.

**The rule**: an instrument that compares two artefacts states the invariant its
comparison assumes, and detects when that invariant does not hold instead of
reporting a number computed under it. A count taken under a broken assumption is
not evidence of anything, and it may not read as a refusal. The gate now detects
a stimulus change, names which dimensions moved and what the bytes did, and
leaves the verdict to the artefacts. What refused before still refuses: a
corrupted dim, a routing field removed or changed, a run that failed.

This is the mirror of 65 in the same way 65 is the mirror of the rest — a false
refusal costs what a false pass costs, and here it would have cost the repair.

### 62 (addendum, 2026-09-16 evening) — the census that answered "none" had a scope, and the instrument now says it

Entry 62's census read the ENGINE and answered "no compile-only probe found
here". It was right about what it read and it did not read the probes that decide
whether a TEST runs — which is where the next one was, the same day:
`test_autotune_correctness_screen.py` gated five tests on
`_detect_gpu_backend() is not None`.

**The finding underneath it is worth more than the instance.** That call OPENS THE
DEVICE on Metal — `metal_device_available`: *"it opens the real device rather than
checking for the import, because a machine with the bindings and no usable GPU
must not be reported as ready"* — and on CUDA and ROCm it succeeds when the
vendor's runtime LIBRARY loads. So **the same call is an executing probe on one
backend and a naming one on the others**, and nothing at the call site shows which
one you got. Measured in one line under the device door: the probe answers `cuda`
while `DeviceAllocator.device_count()` answers `0`.

Three things landed rather than a reading:

* the function now says at its own definition what it does NOT answer, and names
  `DeviceAllocator.device_count()` — which asks the driver — for callers who mean
  "can I run here";
* `tests/unit/kernels/test_naming_a_backend_is_not_finding_a_device.py` pins the
  asymmetry by asking both questions in a child process with no device visible, on
  a host that has one, and skips where the premise does not hold. Seen red against
  a `device_count` that answers from the install instead of the driver;
* `tools/probes_that_compile_without_executing.py` makes the census repeatable —
  it classifies every function whose name announces a capability decision as
  EXECUTES / DETECTS ONLY / DELEGATES / UNCLEAR, excludes the vendored reference
  tree, and exits 0 because it is a census and an exit code would turn eight
  benign candidates into an alarm.

Re-run over the trunk after the conversions: **8 DETECTS ONLY, all read, none a
device-capability probe** — three ask about a tokenizer, a cache and a DAG; two
are honest pre-filters (`_triton_cpu_available`, `triton_metal_available`) whose
refusal names the install command; three ask whether a FILE is present.

**The rule, sharpened**: "a capability probe executes" is satisfied PER BACKEND,
not per call site. A shared probe that executes on one backend and reads a name on
another must say so where it is defined, because no caller can see it.

### 68 — a guard whose constants were a written list, and the eleventh model it knew nothing about

The trace stimulus for the upscaler family was moved to 112x80 on 2026-08-29 to clear the
collisions that had frozen `real-esrgan`'s spatial dims. The choice was correct and the reasoning
was written out in full: window areas (49, 64, 144, 256, 576, 1024), relative-position table
sizes (225, 529, 961), "the usual embed/feature widths (48, 60, 64, 96, 180)", and the scale
multiples of each.

**Every one of those numbers is a constant in code answering a live question.** The list was
right for the ten upscalers that existed when it was written, and it says nothing about the
eleventh. Worse, it was not even complete for the ten: reading the constants OFF THE MODELS
instead, on 2026-09-16, found that nine clear 112x80 and **`real-esrgan-x2` does not** — its
width times its own scale, 80 x 2 = 160, is the RDB dense-concat width (64 + 3x32) carried by 69
of its convolution weights. The container whose renders were proven correct at four sizes that
same evening sits on a stimulus the rule refuses. Its correctness there is luck, not a property,
and that is the whole distinction this entry exists for.

**The rule**: a guard that must avoid a model's constants READS THEM FROM THE MODEL. Parameters
and buffers for what the graph's literals are made of; the configuration's own integers, and
their squares, for what a view computes from a side and never stores (a window area). And it
holds over the scale multiples, because the collision that cost HAT on 2026-08-27 was with the
trace OUTPUT size, 64 x 4 = 256 — a number the old list had to be told and the new rule finds,
because `hat-l-x4` carries 256 as a parameter extent.

Landed in the build toolchain as `stimulus_collision.py` (`cad67f7`), where the stimulus is
corrected at the source rather than left to a default's luck: clean models are returned unchanged
so they re-trace byte-identically (R27), and `real-esrgan-x2` moves to 112x144. Seen RED on the
real 64x64 case against that model's measured extents, with both of its reasons — the square and
the 64 — and on three injections: the scale multiples ignored, the extents taken from a written
list, the square rule dropped.

**Why it is not the spatial net (b69e51c) again.** The net catches a graph where a dimension ends
up named NOWHERE, which is the loudest outcome and not the only one: a collision can freeze ONE
expression in a graph that carries the dimension elsewhere, and the net sees nothing at all. The
net is downstream and after the fact; this is upstream and before it.

### 69 — a negative control that upstream switched off, and a table that kept printing verdicts

`tools/r33_execution_proof.py` runs each step of the engine's Triton startup in a fresh
process with a cold cache and asks whether `torch` is in `sys.modules` at the end. A table
where every line reads `False` and nothing CAN read `True` measures nothing, so the file
carried a negative control, and the control was Triton's own `kernel[grid]` — whose C++
argument binder imported torch on every backend. That was true, and it was the reason the
NeuroBrix launcher exists.

Upstream made the CUDA driver probe native in Triton 3.7 (triton#9578, #10935). On the
candidate stack's 3.8.0 the control reads `False`. **Nothing announced it.** The table went
on printing a verdict every run, having lost the ability to detect torch at all — every
`False` above it now unfalsifiable, and read by anyone opening the file as thirteen proofs.

The control was chosen from the world outside the repository, and the world moved. The
replacement cannot: it is a bare `import torch`, whose reading is a property of the harness
and of nothing else. The old row is kept one line above as an OBSERVATION — it is still
interesting that upstream no longer pulls torch in — but it is no longer load-bearing.

**The rule**: a control's job is to prove the instrument can still fire. Build it out of
something YOU own. A control that depends on an external implementation detail is a gate
whose off-switch is in someone else's repository, and it will be thrown without a message.

Seen red on the injection that restores the old control (`test_the_last_case_is_a_control_that_cannot_go_inert`).

### 70 — "could not run here" read as "torch was here", on every CUDA box, for weeks

The same file, same day. Its verdict was one line:

```python
clean = all(not t and not e for _, t, e in owned)
```

`t` is "torch was present". `e` is "this step did not run". Two Metal steps in the table
compile to MSL through our own driver, and a CUDA box has no Metal device, so those two rows
error **every single time they are run here**. Folding `e` into `t` meant the table printed
`*** R33 VIOLATION ***` on every CUDA run since the Metal rows landed.

Measured 2026-09-17 on both stacks, one variable apart, while verifying an unrelated launcher
change: candidate stack `*** R33 VIOLATION ***`, old stack `*** R33 VIOLATION ***`, and
**torch appeared in zero owned rows on either**. The alarm had no relationship to its subject.
A verdict that is red whatever happens carries exactly as much information as one that is
green whatever happens, and costs more, because it trains its reader to scroll past it.

The fix is NOT "an error is benign". That swaps this failure for the register's most expensive
family — a step that quietly stops being measured would then read exactly like a step that
passed, and silence and success would again be one reading. Three outcomes are now kept apart,
and each names itself: torch seen in an owned step is a VIOLATION; a step that should run here
and did not is a BROKEN HARNESS; a silent detector control is UNPROVEN. Which steps may be
excused is not a judgement made at verdict time but a declaration each case carries — the two
Metal rows say `"darwin"`, and are excused on `linux` and **nowhere else**, so the same unrun
row read on a Mac turns the table red.

The exit code had the identical defect and was fixed with it: `failures` counted the unrun
Metal rows, so the tool returned 1 on every CUDA box no matter what it found. It now returns
the verdict.

**The rule**: when an instrument can report "yes", "no" and "I could not look", never let two
of those three share a branch. And the excuse for not looking belongs to the CASE, declared
in advance, never to the verdict that would rather be green.

Both defects survived because the verdict lived inside `main()`, below eight subprocess
launches — nothing could reach it without fifteen minutes of card time, so nothing ever did.
It is now `build_report(rows, platform)`, pure, and the injections above run in 0.03 s.

### 71 — a test that chose its card from NVML and then allocated with CUDA

`tests/unit/kernels/test_prefill_determinism.py` proves the prefill route on BOTH memory
classes: chunked inside the 16G window, plain math on a 32G card. To do that it has to find a
card of each class, and it asked `nvidia-smi`.

`CUDA_VISIBLE_DEVICES` is a CUDA-RUNTIME mask. It renumbers ordinals for everything that goes
through libcuda — every allocation the test then makes. `nvidia-smi` answers from NVML, which
sits outside that mask and always reports the whole board. On this rack the 32G cards are
physical 2 and 3, so under `CUDA_VISIBLE_DEVICES=0` the helper returned `big=2`, the test
declined to skip, and `_route_spy` asked for `cuda:2` on a process that owns exactly one
ordinal. The suite went red at `DeviceAllocator.set_device(2)`.

**It is not an error that reads as an error.** It is a WRONG ANSWER, because `2` is a valid
integer in both namespaces and merely names different cards in each. Had the rack's classes
been laid out the other way round the same helper would have returned a plausible ordinal, the
test would have passed, and it would have proven the 32G route using a 16G card.

Found on 2026-09-17 while checking that the launcher change (27b05cc4) had not regressed
anything — so the cost was not only the red, it was a red sitting in the one suite being read
as the verdict on an unrelated fix. That is the second time in one day that an always-red
signal had to be cleared before a real question could be asked (see 70).

**The rule**: decide in the namespace you are going to act in. A device index that will be
handed to an allocator comes from the allocator — `DeviceAllocator.visible_device_memory()`,
added for this and used by `most_free_device`'s caller path, whose inline copy of the same
ctypes walk it replaces. NVML is the right authority for what the RACK has and the wrong one
for what THIS PROCESS may touch.

Seen red on the injection that restores the `nvidia-smi` helper: six of the seven new cells
turn, and the original `test_32g_pow2_window_keeps_prefix_route_on_device` failure reproduces
under the pin.

### 72 — the machine's own profile, written by a process that could see a third of it

`config/hardware/default.yml` is the MACHINE's hardware profile. It is not decoration: a process
with no `CUDA_VISIBLE_DEVICES` reads it as its profile, and it is the fallback when detection is
unavailable. The battery is such a process.

On 2026-09-17 it read:

    # Hardware Profile: 2 x Tesla V100-SXM2-16GB
    total_gpus: 2
    total_vram_gb: 32.0

on a rack of two 16 GB and two 32 GB cards — 96 GB, of which 64 GB and both large cards were
absent. Written 09-16 12:37 by a run pinned to `CUDA_VISIBLE_DEVICES=0,1`. Every unmasked run
since had been planning against a machine two thirds smaller than the one under it, and the
battery re-run this block is working toward would have done the same.

The branch that writes the shared file is only reached by a process that HAS a mask set. Its only
writers are, by construction, the ones most likely to be partial. The guard was:

```python
if tag != "cpu":
```

— which is the 2026-09-13 incident, fixed by name: a process seeing NO card had turned the rack
into a CPU host for every reader of the shared file. **The instance was named, the class was
not**, so every partial view that was not zero-card kept the pen, and the same failure returned
three days later one step up: not zero of four, but two of four.

**The rule**: a file that describes the machine may only be written by a process that can see the
machine. `_describes_the_whole_machine()` compares the detection against NVML's count — and
"unknowable" counts as "no", because the shared file is read by processes that cannot check it,
so a writer proves the right rather than assuming it.

This is the converse of **71**, and the two make one rule: **NVML is the authority for what the
RACK has; the CUDA runtime is the authority for what THIS PROCESS may touch.** 71 was code asking
NVML a process-scoped question. 72 is code letting a process-scoped answer overwrite a rack-scoped
file. Both were found in one morning, both by asking what a green or a red actually stood on.

An existing test pinned the old behaviour — `the human-facing default.yml mirrors the latest
detection` — and it was rewritten rather than deleted, because the file's OTHER role (the profile
an unmasked process reads) is the load-bearing one and the two cannot both hold. The same test
also learned to state its host: it compared against the real rig's card count, so it would have
passed or failed by how many GPUs the developer's box happened to have. Seen red on the injection
that restores `tag != "cpu"`.
