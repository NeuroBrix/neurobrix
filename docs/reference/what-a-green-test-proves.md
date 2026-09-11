# What a green test proves — the two spaces a test can be empty in

A test that exercises nothing is green. It is the most expensive class of defect
this project has met — **sixteen recorded instances across two machines**, all
of them in `docs/reference/vacuous-gates-register.md` — and it always wears the
same disguise: a verdict everyone reads as *"correct"* which really says *"never
ran"*.

A test can be empty in three independent spaces, and passing in one does not
save you in the others.

---

## First space: the CODE the test can reach

**A cell that runs models proves nothing about a kernel no model reaches.**

Measured 2026-09-10 across the 56 containers installed on this machine:

| op | containers that carry it |
|---|---|
| `aten::where` | 39 |
| `aten::tril` | 22 |
| `aten::all` | 10 |
| `aten::_weight_norm` | 2 |
| `aten::argmin`, `aten::min`, `aten::var` | **0** |

Nineteen kernel edits had been landed with a plan to validate them "on the zoo".
Three of the nineteen sites sit in kernels that **no installed container
reaches**. That cell would have been green having touched nothing — and its
green would have been read as coverage of all nineteen.

Two sibling cases from the same week, the same shape:

* A benchmark harness composed every row with `--temperature 0`, fourteen times
  over. Every sampling path above greedy went unentered by the campaign meant
  to cover them.
* The flash-attention clamp read `sdpa_thresholds`, which exists only in the
  seven shipped profiles. On any other card it returned `(None, None)` and
  pruned nothing — a guard that had never once guarded, discovered when a
  user's A40 refused to start.

**The instrument.** `neurobrix coverage <symbol>` answers, before the cell is
written, how many installed containers reach it and which. Ask it first. A count
of zero does not mean dead code — it means *this* cell cannot be the proof, and
a direct kernel test has to be.

Its own limit, stated so nobody discovers it the hard way: it reads graphs, so
it answers questions about the graph. A **runtime decision** — which attention
branch ran, which autotune config was seated, whether a fallback fired — is
chosen while running and is not a node in any graph. That class needs its own
instrument, on the execution path.

---

## Second space: the SHAPE the test runs at

**A test proves nothing at a shape that makes all its branches equivalent.**

Same lesson, different space, and it is the one that is easier to miss because
the test *does* run and the code *is* reached.

The nineteen edits above are mask expressions: `mask = (m < M) & (n < N)`. A
careless rewrite plants `m < (M & n)` — `&` binds tighter than `<` — and on
`weight_norm` that reads `col_offset < (N & 1)`, i.e. one column of N, with the
norm wrong by roughly sqrt(N) in silence.

Run that kernel at `M = 2·BLOCK_M`, `N = 4·BLOCK_N` and it passes. Every lane of
every block is valid, the mask is true everywhere, and a mask that is true
everywhere cannot be wrong. **The bug is invisible at the shape most people
would reach for**, because round numbers are what one types.

So the shapes that make the test a test:

* **reductions**: `M = BLOCK_M·k + 1`, `N = BLOCK_N·k + 3` — the worst case is
  one valid row in the last block of rows.
* **triangular kernels**: `M`, `N` coprime and non-multiples of the block, and
  `diagonal` negative, zero and positive.
* **two-pass kernels** (`weight_norm`): overhang on the axis each pass walks,
  and read the SECOND pass's output — a wrong mask in pass 1 only shows through
  what pass 2 does with the norm.
* **`all` / `any`**: exactly ONE element disagreeing with the rest, placed in
  the tail of the LAST block. A uniformly-true tensor returns the right answer
  under any mask at all.
* **anything with a fast path**: a size below the threshold never enters it.

**The choice of shape is part of the test, not part of its decor.** Write down
why each dimension has the value it has, in the test, next to the value. A shape
with no stated reason is a shape nobody chose.

---

## Third space: the PROSE nothing checks

The first two spaces are about a test that runs. This one is about a claim that
never had a test at all — and it is the easiest to write, because writing it
feels like documenting rather than asserting.

A comment that states a **cost**, a **neutrality** or an **equivalence** is an
assertion exactly like an `assert` is, and it is read as one. The difference is
that nothing re-checks it when the code, the compiler or the hardware moves.

The instance, 2026-09-10. A kernel edit shipped with this comment:

> *"Comparing here is exact for any integer width and costs nothing."*

Nobody had checked it. `cond != 0` promotes the Python literal to i32, so the
comparison gains an `arith.extsi` i8→i32 on **every element** — on `aten::where`,
the kernel 39 of the 56 installed containers reach. The boolean result was
identical; the claim of cost was false. It was found by an IR harness written for
a different purpose, not by anything guarding that sentence.

The correct form (`cond.to(tl.int1)`) emits exactly the cast the compiler already
performed internally, so the IR is identical to the pre-edit kernel — and that
sentence is now **pinned by a test**, with the reason: two forms silence the
deprecation and only one is free.

### The temporal form: a number written before the measurement

Three instances in one day, on two machines:

* a comment asserting `costs nothing` on a kernel nobody had profiled;
* a commit message reporting `245 passed` — written before the suite ran, which
  returned 241;
* a report on the other machine announcing a push that had not happened.

Same defect, and the tell is always the same: the sentence was composed while
the thing it describes was still in the future.

**The house rule.** A number in a commit message, a report or a verdict is
written **after** the measurement, never before. A sentence composed before the
run carries, in itself, the mark that it is unverified — `expected`, `to be
confirmed`, or simply not written yet. There is no version of "it will
obviously be 245" that is not a guess wearing a fact's clothes.

It is not a personal lapse and must not be filed as one. It is a house rule,
because the pressure that produces it — writing the summary while the work is
fresh, then running the check — is structural and recurs on every machine.

**The rule.** A statement of cost, neutrality or equivalence in a comment,
docstring or commit message is either:

* **pinned by a test** that fails when it stops being true — an IR comparison for
  "emits the same instructions", a measurement with its dispersion for "costs
  nothing", a byte gate for "changes no output"; or
* **marked as unverified**, in the sentence itself, naming what would verify it.

There is no third option, and "it is obviously true" is the first option's
failure mode. `costs nothing`, `this is a no-op`, `same instructions`, `no
measurable overhead`, `equivalent to X`, `changes no result` — every one of these
is a measurement someone has skipped, written in the tone of a fact.

This is the same defect as the first two spaces, seen from a third angle: code
the test cannot reach, a shape that makes the branches equivalent, and prose that
no test controls. All three produce a green — or, here, a confident sentence —
where nothing was checked.

---

## A difference is not yet an attribution

A byte gate says two outputs differ. It does not say the change caused it, and
the distance between those two sentences is one more run.

Twice on 2026-09-11 a `DIFFERENT` verdict accused a change of what the model
does on its own:

* `CogVideoX-2b` — three repetitions per arm, three shas, in BOTH arms. The
  record carries `nondeterministic: ["A", "B"]` and the video comparison agrees
  at 43.6 dB. A byte gate cannot adjudicate a model that differs from itself.
* `Kokoro-82M` — printed `DIFFERENT` by a one-run-per-arm tree gate against
  `7de1560`, a commit touching exactly one file, `core/prism/solver.py`. Two
  runs of the SAME tree then produced two shas. The change could not have moved
  it, and did not.

**The rule.** Before attributing a difference to a change, run ONE SIDE TWICE.
If the side differs from itself, the gate has not measured the change and the
verdict says so rather than naming a culprit.

Repetitions make this visible for free — with three runs per arm the record
shows self-difference and the tool marks it. At one run per arm nothing in the
record can separate the two cases, so the verdict now carries
`UNADJUDICATED` and names the run that would settle it. A verdict that cannot
say which question it answered is worse than no verdict, because it reads as an
answer to the interesting one.

## A trap that makes the wrong instrument look green

Third reason to test **structure** rather than **diagnostics**, and the nastiest
of the three because it manifests as a pass.

Compilers cache by source hash. Triton writes compiled artefacts to disk keyed on
the source, so a check that watches for a **warning** fires only on the first
compilation of a given source and reads as **silence** on every run afterwards —
on the same machine, in CI, in a loop. A gate built on "no deprecation warning
was emitted" is therefore green the second time whether or not the defect is
present, and it becomes greener the more it is run.

This happened during the very session that produced this page: a variant compiled
with `warnings.filterwarnings("error")` raised nothing, not because it was clean
but because the artefact came from cache and code generation never ran.

An IR comparison is immune — a cached artefact carries the IR that code
generation produced — which is why the instrument compares what the compiler
BUILT, never what it SAID. The general form:

> Test the artefact, not the commentary about the artefact. Logs, warnings and
> diagnostics are produced by a path that may be skipped; the artefact is what
> the machine will actually run.

---

## The one question that covers all three

Before believing a green, answer this and write the answer down:

> **What would this test do if the code were wrong?**

If the honest answer is "pass" — because no container reaches the code, because
the shape makes every branch identical, because the assertion reads a key the
producer never writes, because the artefact came from a cache and the check
watched a warning, or because there is no test at all and only a sentence saying
it is fine — then the test is not weak. It is empty, and its green is worse than
no test, because it is read as an answer.

The operational version of the question is the injection: break the thing
deliberately and watch the gate go red. **A gate is worth something only once it
has been seen failing.** That is why every gate in this repository is landed
with the injection that turned it red recorded beside it, and why a green from a
gate nobody has seen bite is a claim, not a result.
