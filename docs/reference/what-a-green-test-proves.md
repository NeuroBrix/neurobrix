# What a green test proves — the two spaces a test can be empty in

A test that exercises nothing is green. It is the most expensive class of defect
this project has met, it has arrived five times, and it always wears the same
disguise: a verdict everyone reads as *"correct"* which really says *"never
ran"*.

A test can be empty in two independent spaces, and passing in one does not save
you in the other.

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

## The one question that covers both

Before believing a green, answer this and write the answer down:

> **What would this test do if the code were wrong?**

If the honest answer is "pass" — because no container reaches the code, because
the shape makes every branch identical, because the assertion reads a key the
producer never writes — then the test is not weak. It is empty, and its green is
worse than no test, because it is read as an answer.

The operational version of the question is the injection: break the thing
deliberately and watch the gate go red. **A gate is worth something only once it
has been seen failing.** That is why every gate in this repository is landed
with the injection that turned it red recorded beside it, and why a green from a
gate nobody has seen bite is a claim, not a result.
