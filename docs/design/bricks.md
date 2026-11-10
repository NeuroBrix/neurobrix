# Code in bricks

NeuroBrix is written in bricks. Everything that DECIDES or runs IN A LOOP is a named component — a sub-engine
responsible for one precise task, as Prism is for planning. This page states the rule and the tool that guards it.

## The rule (the owner, 2026-11-10)

For every new or touched component — the engine, nbxtensor, nbxgrammar, a device backend:

1. **One responsibility**, said in one sentence at the head of the module. If the sentence needs "and", it is two
   components.
2. **A written contract**: what it takes, what it returns, what it refuses and under which name. Other components go
   through the contract only — no private name imported, no hidden shared state.
3. **Decision apart from execution**: what chooses (a policy, a plan) is not what executes (a mechanism). It is the
   boundary between the engine and nbxtensor, applied inside each brick.
4. **No catch-all function**: a function longer than a screen, or one that chains a branch per model or per case, is
   cut into named strategies in a table — the data in configuration, no value in code.
5. **A design page before the code** of a new component (its responsibility, its contract, where it plugs in, what it
   replaces), reread by another contributor on the architecture, not only on correctness.
6. **A ratchet** measures it at every change (below); it can only improve.
7. **No over-engineering**: a component exists where there is a real loop or a real decision. Three lines are not
   wrapped in a class.

## The ratchet — `tools/brick_ratchet.py`

It reads the code (AST and tokens; nothing is imported) and counts, for `src/neurobrix` and `nbxtensor/src/nbxtensor`:

| figure | what is counted |
|---|---|
| length | a function's code lines — no docstring, no comment-only line, no blank line |
| depth | the `if` / `for` / `while` / `try` nested inside one function; an `elif` and a `with` add none |
| direction | an import against the declared direction between components |
| private | a private name taken from another module (`from x import _name`) |
| reach | `x._name` where `x` is not `self` / `cls`, counted per package |
| module | a module over 1 500 code lines; a module with no docstring |

**New code** is held to its package's limits, which are that package's 95th percentile on the day they were measured
(`tools/brick_ratchet.json`, dated; lowered when the percentile falls, never raised):

| package | function length | depth | measured 2026-11-10 (functions; p95 / p99 / max length; p95 / p99 / max depth) |
|---|---|---|---|
| neurobrix | 64 code lines | 4 | 4 399; 64 / 169 / 609; 4 / 5 / 13 |
| nbxtensor | 37 code lines | 3 | 2 193; 37 / 74 / 297; 3 / 4 / 7 |

**Existing code** over a limit is listed in `tools/brick_ratchet.baseline.json` (that day: 264 functions and 6 modules
in neurobrix, 104 and 1 in nbxtensor; 76 and 78 private imports; 13 imports against the direction; 705 and 822
reaches; 5 modules without a docstring). No listed function or module grows, nothing is added to a list, and an entry
that no longer offends must leave it (`--update` rewrites the baseline, downward only). The ratchet does not order a
rewrite of what works: a listed catch-all is cut in the change that would otherwise add to it.

**The declared direction** (the json's `direction` table): nbxtensor never imports neurobrix or torch; the engine's
core never imports its doors (`cli`, `serving`); the Triton branch and the kernels never import torch or the doors;
the planner never imports a flow or the executing runtime.

**Exemption**: a `@triton.jit` / `@gluon.jit` kernel body is not held to the length limit — one loop nest is one
kernel — and is recognised by its decorator, never by a list of names. Any other exception is a row here, with its
reason, in the same change:

| date | what | why |
|---|---|---|
| | | |

**With the slim ratchet.** `tools/engine_slim_scan.py` counts the engine's lines the same way — code lines — so
cutting a catch-all into named pieces with their docstrings, or writing a reason in a comment, never reads as growth.

**Running it.** `python tools/brick_ratchet.py` before every landing: it prints each refusal by name and exits 1, or
one summary line. Its tests (`tests/unit/brick_ratchet`) inject each kind of regression into a small package and
hold the repository to its baseline.

Not in it: no complexity score, no rule on the number of classes, no style check.
