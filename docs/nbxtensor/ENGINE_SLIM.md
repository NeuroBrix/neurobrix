# The engine slims: what it carries that nbxtensor makes unnecessary

Hocine's decision (2026-10-29 22:39, precised 22:45): the engine imports nbxtensor and keeps only its own job — no
duplication, no dead function, no adaptation layer, no over-engineering — and its size can be measured. nbxtensor
replaces torch for the whole project: tensor work the engine still carries (an operation, a shape or stride
computation, a conversion, a copy, a reduction, a memory placement, a launch utility, a device probe) moves INTO the
library as a generic API, and the engine calls it. The mechanism goes to the library; the policy stays in the engine
(rule 3); a piece that mixes the two is cut in two.

## The scan

`tools/engine_slim_scan.py` (standard library, ast) lists four kinds of finding in `src/neurobrix`, both branches:

| kind | what it finds |
|---|---|
| `dead` | a definition nothing outside the tests refers to (by name, attribute, import, keyword, string or data file; registered by a decorator counts as used, a call to itself does not) |
| `duplicate` | a function whose normalised body (docstring dropped, parameters and locals renamed in order) is another engine or library function's |
| `wrapper` | a function that only hands its parameters on; a class that only renames its base; an alias of an imported name; a library name other engine modules import through an engine module (a re-export); a parameter no call site passes |
| `fact` | a definition carrying the name of a public nbxtensor fact function (`dtype`, `shape`, `memory`, the device facts, the `*_cost` declarations) or of a recorded synonym, and literal dtype-size tables |

Its rules are syntactic: a finding is a question a person answers, never a deletion on the scan's word alone.

## Three fates

Every finding ends in exactly one of three fates:

1. **Deleted** — dead or duplicated: the code goes, the callers read the one that stays.
2. **Moved into nbxtensor** — tensor work the library did not yet do: a generic API with its tests (seen red), its
   documentation in `API.md` and its README example in the same commit; the engine then calls it and its copy goes.
3. **Kept in the engine** — policy (a decision of the engine: Prism, the flow, the formats, the families): one sentence
   says why, in the table below.

Open findings are listed in `docs/nbxtensor/engine_slim_list.tsv`, whose header records the engine's Python line
count and the number of open findings. `tests/unit/nbxtensor_boundary/test_the_engine_only_slims.py` holds the rules:

- a finding that is neither open in the list nor kept below fails — a new one was written: delete it, move it, or
  justify it here;
- a listed or kept finding the scan no longer finds fails — it was resolved: delete its line;
- the engine's line count and the number of open findings may only go down.

## Deleted

| when | what | lines |
|---|---|---|
| 2026-10-29 | `src/neurobrix/kernels/triton_kernels_ref/` — 1 097 files of vendored third-party reference kernels (flaggems for a dozen vendors, fla, attorch, gemlite, ...), imported by no code, excluded from both package targets | 262 132 |

## Moved into nbxtensor

| when | engine definition | nbxtensor API |
|---|---|---|

## Kept in the engine

| kind | path | symbol | why it stays |
|---|---|---|---|
