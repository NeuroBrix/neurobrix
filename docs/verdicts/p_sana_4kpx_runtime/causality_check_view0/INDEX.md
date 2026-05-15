# Causality check view::0 dtype — 2×2 matrix
## P-SANA-4KPX-RUNTIME 2026-05-09 — Hocine's causality directive

R29 inspectable artefact for: "is the dtype gap at view::0/add::0
the CAUSAL root or merely correlated with the green-texture bug?"

## Methodology

`/tmp/causality_view0.py` — register `op_uid_interceptor` on
`aten.view::0` that does the natural reshape, then casts the
result to an override dtype. No NeuroBrix code modification; pure
runtime override. Then run VAE-isolation (saved latent
`vae_isolation_input.pt`) end-to-end, save final PNG.

Symmetric on both modes (R30):
- triton_sequential default: view::0 stays fp32 (graph default); test forced fp16.
- sequential default: view::0 cast to fp16 by runtime (DtypeEngine implicit); test forced fp32.

## Results — the 2×2 matrix (R29 visual inspection)

| cell | view::0 dtype (natural→override) | add::0 (a/b/out) | PNG verdict |
|---|---|---|---|
| **seq_default** | fp16 → fp16 | fp16/fp16/fp16 | **RED APPLE** (oracle) |
| **seq_force_view0_fp32** | fp16 → **fp32** | fp16/fp32/fp32 | **RED APPLE** (still coherent!) |
| **tri_default** | fp32 → fp32 | fp16/fp32/fp32 | **GREEN TEXTURE** (current bug) |
| **tri_force_view0_fp16** | fp32 → **fp16** | fp16/fp16/fp16 | **GREEN TEXTURE** (still garbage!) |

PNGs in this directory:
- `seq_default.png` — red apple (oracle)
- `seq_force_view0_fp32.png` — **red apple even with view::0=fp32 + add::0 output fp32**
- `tri_default.png` — green texture (current bug)
- `tri_force_view0_fp16.png` — **green texture even with view::0=fp16 + add::0 output fp16**

## Verdict — dtype at view::0/add::0 is CORRELATED but NOT CAUSAL

**Sequential mode produces coherent PNG regardless of view::0 dtype.**
Forcing view::0 to fp32 in seq (so add::0 outputs fp32, matching the
exact dtype profile of tri mode) — STILL produces a red apple. The
seq path has some downstream mechanism that maintains semantic
correctness even with fp32 propagation through add::0.

**Triton mode produces garbage regardless of view::0 dtype.** Forcing
view::0 to fp16 in tri (so add::0 inputs/output match seq's profile
fp16/fp16/fp16) — STILL produces green texture. The tri path's bug
does not lift simply by aligning view::0/add::0 dtype to seq's pattern.

The dtype mismatch identified at fc9d754 (in `first_divergent_op/walk_table.md`)
was the FIRST op where seq and tri DIVERGE in trace order — but it
is **not the causal root**. The bug source is downstream of add::0.

## What this rules out

- **Fix at NBX add wrapper dtype narrowing** (option 1 in walk_table.md
  "Awaiting Hocine arbitrage"): would not fix the green texture.
  seq_force_view0_fp32 demonstrates that fp32 propagation through
  add::0 is not the bug.
- **Fix at TritonDtypeEngine metadata-chain narrowing** (option 2):
  same logic. Would just align dtype patterns, but the matrix shows
  alignment doesn't lift the bug.
- **Fix at forge-level _to_copy injection** (option 3): same.

## What this implies for the next investigation

The bug is at some op DOWNSTREAM of add::0 where:
- Seq path recovers semantic correctness (regardless of fp32 vs fp16
  precision propagation upstream).
- Tri path does not recover.

Critical observation from this 2×2:
- seq_force_view0_fp32 has add::0 OUTPUT in **fp32** (same as tri_default)
- seq_force_view0_fp32 → red apple
- tri_default → green texture

So given the SAME add::0 fp32 output entering the rest of the chain,
seq produces red apple, tri produces garbage. The divergence must
be somewhere AFTER add::0 in the graph, in an op whose seq vs tri
implementations differ in a way that ONLY matters when noise/precision
patterns reach a certain threshold.

Candidates to walk next (compute ops between add::0 and conv::36
where the catastrophe was observed):
- mm/bmm chain (op 9, 13, 17, 40, 102, 149, 246, 297) — full-tensor
  diff already showed 70-99% frac>1e-3, but with mixed dtype origins
  in the input cascade.
- conv::1, conv::2, conv::3, conv::14, conv::29 — all show
  divergence in walk; need to identify which has SHAPE-SPECIFIC
  algorithmic asymmetry beyond "input precision differs".

Recommendation: NEW walk pass with view::0 forced to fp32 in BOTH
modes (eliminating the dtype variable) so any remaining divergence
can be attributed to algorithmic differences only.

## Awaiting Hocine direction

Per Hocine's directive: "Si elle te donne un signal ambigu, on
aura factuellement identifié que la cascade dtype add::0 n'est
pas la racine et on walkera plus loin. Pas de fix avant ce verdict."

The signal is UNAMBIGUOUS: dtype gap is correlated, not causal.
Walking further required. Awaiting your call on:
- Re-walk with both modes' view::0 forced fp32 (eliminate dtype variable)?
- Different bisection strategy (binary search on op_idx with stricter
  causality test)?
- Other?
