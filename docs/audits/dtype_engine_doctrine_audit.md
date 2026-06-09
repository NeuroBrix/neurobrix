# DtypeEngine doctrine — audit & harmonization (Ch7, 2026-05-19)

This document harmonizes a documented contradiction between two
runtime-debugging lessons and the live state of
`src/neurobrix/core/dtype/engine.py`. The audit was the explicit goal
of Ch7 (P-DTYPE-ENGINE-DOCTRINE-RECONCILE); the conclusion supersedes
both lessons as Resolution R1 — the live code is the correct evolved
doctrine, the lessons were time-bounded debugging captures, and a
later authoritative roadmap document (with status "✅ DONE") already
canonized the live state but the original lessons were not marked
SUPERSEDED at the time.

Related references:
- Audit-dettes document of 2026-05-16 — D7 entry — surfaced the
  apparent contradiction.
- `docs/lessons/002-dtype-engine-prism-master.md` (2026-02-02).
- `docs/lessons/004-fp16-matmul-overflow-inf-fix.md` (2026-02-03).
- `docs/roadmap/amp-mixed-precision.md` (Status: ✅ DONE,
  Updated 2026-02-08).
- `src/neurobrix/core/dtype/engine.py` (live, 532 lines).
- `validation_outputs/p_dtype_engine_doctrine_reconcile/` (R29
  forward-validation artefacts on the dtype-sensitive matrix).

## 1. The apparent contradiction (audit-dettes D7)

The 2026-05-16 audit-dettes document flagged D7 by direct line-citation:
- Lesson 002 (lines 89-95, 109) prescribes "PAS de AMP_FP32_OPS /
  `_make_fp32_wrapper` / amp_cast_inputs"; live engine.py reintroduces
  exactly those (`engine.py:79, 227, 274, 455, 470`).
- Lesson 004 (lines 147-151) marks `_make_inf_fix_wrapper` ✅ Active /
  `_make_fp32_wrapper` ❌ Disabled, with the rule "never clamp"; live
  engine.py has no `_make_inf_fix_wrapper` at all, and contains two
  clamps at `engine.py:420, 446`.

Read as eternal rules, the lessons contradict the live code on every
point. Read as time-bounded debugging captures that pointed forward
to a roadmap, the contradiction dissolves — see Sections 2 and 3.

## 2. Factual timeline (git archaeology)

| Date | Event | Reference |
|---|---|---|
| 2026-02-02 | Lesson 002 written ("DtypeEngine simplified — NO AMP"). | `docs/lessons/002-…` |
| 2026-02-03 | Lesson 004 written. **Self-flags AMP as future plan**, lines 137-143: "After the inf fix, an attempt to implement AMP (FP32_OPS wrapping) produced catastrophic results (diagonal stripes). Cause: the implementation performed an immediate fp32→fp16 downcast after every op, whereas PyTorch autocast leaves the output in fp32 and lets the next op handle the cast. Lesson: AMP requires an implementation upstream of the runtime. See `docs/roadmap/amp-mixed-precision.md` for the future plan." | `docs/lessons/004-…` |
| 2026-02-08 | AMP roadmap doc **updated with status "✅ DONE - DtypeEngine implémenté et fonctionnel"**. The doc quotes the *current* `_make_fp32_wrapper` body verbatim as "Solution Implémentée", with the critical fix of lesson 004's failed attempt: "**les ops FP32 retournent fp32 sans downcast. Les ops FP16 suivantes (matmul) castent automatiquement.**" Explicitly: "Stabilité numérique restaurée, zero-corruption éliminée." | `docs/roadmap/amp-mixed-precision.md` |
| 2026-02-25 | NBX v0.1.0-alpha (commit `efe605b`) — engine.py already at AMP-wired state. | git |
| 2026-02-27 | `f1163aa` "strategy system overhaul + dtype consolidation". | git |
| 2026-03-05 | `0f71a1d` "DtypeEngine standard AMP + hardware auto-detection" + `4cfa047` "**DtypeEngine AMP wiring + TilingEngine + audio flow + fix grid artifacts**" (same minute). The "fix grid artifacts" string in the commit title is the empirical resolution of lesson 004's "marble/diagonal stripes" failure mode under the *correct* AMP (output stays fp32, no immediate downcast). | git |
| 2026-04-10 | `75b482c` triton compiled-decode AMP. | git |
| 2026-04-19 | `e367981` 4 NBXTensor/dtype/triton fixes (current head era). | git |
| 2026-05-16 | Audit-dettes document flagged D7 — read the lessons as eternal rules, did not cross-reference the AMP roadmap status. | `docs/audit_dettes_…` |
| 2026-05-19 | Ch7 audit (this document). | here |

**Single decisive empirical fact**: `git log --all -S "_make_inf_fix_wrapper"` returns zero commits across the entire NBX repository history (any file). The function lesson 004 marked "✅ Active" **was never committed** to this repository under that name. There is no code to "re-align onto"; R2 (reverting the live code to the lesson-era prescription) would require reverse-engineering a function from a doc body, for a hypothesis the project itself documented as superseded six days later.

## 3. Live doctrine (live `engine.py`)

The live engine implements PyTorch Automatic Mixed Precision exactly,
with the V100-specific refinement that emerged after the AMP roadmap
shipped. The module docstring at the top of the file cites PyTorch's
own `aten/src/ATen/autocast_mode.h` as source of truth.

Five op classes, all data-driven (no model/family knowledge):

| Class | Behaviour | Set | Wrapper |
|---|---|---|---|
| `_to_copy` | Prism-driven dtype remap (fp16/bf16 targets → compute_dtype; fp32 targets preserved for stability — RMSNorm, RoPE, MoE routing). | n/a (op `aten::_to_copy`) | `_make_to_copy` |
| AMP FP32 | Inputs upcast to fp32 for numerical stability. **Output stays in fp32** (the fix lesson 004 identified as the correct AMP pattern); downstream FP16 ops cast back. | `AMP_FP32_OPS` (50+ ops: pow, rsqrt, layer_norm, native_group_norm, softmax/_softmax/log_softmax, sum/prod/cumsum, upsample*, polar, etc.) | `_make_fp32_wrapper` + safe-softmax variant |
| AMP FP16 | Inputs cast to compute_dtype; pure input cast, no output clamping (standard PyTorch AMP). | `AMP_FP16_OPS` (matmul-class + conv-class + RNN-cell, etc.) | `_make_lower_precision_wrapper` |
| AMP FP16 → FP32 (V100 only) | When compute_dtype == fp16 AND op in `_FP16_NEED_FP32`, upcast to fp32 (mm/bmm/addmm: inner-dim accumulation exceeds fp16 max; div: epsilon 1e-15 rounds to 0 in fp16 min ~6e-8). | `_FP16_NEED_FP32` = {mm, bmm, div, addmm} | `_make_fp32_wrapper` |
| AMP Promote | Promote to widest input dtype (PyTorch AT_FORALL_PROMOTE 100% match). | `AMP_PROMOTE_OPS` (addcdiv, addcmul, atan2, index_put, scatter_add, etc.) | `_make_promote_wrapper` |
| Constants | Convert constants whose dtype matches graph_dtype to compute_dtype; preserve constants in different dtype (intentional fp32 inv_freq in fp16 graph etc.). | n/a (graph-embedded tensors) | `convert_constant` |

Hardware split is conditional, not hardcoded:
- bf16 hardware (A100, H100, RTX 30xx/40xx): zero output protection. bf16 exponent range = fp32 → overflow impossible.
- fp16 hardware (V100): targeted fp32 upcast on `_FP16_NEED_FP32` only.

`_to_copy` clamp (lines 420 and 446): when the *graph itself* emits an `aten::_to_copy` narrowing an fp32/fp64/bf16 source into fp16, and the source actually exceeds the fp16 range (±65504), clip pre-cast to ±65504 instead of letting `inp.to(fp16)` saturate to ±Inf. Code comment cites OpenAudio DualAR pre-projection activations + certain attention bias paths as the empirical motivation.

## 4. The lesson-004 "never clamp" rule — site discrimination

Lesson 004's prohibition was specifically about clamping a **matmul output** that contains valid large in-range intermediates. The corruption mechanism it described: a value of, say, 70 000 produced by matmul that would naturally drop to 50 000 on the next op is clipped to 65 504 by an output clamp — the valid downstream value is now wrong → marble effect.

The live code does **not** do this. Matmul ops are handled by `_make_fp32_wrapper` (on V100 fp16) or `_make_lower_precision_wrapper` (elsewhere) — neither clamps output; `_make_fp32_wrapper` leaves the output in fp32 (no downcast pressure on the matmul output at all).

The clamp lives at a fundamentally different site — the `_to_copy` boundary cast emitted by the *graph* itself, narrowing fp32→fp16. There, the value is already required to become fp16, and fp16 *cannot* represent anything > 65 504 by definition. The choice is:
- Let `inp.to(fp16)` saturate to ±Inf → subsequent mm produces NaN → propagates through the rest of the graph (the exact symptom the comment cites: OpenAudio DualAR pre-projection).
- Clip pre-cast to ±65 504 → finite max-representable; subsequent mm produces a finite result.

Both options lose information (any fp32 value > 65 504 is unrepresentable in fp16 either way). The clamp chooses the option that preserves graph integrity over the NaN-cascade alternative. The lesson-004 prohibition does not apply — different site, different physics, different choice space.

## 5. Forward validation (R29 sensitive-matrix)

The doctrine being canonized must work on the dtype-sensitive models the original lessons were written about. Validation under `NBX_DTYPE_CLAMP_DIAG=1` (the env-gated diagnostic added to `engine.py` to empirically observe the clamp firing):

| Model | Family | Result | clamp diag | duration |
|---|---|---|---|---|
| openaudio-s1-mini | audio (TTS, dual_ar) | PASS (audible WAV) | 0 hits | 157.0 s |
| Sana_1600M_1024px_MultiLing | image (diffusion fp16) | PASS (coherent PNG, no marble/grid) | 0 hits | 15.5 s |
| PixArt-Sigma-XL-2-1024-MS | image (diffusion fp16) | PASS (coherent PNG, no marble/grid) | 0 hits | 33.6 s |

Empirical findings:
1. The catastrophic failure modes lessons 002+004 described (marble effect, diagonal stripes, NaN propagation, black images) **do not reproduce** under the live AMP+clamp doctrine on any of the three sensitive models. This is the discriminator R1 requires.
2. `NBX_DTYPE_CLAMP_DIAG=1` fired **zero times** across all three runs, including the OpenAudio model the clamp comment explicitly cites. The `_to_copy` clamp is a **tail-path defensive guard** for activation patterns specific model content can produce; standard prompts do not exercise the protected paths. This empirically confirms point 4 above — the clamp is dormant under typical inference and only activates on the narrow class of fp32→fp16 narrowings whose source actually overflows.

R29 artefacts: `validation_outputs/p_dtype_engine_doctrine_reconcile/{openaudio-s1-mini,Sana_1600M_1024px_MultiLing,PixArt-Sigma-XL-2-1024-MS}/` (output.{wav,png}, run.log, prompt.txt, stats.json, verdict.md, INDEX.md).

## 6. Counterfactual scope-out (lesson-era state)

The mandate asked for experimental discrimination by running both doctrines. The counterfactual (lesson-era state: no AMP, only `_make_inf_fix_wrapper` on matmul, no `_to_copy` clamp) is **deliberately not run** because:
1. `_make_inf_fix_wrapper` has no presence in NBX repo history (proven by `git log --all -S`). The lesson described a fix that never landed under that name.
2. Reconstructing it from the lesson body — removing AMP_FP32_OPS, AMP_FP16_OPS, all wrappers, `_to_copy` clamp, then re-adding a never-shipped function — is days of code surgery for a hypothesis the project itself documented as superseded six days later (Feb 8 AMP roadmap ✅ DONE).
3. R23 doctrine (live-prod-is-king: doctrine must serve the prod path, not destabilize it) explicitly forbids destabilizing the live code to test a superseded hypothesis.
4. The forward validation in Section 5 is the discriminator R1 demands: the doctrine being canonized must work today on the sensitive matrix. It does.

Per the Ch6 advisor pattern: this scope-out is surfaced explicitly so Hocine can override if desired. The substitution is bounded and rationale-backed, not papered over.

## 7. State of the art consulted (R16)

The live `engine.py` already cites the most authoritative source for this domain: PyTorch's `aten/src/ATen/autocast_mode.h` (the `AT_FORALL_FP32`, `AT_FORALL_FP32_SET_OPT_DTYPE`, `AT_FORALL_LOWER_PRECISION_FP`, `AT_FORALL_PROMOTE` macros). The file's audit-status comment claims 100% match on AT_FORALL_FP32 (50+ ops), 100% on AT_FORALL_FP32_SET_OPT_DTYPE (9 ops), 97% on AT_FORALL_LOWER_PRECISION_FP (30/32 ops), 100% on AT_FORALL_PROMOTE (11 ops), with documented deviations and extensions.

Adjacent industry references for context:
- **PyTorch `torch.autocast`** (the source of truth above). Wraps ops by class; output of fp32 ops stays fp32, downstream fp16 ops cast back — exactly the pattern lesson 004's failed attempt got wrong (immediate downcast) and the AMP roadmap fixed.
- **NVIDIA Transformer Engine**: introduces fp8 with per-tensor scaling; orthogonal to the V100 fp16 question Ch7 addresses, but confirms the principle that mixed-precision frameworks classify ops, not models.
- **DeepSpeed mixed-precision**: uses loss-scaling for training; not directly applicable to inference but confirms the matmul-overflow-on-fp16 phenomenon as a known industry pattern (their solution: fp32 master weights — comparable in spirit to AMP_FP32_OPS fp32 upcast for sensitive ops).
- **JAX bfloat16**: bf16 throughout with no protection; matches the live engine's bf16-hardware branch (zero output protection for bf16 hardware because bf16 exponent range = fp32).

R16 considered. The live doctrine matches the industry-canonical mixed-precision pattern; no external algorithm needs adoption. The narrow V100-specific refinement (`_FP16_NEED_FP32` = {mm, bmm, div, addmm} upcast) addresses a hardware-class numeric problem (fp16 inner-dim accumulation overflow) that PyTorch autocast does not explicitly handle because most production PyTorch deployments are A100/H100 with bf16.

## 8. Resolution chosen: R1 — supersede both lessons

Lessons 002 and 004 are time-bounded debugging captures from 2026-02-02 / 2026-02-03 that were superseded by the AMP roadmap implementation (status ✅ DONE 2026-02-08) and subsequent commits (`0f71a1d`, `4cfa047`, `75b482c`, `e367981`). They should be marked SUPERSEDED with a pointer to the live doctrine. The R3 (nuance) sub-argument about the `_to_copy` clamp site discrimination is folded into the SUPERSEDED notes (Section 4 of this document is its canonical statement).

R2 (reverting live code to lesson-era state) is rejected because: (a) the prescribed `_make_inf_fix_wrapper` function never existed in the repo; (b) R23 forbids destabilizing the live prod code for a superseded hypothesis; (c) the forward validation in Section 5 empirically confirms the live doctrine works on the sensitive matrix the lessons were originally written about.

## 9. Latent observations (D10 — not in scope for Ch7)

- **Lesson-vs-roadmap hygiene meta-pattern**: the audit-dettes D7 flag came from reading the lessons as eternal rules without cross-referencing the AMP roadmap doc that already marked the matter resolved. A grep of `docs/` for `Status:.*DONE` and `SUPERSEDED` markers may surface 2-3 other lesson+roadmap pairs in the same state. Not fixed in Ch7; named for a dedicated documentation-hygiene chantier (e.g. Ch10 P-VERDICTS-HYGIENE).
- **`_make_inf_fix_wrapper` as documentation artefact**: lesson 004 describes a function that never shipped. Either the lesson was a write-up of an investigation that landed via a different design (the AMP roadmap), or it documents a prototype that was abandoned. The historical question is not actionable; the present consequence is that the lesson text should be marked SUPERSEDED, which Ch7 does.
- **OpenAudio DualAR motivating site empirically dormant**: under the standard validation prompt the `_to_copy` clamp's cited motivating model did not exercise the protected path. The clamp is correct as documented defensive infrastructure; an opportunistic future hygiene step might instrument production runs (default-off counter) to confirm the path is exercised under real workloads — not in Ch7 scope.

## 10. Hocine validation: TODO

R29 artefacts under `validation_outputs/p_dtype_engine_doctrine_reconcile/` provide audible WAV + visually-inspectable PNGs for human sign-off.
