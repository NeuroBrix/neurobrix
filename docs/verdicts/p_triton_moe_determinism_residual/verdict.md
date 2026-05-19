# P-TRITON-MOE-DETERMINISM-RESIDUAL — verdict (diagnosis airtight + fix proven; scope escalated, 2026-05-19)

Branch `p-triton-moe-determinism-residual` from `78d6b1d`. Goal:
isolate the residual Qwen3-30B::triton forward-pass non-determinism
that survived the Ch4 `index_add` deterministic gather (`feb6103`).

**Outcome: root cause airtight (flash-attention non-determinism on
Volta), deterministic fix counterfactually proven (`_math_attention`).
The remaining decision is the SCOPE of the routing change (cross-family
/ hardware / memory blast radius) — escalated to Hocine.**

---

## Section 1 — Method (P-SANA op-by-op differential, §5.8)

A gated `NBX_OP_FINGERPRINT` diagnostic was added to
`triton/sequence.py` (single- AND multi-device store-output hooks;
sha256 per op output; `NBX_OP_FINGERPRINT_CAP` byte-cap and
`NBX_OP_FINGERPRINT_MAX` cutoff for tractable 115k-op model traces).
Same retained-diagnostic class as `NBX_DUMP_TIDS` /
`NBX_DISABLE_AUTOTUNE`.

Cheap discriminant first (§5.8): **TinyLlama-1.1B::triton is fully
deterministic** (3 runs byte-identical, "The capital of France is
Paris."). TinyLlama is dense + fits one GPU (single-device);
Qwen3-30B-A3B is MoE + 30B (zero3 CPU-offload, `_run_multi_device`).
So the cause is in the MoE/zero3/large-model path, not the core
ops shared with TinyLlama (mm/rms_norm/SDPA are deterministic at
TinyLlama's shapes).

## Section 2 — Elimination chain (every claim fingerprint-sourced)

- **autotune-mm ruled out.** Counterfactual `NBX_DISABLE_AUTOTUNE=1`
  (pins matmul-class kernels to one config): Qwen3 still 3 distinct
  outputs (`...Hmm, this` / `...and then it` / `...and left it`).
  `@triton.autotune` config-selection variance is NOT the cause.
- **o-buffer padding artifact ruled out.** Qwen3 SDPA `o =
  NBXTensor.empty_like(q)` is shaped [1,32,15,128]; headdim=128 →
  `BLOCK_HEADDIM=128` → `EVEN_HEADDIM=True`; the kernel's final
  `tl.store(out_ptrs, acc_o, mask=offs_m<seqlen_q)` writes every
  15×128 element per head — no uninitialised region.
- **cached-bias non-determinism ruled out.** `_get_zero_bias` /
  `_get_causal_bias` build via `NBXTensor.zeros` + deterministic
  ops and cache by `(device,seqlen_q,seqlen_k,dtype)`.
- **all pre-SDPA ops deterministic.** Full-hash (`CAP=0`),
  full-coverage (`MAX=80`), 3 runs: positions 0–74 byte-identical
  across all runs; first divergence at **pos 75
  `aten.scaled_dot_product_attention::0`** (shape [1,32,15,128]
  fp32, sha r1=32d52f r2=c4759b r3=4e742d). pos 73–74
  (`aten.expand`, the GQA K/V expand feeding SDPA) are
  byte-identical → SDPA's Q/K/V inputs are airtight identical.

Conclusion by elimination: **the flash-attention Triton kernel
itself is non-deterministic for Qwen3's shape** (deterministic
inputs, deterministic bias, fully-written output buffer → 3 distinct
outputs). The divergence cascades: SDPA → router projection mm →
router softmax → non-deterministic top-k expert selection →
divergent MoE op-trace (op-order divergence at pos ~90) → divergent
logits → greedy-argmax flip at the first low-margin token.

## Section 3 — Root cause + counterfactual fix proof

The SDPA wrapper already documents the mechanism
(`wrappers.py` ~5247–5270): the Dao-AILab flash Triton kernel's
masked-load path "is non-deterministic on Volta SIMT", with an
existing deterministic mitigation `_math_attention`
(`scores=Q@K^T → softmax → @V`, plain bmm+softmax, no online-softmax
accumulation / no MMA reorder across K-tiles) — but it is gated only
by `if not _is_power_of_2(headdim)`. **Qwen3's headdim=128 is a power
of two, so the guard does NOT route it to the deterministic path, yet
the flash kernel is still non-deterministic for this GQA/fp32 Volta
shape.** The `_is_power_of_2` heuristic is incomplete.

**Counterfactual proof**: a gated `NBX_FORCE_MATH_ATTENTION=1`
diagnostic forces the deterministic path regardless of headdim.
Qwen3-30B::triton, 3 runs, greedy temp=0, max-tokens 16:
- ma1/ma2/ma3 = `Okay, the user asked, "The capital of France is" and left it`
- **3 runs BYTE-IDENTICAL.**

→ flash-attention is the residual root cause; `_math_attention` is a
proven deterministic fix.

## Section 4 — State of the art consulted (R16)

- Triton `@triton.autotune` + `cache_results` documented
  non-deterministic across runs (Triton #9368/#5339/#781) — relevant
  but ruled out here by counterfactual.
- Dao-AILab flash-attention: FlashAttention-2+ requires SM80+ (V100
  unsupported); the original Triton `flash_attn_triton.py` documents
  "caveats about race conditions on non-64/128 head dimensions"
  (https://github.com/Dao-AILab/flash-attention,
  issue #1760 V100 support). Confirms V100/Volta + Triton-flash is a
  known-fragile combination; our wrapper comment cites the same
  caveat. No autotune-free / Volta-deterministic upstream flash
  variant exists.

## Section 5 — Scope decision (ESCALATED)

The fix is proven; the open question is the **routing-change scope**,
which has cross-family / hardware / memory blast radius:

- `_math_attention` materialises a `[B*H, T_q, T_k]` scores tensor:
  negligible for LLM small-T, but ~1 GB fp32 for PixArt self-attn
  and prohibitive for Sana 4Kpx (documented in the wrapper). So a
  blanket "always math attention" would OOM/regress image diffusion.
- The flash path is deterministic on A100/H100 (SM80+); only Volta
  SIMT is affected (R23 — must not penalise non-Volta).
- Image-diffusion numerics are tuned on the flash path; changing
  their attention on Volta is a regression risk for tuned models.

Candidate scopes (recommendation: **B**):
- **(A)** Volta + LLM/autoregressive family only → `_math_attention`
  (small-T, memory-safe, zero change to image / non-Volta). Bounded,
  R34 family-aware via taxonomy, R23 hardware-gated, no regression
  by construction.
- **(B)** Volta + memory-affordable regime (scores tensor within a
  budget) → `_math_attention`, else flash. Model-agnostic (R34, no
  family logic), R23 hardware-gated, no OOM regression. Slightly more
  wiring (scores-size estimate) but the most principled.
- **(C)** Volta blanket → `_math_attention` (rejected: OOMs image
  diffusion large-T).
- **(D)** Make flash deterministic on Volta (rejected here: deep
  Triton-codegen chantier, upstream-documented race; out of scope —
  becomes a named follow-up if ever needed).

This is a `dette latente → arbitrage scope` decision per the global
mandate's escalation criteria (cross-family blast radius + memory
tradeoff + R23/R30/R34). Diagnosis and fix are not in question; the
scope of the routing predicate is.

## Section 6 — Commits

- (diagnostic instrumentation) gated `NBX_OP_FINGERPRINT` in
  `triton/sequence.py`, gated `NBX_FORCE_MATH_ATTENTION` in
  `kernels/wrappers.py` — retained-diagnostic class, default-off,
  zero runtime impact. SHA reported in the closure note.
- The functional fix commit is deferred to the scope decision
  (Section 5).

## Section 7

Hocine validation: TODO (scope decision A/B/C/D required before the
functional fix commit + tag).
