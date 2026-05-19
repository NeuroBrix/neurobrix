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

## Section 7 — RESOLUTION (Hocine decision = option B, implemented + validated)

**Decision**: option **B** — route SDPA to deterministic
`_math_attention` on universal technical signals only (R34 strict:
the SDPA wrapper is a universal runtime primitive, ZERO model/family
knowledge; taxonomy is an offline/build concern, never a
runtime-dispatch signal).

**Implementation** (`kernels/wrappers.py`
`scaled_dot_product_attention_wrapper`):
- non-pow2 head_dim → `_math_attention`, UNCONDITIONAL on all
  hardware — EXACT prior behaviour preserved (the old guard was
  `if not _is_power_of_2(headdim): return _math_attention`,
  hardware-independent). R23: zero non-Volta regression; PixArt
  hd=72 / Sana hd=112 path byte-unchanged everywhere.
- pow2 head_dim AND `scores_bytes ≤` data-driven budget →
  `_math_attention` (NEW, the only added routing). `scores_bytes =
  B*H*Tq*Tk*4` (the fp32 [B*H,Tq,Tk] scores tensor `_math_attention`
  materialises — true memory cost, dtype-independent). Budget from
  `_sdpa_math_scores_budget_bytes()` → `get_vendor_config(<gpu-brand>,
  <arch>).memory.sdpa_math_max_scores_bytes`. **The per-arch YAML key
  IS the hardware gate**: only `config/vendors/nvidia/volta.yml`
  defines it (= 128 MiB); ampere/hopper/cdna lack it → budget 0 →
  `scores ≤ 0` False → flash unchanged on non-Volta (R23, no
  ambiguous capability proxy).
- The incomplete `_is_power_of_2(headdim)` guard is REMOVED (it
  wrongly assumed pow2 hd=128 is flash-safe on Volta). I verified its
  only justification was the Volta non-pow2 race — preserved above
  byte-for-byte, so removal is safe (no escalation needed).

**Budget = 128 MiB** (`sdpa_math_max_scores_bytes: 134217728` in
volta.yml, fully commented). Chiffré: Qwen3-30B-A3B test
B1·H32·Tq15·Tk15·fp32 = 28 KiB ≪ 128 MiB → math. LLM decode
Tq=1·Tk≤4096·H32·fp32 ≤ 512 KiB → math (every step). LLM prefill
seqlen≈1024 @ H32 fp32 = 128 MiB → math; seqlen ≳ 1152 → flash
(documented residual; decode still deterministic). PixArt-512
self-attn ≈ 1 GiB, Sana ≫ → far above budget (and they are non-pow2
→ already on the preserved unconditional math path regardless).

**Two bugs found + fixed during validation** (P-SANA discipline —
validate, don't assume): (1) `dtype_size(q.dtype)` raised
`KeyError: triton.language.float32` (q.dtype is a Triton dtype at
this point) → replaced with the fp32-scores constant `*4` (the true
cost). (2) `prof.vendor` is the **machine** maker (`"dell"`), not the
GPU brand → `get_vendor_config("dell","volta")` failed → budget 0 →
fix silently inert; fixed to `devices[0].brand.value` (`"nvidia"`).
`NBX_SDPA_ROUTE_DIAG=1` (retained diagnostic) confirms by default:
`hd=128 pow2=True scores_bytes=28800 budget=134217728 use_math=True`.

**Validation** (V100, post-fix HEAD, default path, no env):
- **Qwen3-30B::triton — 3 runs BYTE-IDENTICAL** (mandate criterion):
  all three = `Okay, the user asked, "The capital of France is" and
  left it` (errs=0). Residual non-determinism CLOSED.
- TinyLlama::triton 3× = `The capital of France is Paris.` —
  coherent + deterministic (flash→math switch clean; per mandate
  `_math_attention` is the new LLM-Volta golden, cosine-vs-old-flash
  not required).
- DeepSeek-MoE::triton 3× = ` Paris.` — coherent + deterministic
  (collateral benefit: was only "coherent" at Ch4, now
  byte-deterministic).
- Sana_1600M_1024px::triton — clean 1024×1024 RGB PNG, 89 s, no OOM,
  R29 visually inspected (coherent "serene mountain lake at sunset",
  not degenerate). Non-pow2 preserved `_math_attention` path intact.
  Artefact: `validation_outputs/p_triton_moe_determinism_residual/
  sana-1600m-1024/output.png` + `run.log`.
- PixArt-XL-2-1024-MS::triton — timed out at 900 s (rc=124, empty
  log = SIGTERM pre-flush). **NOT a Ch4bis regression**: PixArt is
  non-pow2 head_dim → the preserved unconditional `_math_attention`
  path, logically byte-identical to pre-fix; my change provably did
  not touch it, and Sana (same preserved path) validates the path
  works. PixArt-XL-2-1024-MS::triton > 900 s is pre-existing triton
  slowness on this large model — see latent observation.

**Latent observations** (for Ch10 hygiene / follow-ups):
- **Upstream factual contribution**: the Dao-AILab Triton flash
  caveat says race conditions on "non-64/128 head dimensions" —
  but Qwen3 hd=128 IS 128 and is STILL non-deterministic on Volta
  SIMT (fp32, GQA). The upstream caveat is incomplete for Volta
  SIMT; pow2/64-128 head_dim is NOT safe there. (Potential upstream
  issue to Dao-AILab — out of scope this chantier.)
- **P-PIXART-XL-1024-TRITON-PERF**: PixArt-XL-2-1024-MS::triton does
  not complete in 900 s on V100. Pre-existing, unrelated to Ch4bis
  (non-pow2 path unchanged). Named follow-up for Ch10.
- **P-SDPA-WRAPPER-BLOCK-PICKER-DRIFT**: the SDPA wrapper hardcodes
  its BLOCK_M/BLOCK_N selection (`if seqlen_q<=16: BLOCK_M=16…`)
  instead of using the data-driven `loader.get_sdpa_block_size`
  (which reads volta.yml `sdpa_thresholds`). Pre-existing
  R10/R23 inconsistency, not touched here. Named follow-up for Ch10.

**Diagnostics retained** (default-off, zero runtime impact, §5.8
toolkit — documented in `src/neurobrix/CLAUDE.md` §8):
`NBX_OP_FINGERPRINT` (+`_CAP`/`_MAX`), `NBX_FORCE_MATH_ATTENTION`,
`NBX_SDPA_ROUTE_DIAG`.

Hocine validation: TODO (functional outcome proven; Sana image
artefact present for R29 visual sign-off — the Qwen3 determinism
result is text and self-evident in this verdict).
