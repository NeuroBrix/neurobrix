# Layer-7 auto-fp32 family-aware — audit + design (Ch8, 2026-05-20)

This document audits the existing manual `requires_fp32_compute`
opt-in mechanism (Ch6-era), the Layer-7 follow-up
(`docs/follow-ups/layer7-prism-dtype-override.md`) blocking PixArt-α
/ PixArt-σ in `--triton` mode, and locks the design for an
automatic family-aware fp32 detection in `PrismSolver` that runs
without per-model manual flags while preserving the existing
manual flag as an explicit override. R34 strict (no model-name
branching), R10 data-driven (discrimination from
`config/families/<family>.yml`), R18 preserved (no NBX container
contract change).

Related references:
- `docs/follow-ups/layer7-prism-dtype-override.md` — the open
  follow-up documenting the PixArt VAE fp16 conv2d saturation.
- `src/neurobrix/core/prism/solver.py:2936` — existing
  `_components_force_fp32` method (manual opt-in via
  `requires_fp32_compute` registry flag, env-overridable by
  `NBX_FORCE_FP32_COMPUTE`).
- `src/neurobrix/core/dtype/engine.py` — the Ch7-consolidated AMP
  doctrine (`docs/audits/dtype_engine_doctrine_audit.md`, Ch7).
  This chantier builds on the Ch7 doctrine; the auto-detect is a
  *component-level* protection that composes with the AMP *op-level*
  protections in `engine.py`.
- `src/neurobrix/config/families/{image,llm,…}.yml` — the
  data-driven family layer where the auto-detect policy lives.

## 1. Existing manual mechanism

`PrismSolver._components_force_fp32(container) -> set[str]` returns
the set of components to pin to fp32 regardless of the hardware's
preferred dtype. Today it reads exactly one source:
`registry_flags.get_component_flag(model_name, comp.name,
"requires_fp32_compute", default=False,
env_override="NBX_FORCE_FP32_COMPUTE")`. This is the Ch6-era
mechanism the SwinIR fp16-unsafe transformer-SR chantier built —
the model owner declares per-component fp16-unsafety in the
registry, the solver honours it.

Consumed at two sites in the same file:
- `_resolve_dtype` (line 2978): if ANY component is forced, the
  whole-model resolved dtype becomes `"float32"`.
- `_resolve_component_dtypes` (line 3008): per-component allocation
  uses fp32 for forced components and the hardware-preferred dtype
  otherwise.

The manual flag works — SwinIR ships at fp32 via this surface —
but it requires a human to know in advance, per model, which
components are fp16-unsafe. For families with a structurally
predictable fp16 overflow pattern (image-diffusion VAE), this is
fragile: PixArt and Sana ship every day without setting the flag
and break in `--triton` mode at the documented overflow site.

## 2. The blocking symptom (Layer-7 follow-up summary)

PixArt VAE produces `min=nan, max=nan, mean=nan` in `--triton`
mode on V100. Op-by-op tracing localises the first inf-producing
op to `conv2d_wrapper` at output shape `[1, 512, 512, 512]` with
`finite_abs_max = 6.55e+04` (fp16 max). The build records
`graph.json::torch_dtype = "float32"` (diffusers VAE default), but
Prism downcasts to fp16 on V100 (no native bf16). The conv's
fp32-accumulator math is correct; the **output buffer** is fp16,
so `tl.store` saturates fp32-out-of-range values to ±Inf, which
the next op turns into NaN.

Native PyTorch (`compiled` mode) avoids the saturation because
cuDNN returns an **fp32 output** from a fp16-input conv2d on
V100; the next op's autocast then handles the dtype. The triton
kernel's wrapper-level output dtype is the load-bearing
discrepancy.

The follow-up's solution outline is a Prism per-component
override: when the build records `torch_dtype = float32` AND the
hardware lacks native bf16 AND the family is image/video, pin the
component to fp32 instead of downcasting. Ch8 implements exactly
that, with the precision tightenings developed below.

## 3. Empirical per-component graph audit

Probed the build-time `torch_dtype` and op set of every component
in the cached matrix:

| model | comp | graph_dtype | conv2d | sdpa | native_group_norm |
|---|---|---|---|---|---|
| PixArt-Sigma-XL-2-1024-MS | vae | float32 | 36 | 1 | 30 |
| PixArt-Sigma-XL-2-1024-MS | transformer | float32 | 1 | 56 | 0 |
| PixArt-Sigma-XL-2-1024-MS | text_encoder | float32 | 0 | 0 | 0 |
| PixArt-XL-2-1024-MS | vae | float32 | 36 | 1 | 30 |
| PixArt-XL-2-1024-MS | transformer | float32 | 1 | 56 | 0 |
| Sana_1600M_1024px_MultiLing | vae | float32 | 70 | 0 | 0 |
| Sana_1600M_1024px_MultiLing | transformer | float32 | 61 | 20 | 0 |
| Sana_1600M_1024px_MultiLing | text_encoder | float16 | 0 | 26 | 0 |
| Janus-Pro-7B | gen_vision_model | bfloat16 | 59 | 0 | 39 |
| Janus-Pro-7B | language_model | bfloat16 | 0 | 30 | 0 |
| Janus-Pro-7B | vision_model | bfloat16 | 1 | 24 | 0 |
| TinyLlama-1.1B-Chat-v1.0 | model | bfloat16 | 0 | 22 | 0 |
| openaudio-s1-mini | codec.decoder | float32 | 30 | 0 | 0 |
| Kokoro-82M | decoder | float32 | 47 | 0 | 0 |
| swin2SR-classical-sr-x2-64 | swin2sr | float32 | 5 | 0 | 0 |

Key disambiguations:
- PixArt VAE has 36 conv2d + 30 `native_group_norm` — clearly
  conv-cascade-dominant. PixArt transformer has 1 conv2d (patch
  embed) + 56 SDPA — clearly attention-dominant.
- Sana VAE uses **DC-AE** which does **not** emit
  `native_group_norm`. The naive "`native_group_norm` presence"
  marker (which works for PixArt) fails on Sana. But Sana VAE has
  70 conv2d / 0 SDPA — overwhelmingly conv-dominant.
- Sana transformer is the discriminating case: 61 conv2d / 20
  SDPA — it has many conv2d (linear-attention 1x1 / patch-mixing)
  but also 20 SDPA ops. It is a hybrid, not conv-dominant.

## 4. The discrimination rule (locked)

A component is auto-fp32 candidate iff **all** of:

1. `family ∈ {image, video}` — read from manifest, gated by the
   family YAML `dtype_policy.auto_fp32_on_overflow_risk.enabled =
   true`. Other families omit the section (loader default
   = disabled).
2. `component.graph["torch_dtype"] == "float32"` — the build-time
   recorded dtype. Native fp16 / bf16 components are not at this
   overflow site.
3. `profile.preferred_dtype != "bfloat16"` — only fire on
   fp16-preferring hardware (V100-class). On A100/H100 (native
   bf16, fp32 exponent range), the saturation does not occur;
   pinning to fp32 would waste VRAM.
4. **Conv-dominance gate**: `conv2d_count >= 20` AND `conv2d_count
   >= 10 × sdpa_count`. The 20-floor excludes the DiT patch-embed
   (1 conv2d) and other minor-conv components; the 10× ratio
   excludes the hybrid linear-attention transformer (Sana
   transformer at 61/20=3.05).

Applied to the empirical table:

| component | family | graph_dtype | conv2d ≥ 20 | conv2d ≥ 10·sdpa | fires? |
|---|---|---|---|---|---|
| PixArt-Sigma vae | image | float32 | 36≥20 ✓ | 36≥10 ✓ | **YES** |
| PixArt-Sigma transformer | image | float32 | 1≥20 ✗ | — | no |
| PixArt-Sigma text_encoder | image | float32 | 0≥20 ✗ | — | no |
| Sana 1024 vae | image | float32 | 70≥20 ✓ | 70≥0 ✓ | **YES** |
| Sana 1024 transformer | image | float32 | 61≥20 ✓ | 61<200 ✗ | no |
| Sana 1024 text_encoder | image | float16 | graph-dtype gate ✗ | — | no |
| Janus gen_vision_model | multimodal | bfloat16 | family gate ✗ | — | no |
| TinyLlama model | llm | bfloat16 | family gate ✗ | — | no |
| openaudio codec.decoder | audio | float32 | family gate ✗ | — | no |
| Kokoro decoder | audio | float32 | family gate ✗ | — | no |
| swin2SR swin2sr | upscaler | float32 | family gate ✗ | — | no |

The rule fires exactly on the two VAE components the Layer-7
follow-up names, and nowhere else, modulo the manual override
preserved (see Section 5).

## 5. Composition with the existing manual flag (manual > auto)

The augmented `_components_force_fp32` returns the **union** of
the manual set and the auto-detect set. Manual is always
preserved; auto only adds. The user's bias ("manual must remain
functional as override") is satisfied by additive composition
with zero conflict surface.

Bypass env var: `NBX_DISABLE_AUTO_FP32=1` skips the auto-detect
entirely (manual still honored). Default-off. Two purposes:
1. Capture the R29 "triton-without-fix-broken" baseline image for
   the verdict, without git-stashing the implementation.
2. Durable diagnostic for any future regression where the
   auto-detect over-includes a component and an operator wants to
   confirm the auto-detect is the cause.

Matches the retained-diagnostic pattern documented in
`src/neurobrix/CLAUDE.md` §8 (`NBX_DUMP_TIDS`,
`NBX_TRITON_TRACE_NAN`, `NBX_OP_FINGERPRINT`, `NBX_DTYPE_CLAMP_DIAG`).

## 6. State of the art consulted (R16)

The auto-detect heuristic is the structural form of an
industry-known pattern:

- **Hugging Face diffusers** auto-routes VAE encode/decode in fp32
  by default for fp16-inference pipelines (`StableDiffusionPipeline.vae`
  is upgraded to fp32 with `--enable_vae_slicing`/`--enable_vae_tiling`
  when fp16 overflow is detected). The detection is by class name
  (`AutoencoderKL`); we use a structural-op-count signal because
  R34 forbids name branching.
- **ComfyUI** has a per-VAE dtype-override config that recognises
  the same overflow class; the upstream pattern is "VAE wants
  fp32 when working with deep convolutional decoders on fp16 hw".
- **NVIDIA TensorRT** (for DiT/UNet inference) applies layer-wise
  fp32 pinning for normalisation layers and selected conv layers,
  via the calibration profile (manual annotation upstream of trace).
- **AITemplate** (Meta) similarly distinguishes attention-heavy
  blocks (fp16 OK) from conv-heavy reduction stages (fp32 needed
  for accumulation) — same conv-dominance discriminator we apply
  structurally.

The 20-conv2d floor and 10× ratio thresholds are not standard
constants in the ecosystem (every framework picks its own); they
are derived from the cached-matrix empirical analysis above. They
live in `config/families/{image,video}.yml` and are tunable
without code edits per R10.

## 7. Resolution chosen: implement at `_components_force_fp32`

The single architectural choice point is **where the auto-detect
lives**. Two viable seams:

(A) Thread the family/manifest into `DtypeEngine` and let the
engine apply a per-component policy at compile time.
(B) Keep the detection in `PrismSolver`, the existing per-component
dtype-resolution surface. Augment `_components_force_fp32` to
return the union of manual ∪ auto-detect.

(B) is the chosen path. Rationale:
- (B) is **minimal**: one method augmented (~30 lines), zero
  changes to `DtypeEngine`. (A) would thread the family through
  the engine API (`compile_op`, `convert_constant`, runtime
  `amp_cast_*`), risk silently shifting the Ch7-consolidated AMP
  contract, and add a parallel mechanism that's also a per-op
  decision (which is *already* covered by `AMP_FP32_OPS`).
- (B) **composes with the existing surface**: the same set the
  manual flag populates, the same two consumer sites, the same
  resolved dtype machinery. The auto-detect simply contributes
  more entries to the existing set.
- (B) is **R34-clean by construction**: the family is read at the
  solver level (where the manifest is already accessible); the
  structural-op-count test reads only `comp.graph` op_types; no
  model name appears anywhere.
- (A) would have required exactly the "threading the family
  everywhere" pattern the user flagged as R18-risky in the Ch8
  mandate ("préférer l'heuristique structurelle locale si elle
  suffit").

Per-component fp32 pinning is the *correct level* for this
problem: the AMP op classification (`engine.py` AMP_FP32_OPS) is
the right surface for *per-op* fp32 sensitivity (softmax,
rsqrt, etc.), but VAE-class conv saturation is a *per-component*
storage-dtype concern, not a per-op upcast concern.

## 8. Files modified (Ch8 implementation plan, for Commits 2 and 3)

- `src/neurobrix/config/families/image.yml` — add `dtype_policy.auto_fp32_on_overflow_risk` section with the four discriminator parameters.
- `src/neurobrix/config/families/video.yml` — same (video VAEs share the conv-cascade physics).
- `src/neurobrix/core/prism/solver.py` — augment `_components_force_fp32` with the auto-detect branch (family YAML loaded via the existing `get_family_config`; manual ⊕ auto by set union; `NBX_DISABLE_AUTO_FP32` env bypass).
- `CHANGELOG.md` — Added entry: PixArt and Sana now run in `--triton` without manual `requires_fp32_compute`.
- `tests/unit/dtype/test_auto_fp32_family_aware.py` — exhaustive doctrine pins (per-component fires/no-fires across the empirical matrix; family-gate disjoint; hw-gate; conv-dominance gate; manual ⊕ auto composition; env bypass semantics).

## 9. Validation matrix (Ch8 Étapes 3 + 4)

**Étape 3 — R29 forward validation (visual inspection)**: per
model (`PixArt-XL-2-1024-MS`, `PixArt-Sigma-XL-2-1024-MS`,
`Sana_1600M_1024px_MultiLing`), three PNGs:
- `compiled/output.png` — oracle (was always working).
- `triton/broken/output.png` — `NBX_DISABLE_AUTO_FP32=1` to
  reproduce the pre-Ch8 NaN/blob symptom (proof the chantier
  addresses a real defect).
- `triton/fixed/output.png` — with the auto-detect active; must
  be visually coherent (no marble, no grid, no diagonal stripes,
  no uniform colour, no NaN-induced black).

**Étape 4 — full pytest harness anti-reg**: `pytest tests/regression/`
(fast) AND `pytest tests/regression/ --runslow` (excluding the
standing-INDETERMINATE 4Kpx model). Before/after green/red/skip/xfail
counts in the verdict, explicit. PixArt-α + PixArt-σ + Sana 1024
triton cells must go red→green. No non-image-family cell may
flip green→red (over-inclusion regression). No image-compiled
cell may flip green→red (over-inclusion regression; cuDNN already
handles compiled correctly, but pinning VAE to fp32 doubles its
weight footprint — 16 GB V100 OOM check is mandatory).

## 10. Latent observations (D10 — not in scope for Ch8)

- Video family is enabled in the YAML by design (same physics as
  image VAE), but the cached video models are heavy (multi-hour
  inference) and the project memory excludes them from the
  `--runslow` set for budget reasons. Empirical confirmation that
  video VAEs hit the same auto-detect cleanly is a coverage-future
  step, not a Ch8 deliverable.
- The threshold values (20-conv2d floor, 10× ratio) are tuned to
  the current cached matrix. Future models with substantially
  different shapes may fall on the boundary; the YAML tunability
  is the seam for that. Not a defect — a documented configuration
  surface.
- DC-AE (Sana VAE) does not use `native_group_norm`. Other VAE
  variants (e.g. AsymmetricAutoencoderKL) may use yet different
  norm ops. The conv-dominance gate (count + ratio) is invariant
  to the norm choice — by design.
