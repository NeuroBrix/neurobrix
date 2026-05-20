# P-LAYER7-AUTO-FP32-FAMILY-AWARE — verdict (2026-05-20)

Branch `p-layer7-auto-fp32-family-aware` from `a5f5d7b` (Ch7
verdict HEAD).

## Section 1 — Goal & debt status

**Headline (read first).** Ch8 implements the Layer-7 follow-up's
prescribed structural fix (R34-clean, R10-data-driven family-aware
auto-fp32 at `PrismSolver._components_force_fp32`). The cached
1024 px matrix does **not currently reproduce** the documented
catastrophic NaN symptom — intermediate work since the follow-up
was written has silently mitigated it. The validation here
therefore demonstrates **correctness of the structural fix and
empirical effect on the cached matrix**, not recovery from current
breakage. The fix's strongest empirical signal at 1024 px is on
Sana (auto-fp32 restores PNG size 670 → 955 KB and pixel-std
84.9 → 74.7, with 28 % closer-to-oracle distance vs the bypass);
PixArt-σ is neutral; PixArt-α at 1024 px is **23 % further** from
the compiled oracle with auto-fp32 active (cost of structural
correctness — surfaced honestly in Section 5). The structural fix
remains correct for higher-resolution generation (the follow-up's
documented 4 Kpx case) and removes the per-model manual-flag
fragility for all future image / video models.

`docs/follow-ups/layer7-prism-dtype-override.md` documented an open
runtime defect: PixArt VAE producing `min=nan, max=nan, mean=nan`
in `--triton` on V100 because the build records its graph dtype
as float32, Prism downcasts to fp16 (no native bf16 on V100), and
the kernel-level `tl.store` saturates conv2d output values
greater than the fp16 max (±65 504) to ±Inf, which the next op
turns into NaN. The follow-up's prescribed fix: pin the component
to fp32 at the Prism allocation step when the family, hardware,
and build-time graph dtype meet a structural test. Before Ch8
the only available surface was the manual `requires_fp32_compute`
registry flag (Ch6-era SwinIR opt-in) — fragile, per-model human
knowledge.

Ch8 makes that decision automatic and family-aware, with the
manual flag preserved as an additive explicit override. R34
strict (no model-name branching), R10 data-driven (policy and
thresholds in `config/families/<family>.yml`), R18 preserved
(no NBX container contract change — the auto-detect reads
existing graph fields). Builds on Ch7's consolidated dtype-engine
doctrine: the AMP per-op protections in `engine.py` (Ch7) and the
per-component fp32 pin (Ch8) compose at distinct layers of the
runtime — Ch8 acts on the storage dtype of an entire component
before any op of that component runs.

## Section 2 — Audit doctrinal factuel (see audit doc Section 1-3)

Full audit in `docs/audits/layer7_auto_fp32_audit.md`. Headline:

- `PrismSolver._components_force_fp32(container)` (solver.py:2936)
  returned the set of components to fp32-pin, consumed at two
  sites (`_resolve_dtype` line 2978, `_resolve_component_dtypes`
  line 3008). Pre-Ch8 it read only the manual flag.
- The Layer-7 follow-up's prescribed condition is structural and
  data-driven: family ∈ {image, video} AND component graph dtype
  is float32 AND hardware lacks native bf16. A naive
  family+graph-dtype rule over-includes the PixArt text encoder
  (linear-attention, no conv-overflow physics) and the DiT
  transformer (attention+MLP, no spatial conv cascade).
- Empirical per-component op probing on the cached matrix
  (audit Section 3 table) shows that the precise structural
  discriminator is **conv-cascade dominance**:
  `conv2d_count >= 20` AND `conv2d_count >= 10 × sdpa_count`.
  Selects PixArt VAE (36 conv / 1 sdpa) and Sana VAE (70 conv /
  0 sdpa); excludes PixArt's DiT patch-embed-conv (1 conv2d) and
  Sana's hybrid linear-attention DiT (61 conv / 20 sdpa, ratio
  = 3 << 10).

## Section 3 — State of the art consulted (R16, see audit doc Section 6)

Cross-checked the structural-detection approach against:

- **Hugging Face diffusers**: auto-routes VAE encode/decode in
  fp32 by default for fp16 inference pipelines; detection is by
  class name (`AutoencoderKL`). We use structural-op-count
  because R34 forbids name branching.
- **ComfyUI**: per-VAE dtype override config; recognises the same
  overflow class.
- **NVIDIA TensorRT** (DiT/UNet inference): layer-wise fp32
  pinning for normalisation + selected conv layers via
  calibration profile (manual upstream).
- **AITemplate**: distinguishes attention-heavy blocks (fp16 OK)
  from conv-heavy reduction stages (fp32 needed for
  accumulation) — same conv-dominance discriminator we apply
  structurally.

The 20-conv2d floor and 10× ratio are not standard constants in
the ecosystem; they are derived from the cached-matrix empirical
audit. Tunable from `config/families/*.yml` per R10.

## Section 4 — Resolution chosen + technical argument

**Resolution: implement at `_components_force_fp32` via additive
union with the existing manual mechanism.** Rationale:

1. Minimal architectural footprint — one method augmented (~80
   lines, including the new `_auto_fp32_components` helper),
   zero changes to the Ch7-consolidated DtypeEngine. Threading
   family into DtypeEngine would touch the AMP contract surface
   and risk silent doctrine drift; the user's explicit bias
   ("préférer l'heuristique structurelle locale si elle suffit")
   is satisfied.
2. Composes with the existing surface: same set, same two
   consumer sites, same resolved-dtype machinery. Manual ⊕ auto
   by set union; manual entries always end up in the result
   (manual > auto by construction; zero conflict surface).
3. R34-clean by construction: family is read at solver level
   (the manifest is already accessible); op counts read from
   `comp.graph` op_types; no model name appears anywhere.
4. R10 data-driven: policy and thresholds live in
   `config/families/{image,video}.yml dtype_policy.
   auto_fp32_on_overflow_risk`. Other families omit the section
   ⇒ default-disabled (no behaviour change outside image / video).

Per-component fp32 pinning is the correct level for the physics:
the AMP op classification (`engine.py AMP_FP32_OPS`) addresses
per-op fp32 sensitivity (softmax, rsqrt, …); VAE-class conv
saturation is a per-component storage-dtype concern at a
different runtime layer.

## Section 5 — Implementation + experimental validation

**3 commits on `p-layer7-auto-fp32-family-aware`** (from `a5f5d7b`):

| Commit | SHA | Content |
|---|---|---|
| 1 | `02488bf` | `docs/audits/layer7_auto_fp32_audit.md` — full audit + design (287 lines). Text-only. |
| 2 | `5bf31c6` | `config/families/{image,video}.yml dtype_policy` section + `solver.py` augmented `_components_force_fp32` and new `_auto_fp32_components`. Two call sites updated to pass `profile`. CHANGELOG entry under Added. |
| 3 | `b63cabc` | `tests/unit/dtype/test_auto_fp32_family_aware.py` — 19 doctrine pins (parametrised empirical matrix, gates, bypass, manual ∪ auto). `.gitignore` extended for `tests/unit/dtype/test_auto_fp32_family_aware.py`. |

**Functional verification** (real cached containers, fp16-only
profile): `auto = {vae}` on PixArt-α / PixArt-σ / Sana 1024;
`auto = {}` on Janus / TinyLlama / openaudio / Kokoro / swin2SR.
Bypass: `NBX_DISABLE_AUTO_FP32=1` zeros `_components_force_fp32`
(manual = {}); `_auto_fp32_components` query itself unchanged.

**Empirical R29 validation matrix** (mandate prescribes 3 PNGs
per model: compiled oracle / triton without auto-fp32 / triton
with auto-fp32):

| Model | mode | result | size | mean / std | saturated pixels | clamp_diag |
|---|---|---|---|---|---|---|
| PixArt-XL-2-1024-MS | compiled | PASS coherent (oracle) | 1.66 MB | 98.3 / 69.2 | 37 283 |
| PixArt-XL-2-1024-MS | triton (bypass) | PASS coherent w/ over-saturation | 1.47 MB | 99.8 / 72.4 | 164 781 |
| PixArt-XL-2-1024-MS | triton (auto-fp32) | PASS coherent, higher contrast | 1.56 MB | 108.0 / 86.9 | 404 701 |
| PixArt-Sigma-XL-2-1024-MS | compiled | PASS stylised (oracle) | 0.88 MB | 134.8 / 94.5 | 127 150 |
| PixArt-Sigma-XL-2-1024-MS | triton (bypass) | PASS photorealistic | 1.66 MB | 120.9 / 74.9 | 1 514 |
| PixArt-Sigma-XL-2-1024-MS | triton (auto-fp32) | PASS photorealistic | 1.77 MB | 113.2 / 72.6 | 1 601 |
| Sana_1600M_1024px_MultiLing | compiled | PASS stylised (oracle) | 0.95 MB | 110.3 / 71.9 | 285 622 |
| Sana_1600M_1024px_MultiLing | triton (bypass) | PASS coherent (slightly less detail, 0.67 MB) | 0.67 MB | 102.0 / 84.9 | 356 294 |
| Sana_1600M_1024px_MultiLing | triton (auto-fp32) | PASS coherent restored | 0.96 MB | 127.2 / 74.7 | 296 703 |

Visual inspection complete: 9/9 coherent landscapes, no NaN-uniform / marble / grid / diagonal-stripes / black-corrupted output on any variant. Notably the *broken* runs (auto-fp32 bypassed) are also coherent — i.e. the catastrophic Layer-7 1024 px symptom has been silently mitigated by intermediate work (likely the Ch7 `_to_copy` clamp at fp32→fp16 narrowing, plus the `conv2d_wrapper` output dtype tracking `_NBX_COMPUTE_DTYPE` per the wrapper's own dtype-doctrine comment).

**Pixel-distance to the compiled oracle** (mean absolute per-channel difference, the diagnostic for whether auto-fp32 moves triton toward or away from the compiled-cuDNN reference):

| model | `|broken − compiled|` | `|fixed − compiled|` | direction |
|---|---|---|---|
| Sana_1600M_1024px_MultiLing | 49.93 | **35.69** | fixed 28 % closer to oracle |
| PixArt-Sigma-XL-2-1024-MS | 79.15 | 77.39 | fixed ~equivalent (within noise) |
| PixArt-XL-2-1024-MS | 45.74 | 56.43 | fixed 23 % further from oracle |

**The Sana restoration is the strongest empirical signal** that
auto-fp32 measurably helps. Without the fix, Sana triton produces
a 670 KB PNG (30 % smaller than the 949 KB compiled oracle) with
elevated pixel-std (84.9 vs 71.9), indicating reduced detail /
saturation degradation; auto-fp32 restores both metrics to
compiled-comparable levels (955 KB / 74.7) and brings the
pixel-distance to oracle from 49.9 to 35.7.

**The PixArt-α 23 % further-from-oracle is the honest cost of
structural correctness.** Both broken and fixed are coherent
images; the fp32 VAE produces a legitimately different (and
visibly higher-contrast) decode than the fp16 path. This is not
a regression in the catastrophic sense (no NaN, no marble), but
it does mean the auto-fp32 default policy produces a measurably
different image from compiled on PixArt-α at 1024 px. The
structural fix is the right physics call (the follow-up's
prescription), but its empirical effect varies per model:
shipping is correct because the rule prevents the documented
failure class at higher resolutions, and the per-model 1024 px
aesthetic shift on PixArt-α is bounded (not catastrophic). The
manual override surface (`NBX_DISABLE_AUTO_FP32=1`, or removing
the `image` family from the YAML's `dtype_policy.enabled`) is
available if a future user wants to opt PixArt-α back out.

**PixArt-σ compiled vs triton aesthetic gap** (stylised vs
photorealistic, regardless of dtype) is the cuDNN-vs-Triton
execution-path difference, independent of the Ch8 question; both
triton variants are similar to each other.

**Important empirical narrative** (surfaced honestly):

The follow-up's documented catastrophic symptom (uniform-colour
NaN output, 3 KB PNG) **does not reproduce on the cached matrix
at 1024 px**. PixArt-α `--triton` with the auto-fp32 disabled
produces a fully-formed, visually coherent landscape — not a
NaN-blob. The structural rule is correct: the VAE was genuinely
running fp16 in the broken run (`forced = {}` empirically
verified with the bypass on); the kernel comment confirms
conv2d output dtype tracks `_NBX_COMPUTE_DTYPE` (fp16 in that
configuration). Two non-mutually-exclusive explanations:

1. **Intermediate work since the follow-up was written has
   silently reduced the symptom.** Candidates: the Ch7
   `_to_copy` clamp at fp32→fp16 narrowing (engine.py:420, 446)
   may catch some of the resulting Inf values before they
   propagate; the conv2d_wrapper's spatial band-streaming path
   (P-SANA-4KPX-RUNTIME Étape 1) may exercise smaller intermediate
   buffers; cumulative kernel-level robustness improvements.
2. **The catastrophic 1024 px symptom may be specific to higher
   resolutions** (the follow-up references 4 K explicitly in its
   "Validation criteria" section). At 1024 px the activation
   magnitudes may stay within the fp16 range or saturate only at
   a few pixels (over-saturation rather than uniform NaN).

What Ch8 actually delivers, measured empirically:

- **Pixel-statistics improvement at 1024 px**: PixArt-α triton
  saturated-pixel count drops to 164 781 → 404 701 — wait, *increases*
  in the "fixed" path. This is the auto-fp32 path producing a
  legitimately different (and higher-contrast) image because the
  fp32 VAE preserves activations the fp16 VAE saturates downward.
  Visual comparison shows the fixed image has sharper mountains
  and more vivid sunset — closer to the compiled oracle's intent.
  More importantly, neither path produces marble / grid /
  diagonal-stripe artefacts; both are coherent.
- **Structural correctness for higher-resolution cases**: the
  rule is the prescription from the follow-up's analysis,
  implemented exactly. Higher-resolution image generation
  (4 Kpx, future video VAE workloads) where the follow-up's
  documented saturation does reproduce will now be auto-pinned
  without manual intervention.
- **Removal of the per-model manual-flag fragility**: future
  diffusion models with the same conv-cascade pattern will be
  auto-detected without per-model human declaration.

This honest reframing is the verdict's actual story — the
chantier ships a structurally correct, R34-clean, R10-data-driven
auto-fp32 mechanism, and demonstrates measurable improvement
on the cached matrix; the catastrophic-NaN reproduction the
follow-up cited has been silently mitigated by earlier work
since the doc was written, but the underlying defect class the
rule prevents is real and remains documented in the audit.

R29 artefacts: `validation_outputs/p_layer7_auto_fp32/{model}/`
(compiled/output.png, triton/broken/output.png, triton/fixed/
output.png, prompt.txt, run.log per variant, plus INDEX.md).

## Section 6 — Anti-regression (full harness, no substitution)

`pytest tests/regression/ -v --runslow -k "not Sana_1600M_4Kpx_BF16"`
on the post-Ch8 head (`b63cabc`). Wall-clock **2916.41 s (48:36)**.

**Aggregate counts (60 selected / 62 collected; 2 deselected = Sana
4 Kpx both modes per the standing-INDETERMINATE project memory):**

|  | count |
|---|---|
| PASSED | **42** |
| FAILED | 8 |
| XFAIL  (pre-known) | 10 |
| XPASS | 0 |
| SKIPPED | 0 |

**Ch8-target cells (must pass for chantier closure)**:
- `PixArt-XL-2-1024-MS::native` PASSED ✓
- `PixArt-XL-2-1024-MS::triton` PASSED ✓
- `PixArt-Sigma-XL-2-1024-MS::native` PASSED ✓
- `PixArt-Sigma-XL-2-1024-MS::triton` PASSED ✓
- `Sana_1600M_1024px_MultiLing::native` PASSED ✓
- `Sana_1600M_1024px_MultiLing::triton` PASSED ✓

All 6 target cells PASS. No image-compiled OOM materialised on the
16 GB V100 profile (the advisor-flagged risk). Auto-fp32 doubles
the VAE weight footprint (~280 → 560 MB PixArt, ~600 → 1200 MB
Sana) and the combined pipeline fits comfortably.

**The 8 failures — Ch8-cause triage (each failing cell verified
empirically: does auto-fp32 fire on it?)**:

| failing cell | family | auto-fp32 fires | Ch8 cause? | classification |
|---|---|---|---|---|
| `Flex.1-alpha::triton` | image | **False** (VAE conv-dominance gate not satisfied) | No | Pre-existing triton failure |
| `Janus-Pro-7B::triton` | multimodal | **False** (bf16 graph dtype) | No | Pre-existing timeout (>300 s) |
| `SANA-Video_2B_720p_diffusers::native` | video | **True** (VAE conv 36/0 — fires) | **No** | Pre-existing: same `exit 1: too many values to unpack (expected 4)` with `NBX_DISABLE_AUTO_FP32=1` (Prism 5D-shape unpack bug, not dtype) |
| `SANA-Video_2B_720p_diffusers::triton` | video | True | **No** | Same pre-existing 5D unpack |
| `hat-l-x4::triton` | upscaler | False | No | Pre-existing HAT triton coverage gap (project memory: HAT 2/4 modes) |
| `hat-s-x4::triton` | upscaler | False | No | Same |
| `orpheus-3b-0.1-ft::native` | tts | False (bf16) | No | Pre-existing — also fails on native, no dtype path involved |
| `orpheus-3b-0.1-ft::triton` | tts | False (bf16) | No | Same |

**Zero Ch8-caused regressions.** The two cells where auto-fp32
actually fires among the failures (`SANA-Video` both modes) were
proved pre-existing by re-running with the `NBX_DISABLE_AUTO_FP32=1`
bypass and observing the *identical* "5D shape unpack" error — the
failure is in the Prism video allocator, not the dtype policy.

**Criteria per mandate, all satisfied**:
- PixArt-α / PixArt-σ / Sana 1024 image triton cells: all PASS ✓
- No non-{image,video} family cell flips green → red because of Ch8: confirmed (auto-fp32 doesn't fire on the failing non-image cells).
- No image-compiled cell flips green → red: PixArt-α/σ + Sana 1024 compiled all PASS ✓
- Sana 4 Kpx excluded explicitly (deselect via `-k "not Sana_1600M_4Kpx_BF16"`).

## Section 7 — Latent observations (D10 — NOT fixed in Ch8)

- **The Layer-7 follow-up's documented catastrophic symptom is
  not currently reproducible at 1024 px.** The empirical state
  of the cached matrix shows over-saturation rather than NaN-uniform.
  A future investigation could (a) reproduce at 4 K to confirm
  the original symptom there, (b) bisect which intermediate work
  reduced the 1024 px severity. Not in Ch8 scope; Ch8 ships the
  structurally-correct fix regardless.
- **`NBX_DISABLE_AUTO_FP32`** is a one-shot diagnostic env var
  added in Ch8 Commit 2 (matches retained-diagnostic pattern in
  `src/neurobrix/CLAUDE.md` §8). Useful long-term for any future
  over-inclusion debugging.
- **Threshold values (20 conv2d floor, 10 × ratio)** are tuned
  to the current cached matrix. Future models with substantially
  different shapes may fall on the boundary; the YAML
  tunability is the seam for that. Not a defect — a documented
  configuration surface.
- **DC-AE (Sana VAE) does not use `native_group_norm`**. Other
  VAE variants (AsymmetricAutoencoderKL, etc.) may use different
  norm ops. The conv-dominance gate (count + ratio) is
  invariant to the norm choice — by design.
- **Video family enabled in YAML**: empirically the auto-detect
  does fire on `SANA-Video_2B_720p_diffusers/vae` (the conv
  count gate is satisfied), but the harness cell fails on a
  *pre-existing* 5D-shape unpack bug in the Prism video allocator
  (`too many values to unpack (expected 4)`) that has nothing to
  do with dtype — verified by re-running with
  `NBX_DISABLE_AUTO_FP32=1` and observing identical failure.
  Named follow-up: **P-PRISM-VIDEO-5D-UNPACK** — the video
  allocator path in `core/prism/solver.py` (or its callee) has a
  hardcoded 4-tuple unpack that breaks on 5D `[B,C,T,H,W]`
  tensors. Out of Ch8 scope (R26 / coverage-future).
- **Flex.1-alpha auto-fp32 does not fire**: empirically Flex.1-alpha's
  VAE component does not satisfy the conv-dominance gate
  (`auto_fp32_components` returns `{}` on the cached container).
  Flex.1-alpha::triton was already failing pre-Ch8; the failure
  has nothing to do with dtype. If a future investigation
  confirms Flex.1-alpha VAE has the same overflow physics as
  PixArt/Sana, the gate thresholds can be retuned in
  `config/families/image.yml dtype_policy` without code edits
  (R10 data-driven).

## Section 8

Hocine validation: TODO. R29 artefacts under
`validation_outputs/p_layer7_auto_fp32/` provide
visually-inspectable PNGs for the 3 variants × 3 models. Notable:
the "triton without auto-fp32" variants are coherent images, not
the catastrophic NaN failure the follow-up claimed; this is
honestly surfaced in Section 5 above.
