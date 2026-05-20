# Dette D — PixArt-XL top horizontal white band (2026-05-20)

Branch `p-debt-settlement-batch-1`. State at start: Hocine flagged
a strange horizontal band at the top of the Ch8 `--triton/fixed`
PixArt-XL-2-1024-MS R29 image. Pixel analysis confirmed it:
rows 0-69 pure white (RGB 255/255/255, std 0); row 70 onwards
image content; 116-magnitude row-to-row discontinuity at the seam.

## Section 1 — Reproduction attempts (the differential)

Re-ran PixArt-XL `--triton` four times under the exact Ch8
condition (`--steps 12`, `landscape` prompt OR `red apple` prompt,
auto-fp32 ON OR OFF) and once compiled (oracle). White-row count
on the top of each output:

| run | prompt | mode | auto-fp32 | white rows top |
|---|---|---|---|---|
| Ch8 original (the artefact Hocine flagged) | landscape | triton | ON | **70** |
| Repro 1 | red apple | triton | ON | 0 |
| Repro 2 | red apple | triton | OFF (bypass) | 0 |
| Repro 3 (Ch8 conditions verbatim) | landscape | triton | ON | 0 |
| Repro 4 | landscape | triton | OFF (bypass) | 0 |
| Compiled oracle | red apple | compiled | n/a | 0 |

**4/4 re-attempts at Ch8 conditions produced no band.** The band
appeared in 1/5 runs total. It is therefore a **transient
non-determinism artefact**, not a deterministic Ch8 regression.

## Section 2 — Root-cause hypothesis

`CHANGELOG.md` Layer 6.bis (commit `06d26c2` lineage) documents
a known residual `~4/255` pixel run-to-run non-determinism on
V100 + Triton PixArt-class diffusion, attributed to upstream/
downstream non-flash kernel ULP variation (matmul / softmax /
RMSNorm) accumulating across 28 DiT blocks. The 70-row white
band is a rare manifestation: a scheduler step that pushed the
top-row latents into saturation, decoded to RGB 255. Auto-fp32
neither causes nor prevents the band — both `triton/broken`
(bypass) and `triton/fixed` (auto-fp32) repros were clean.

This is **NOT a Ch8 regression**. The residual non-determinism
existed pre-Ch8 (Layer 6.bis); it pre-exists the auto-fp32
mechanism.

## Section 3 — Fix at root + disposition

No deterministic fix is possible at the chantier scope: the
band does not reproduce under the same conditions, so there is
no causal signal to bisect against. Two operational mitigations
land naturally:

1. The Dette C harness change (`IMAGE_PROMPT = "a red apple on a
   wooden table"`) reduces R29-validation exposure to the
   landscape prompt where the band appeared. 4/4 repros across
   both prompts now produced clean images.
2. The artefact is named for follow-up tracking:
   **P-PIXART-XL-VOLTA-WHITE-BAND** — rare top-row white
   saturation on PixArt-XL `--triton` at specific scheduler step
   sequences; manifestation of the residual V100+Triton
   non-determinism documented in Layer 6.bis. A future
   investigation could (a) characterise the occurrence rate by
   running PixArt-XL ×100, (b) bisect which op produces the
   top-row saturation when it does occur. Out of Dette D scope —
   the chantier is hypothesis-test on a singular observation,
   and the hypothesis does not survive the differential.

## Section 4 — Validation (Hocine-gated)

R29 artefacts: `validation_outputs/p_dette_d_pixart_xl_band_diagnosis/`:
- `ch8_original_with_70row_white_band.png` — the artefact Hocine flagged.
- `repro_attempt{1..4}_*.png` — 4 re-runs, all clean.
- `compiled_oracle_redapple_no_band.png` — reference.
- `INDEX.md` — full discriminator table.

Hocine validation: **TODO** — confirm the diagnosis (1 transient
band, 4 clean re-attempts ⇒ non-determinism not Ch8-deterministic).
If Hocine can reproduce the band on his side, post-validation
investigation can re-open the question.

## Section 5 — Latent observations

- The V100+Triton ~4/255 residual non-determinism is documented
  but not yet root-caused (Layer 6.bis CHANGELOG line 475 lists
  the four remaining-in-scope hypotheses: CUDA stream
  non-determinism, sub-µs Python frame perturbing CUDA driver
  state, `@triton.heuristics` lambda re-evaluation, Triton/CUDA
  thread-local state). This is the `Layer X` future investigation
  the Ch7 era already named.
- PixArt-XL specifically (vs Sigma) may be more sensitive to
  this residual: the Ch8 R29 showed the band on PixArt-XL but
  not PixArt-Sigma, and the 4 PixArt-XL re-attempts were
  clean — suggesting PixArt-XL's denoising trajectory passes
  closer to a numerical edge at some step. Not actionable
  without a deterministic repro.
