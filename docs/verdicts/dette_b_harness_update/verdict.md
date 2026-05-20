# Dette B — Harness update: current matrix + R26 families + Sana 4Kpx + coverage-debt visibility (2026-05-20)

Branch `p-debt-settlement-batch-1` (from `c3fa116` Ch8 verdict
HEAD; Dette A landed `eb7ed9d`).

## Section 1 — Goal & state at start

The harness was reporting an inaccurate picture of what NeuroBrix
should run:

1. **Pre-existing failures not declared xfail.** Ch8's full
   `--runslow` harness showed 8 FAILED cells — 7 of them
   pre-existing (Flex.1-alpha::triton, Janus-Pro-7B::triton,
   SANA-Video both, hat-l-x4::triton, hat-s-x4::triton,
   orpheus both). They were each triaged in the Ch8 verdict
   as not-Ch8-caused, but they appeared as red regressions in
   the harness output anyway.
2. **Sana_1600M_4Kpx_BF16 invisible.** The standing-INDETERMINATE
   triton case was not in the harness `KNOWN_FAILURES` list at
   all — the harness simply did not test it, so the run showed
   no signal that this model exists and what its state is.
   Worse: the project memory said "do not relaunch under longer
   budgets" without distinguishing compiled (green, 36s per the
   P-SANA-4KPX-RUNTIME closure) from triton (SIGTERM at >3h).
3. **Coverage-gap families invisible.** The harness discovered
   models dynamically from `~/.neurobrix/cache/`. Any family
   that had never been opened (notably **VLM — whole family
   never traced**) showed no signal at all. The harness was an
   inventory of "what we built", not "what we are expected to
   build".
4. **R26 family taxonomy partially absent from `FAMILY_TIMEOUT_S`.**
   Audio_llm / multimodal / stt / tts / vlm fell through to the
   unknown-family default (300s), masking budget mismatches.

## Section 2 — Fix (three layers)

**Layer 1 — declare the 7 pre-existing failures + Sana 4Kpx.**
Added 8 new entries to `tests/regression/conftest.py:KNOWN_FAILURES`,
each pointing at the named follow-up that tracks the underlying
defect (`P-FLEX1-VAE-FP32-GATE`, `P-PRISM-VIDEO-5D-UNPACK`,
`P-TRITON-IM2COL-KERNEL`, etc.) so the harness output makes the
debt visible. Sana_1600M_4Kpx_BF16 listed as **triton-only**
xfail (`P-SANA-4KPX-TRITON`) — compiled is green and demonstrably
runs in 146 s on the current head, so marking both modes xfail
would have been dishonest (and verified empirically: `pytest
Sana_1600M_4Kpx_BF16::native --runslow` returns XPASS).

**Layer 2 — fill `FAMILY_TIMEOUT_S` for the R26 sub-families.**
Added `audio_llm: 300`, `stt: 240`, `tts: 240`, `multimodal: 300`,
`vlm: 180`. Each carries an inline rationale (audio_llm is
audio encode + LLM decode; multimodal autoregressive image is
heavier than plain LLM; vlm placeholder for taxonomy completeness
until a vlm lands). `SLOW_FAMILIES` deliberately unchanged
(`{"image", "video"}`) — audio_llm and multimodal complete
within the fast budget on the cached matrix.

**Layer 3 — make coverage-gap families visible (the complement
Hocine added mid-debt).** New `TARGET_MATRIX_NOT_TRACED` list +
`discover_target_matrix()` helper + `_build_parametrize()`
extension. 13 cells now appear in every collection-only output:

| family | count | models |
|---|---|---|
| vlm | 1 | Qwen3-VL-30B-A3B-Thinking |
| llm (gemma sub-family) | 2 | gemma-4-26B-A4B-it, gemma-4-E4B-it |
| video | 10 | Allegro, Allegro-TI2V, CogVideoX-2b, CogVideoX-5b-I2V, mochi-1-preview, Open-Sora-v2, Wan2.1-I2V-14B, Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B, Wan2.2-I2V-A14B |

Each cell is parameterised as `<model>::not_traced` (single mode
— the model isn't built, so the mode distinction is moot until
it lands in the cache). `_marks_for` now special-cases
`status="not_traced"` with `pytest.mark.skip(reason=
"DETTE_TECHNIQUE_NON_OUVERTE: …")`. The harness output therefore
lists every coverage-gap as SKIPPED at every run — the inventory
is honest about what NeuroBrix should run, not only what it
currently does.

## Section 3 — Validation

- `pytest --collect-only --runslow`: **75 tests collected**
  (was 62). 31 cached models × 2 modes (= 62) + 13 not-traced
  reference-matrix cells. Sana_1600M_4Kpx_BF16::native and
  ::triton both collected (was: implicit, only ran if invoked
  by name; now: visible in the matrix).
- `pytest Sana_1600M_4Kpx_BF16::native --runslow` →
  **XPASS 146.21s** (compiled green, single PNG written).
- `pytest Sana_1600M_4Kpx_BF16::triton --runslow` →
  expected xfail (will SIGTERM per memory; harness now declares
  this).
- `pytest hat-l-x4::triton` → **XFAIL 5.53s** (was: FAILED 0s
  in Ch8 harness; now: declared known-failure with
  P-TRITON-IM2COL-KERNEL reason).
- `pytest Qwen3-VL-30B-A3B-Thinking::not_traced --runslow` →
  **SKIPPED with reason "DETTE_TECHNIQUE_NON_OUVERTE: VLM
  family has not been opened — no traced model. Whole-family
  technical debt (Hocine's dedicated chantier)."**
- `pytest Allegro::not_traced --runslow` → **SKIPPED with
  video-family-coverage-gap reason.**

## Section 4 — Cells reconciled per category

**Existing xfails kept** (10 — all confirmed alive by the Ch8
harness): whisper-large::triton, whisper-large-v3-turbo::triton,
parakeet-tdt-1.1b::triton, canary-qwen-2.5b::triton,
granite-speech-3.3-8b::triton, chatterbox::triton,
openaudio-s1-mini::triton, Kokoro-82M::triton, VibeVoice-1.5B
(both).

**New xfails added** (8): Flex.1-alpha::triton (P-FLEX1-VAE-
FP32-GATE), Janus-Pro-7B::triton (timeout), SANA-Video both
(P-PRISM-VIDEO-5D-UNPACK), hat-l-x4::triton + hat-s-x4::triton
(P-TRITON-IM2COL-KERNEL), orpheus both (audio chantier follow-up),
Sana_1600M_4Kpx_BF16::triton (P-SANA-4KPX-TRITON).

**New not-traced reference cells** (13): listed in Section 2
Layer 3.

## Section 5 — Latent observations (D10)

- Several cached manifests have inconsistent `family` values
  vs the R26 taxonomy: `whisper-large` is `family=audio` (vs
  `family=stt` like whisper-large-v3-turbo); `canary-qwen-2.5b`
  is `family=audio` (vs `family=audio_llm` like Voxtral);
  `parakeet-tdt-1.1b` is `family=audio` (vs `family=stt`);
  `Kokoro-82M` and `VibeVoice-1.5B` are `family=audio` (vs
  `family=tts`). The harness does not fix this — that's a
  build-side taxonomy harmonisation (out of scope for Dette
  B; should be a separate model-registry hygiene chantier).
- `SLOW_FAMILIES = {"image", "video"}` does not include
  audio_llm or multimodal. Audio_llm runs (Voxtral 157s on
  Ch7, canary-qwen, granite-speech) and multimodal (Janus
  193s on Ch7) currently fit within the fast budget; if a
  future heavier audio_llm or multimodal model lands, this
  set may need extension.
- The harness deliberately enumerates the target matrix in
  one place (`TARGET_MATRIX_NOT_TRACED` in conftest.py).
  When a not-yet-traced model lands in the cache, the entry
  must be removed from this list manually (otherwise the
  model would be listed twice — once as testable, once as
  not-traced). A follow-up could auto-remove based on
  cache-presence, but doing it manually keeps the list as
  the explicit reference of "what we expect" — drift is
  surfaced rather than silent.

## Section 6

Hocine validation: not required (infrastructure / inventory
change; no semantic output to inspect). The next harness run
will display 75 cells with the correct PASS / XFAIL / SKIP
profile.
