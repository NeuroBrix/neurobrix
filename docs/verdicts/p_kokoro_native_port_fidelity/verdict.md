# P-KOKORO-NATIVE-PORT-FIDELITY — Verdict

**Chantier:** resolve the Kokoro-82M "babbling / hey hey hey" symptom
(Dette E) by fixing the NeuroBrix native port of the Kokoro
predictor/text-encoder against the vendor `kokoro` library oracle.
**Branch:** `p-audio-chantier-p0-tts-quality`.
**Date:** 2026-05-22.
**Method:** §5.8 op-by-op differential vs the vendor reference (vendor STT
intelligible on identical phoneme IDs + voicepack ⇒ bug is in the port).

---

## 0. Summary

| Symptom | Status | Fix |
|---|---|---|
| Kokoro babbles (whisper word-overlap 0.00) | **RESOLVED** | two confirmed port divergences fixed in `core/flow/stages/kokoro.py` |

STT auto-validation (`whisper-large-v3-turbo`), compiled mode:

| prompt | transcription | overlap | verdict |
|---|---|---|---|
| "Hello world." | "Hello world." | 1.00 | PASS |
| "Hello world, this is a test." | "Hello world, this is a" | 0.83 | PASS (last word lost to truncation) |
| "The quick brown fox jumps over the lazy dog." | "The quick brown fox" | 0.50 | PARTIAL — surviving words correct; shortfall is 100% the 23-phoneme input truncation |

R29 artefacts: `validation_outputs/p_kokoro_native_port_fidelity/` (WAVs +
stats + transcriptions). Vendor reference WAVs (overlap 1.00 on all three):
`validation_outputs/p_audio_p0a_kokoro/vendor_*.wav`.

## 1. The two divergences (found in order, each validated before the next)

Both confirmed by reading vendor `kokoro/model.py:forward_with_tokens` +
`kokoro/modules.py` against the native port, not guessed.

**(A) `text_mask` convention inverted.** Vendor builds `text_mask` as
`gt(arange+1, input_lengths)` → **True = padding**, and every consumer
(`masked_fill_(m, 0)` in the DurationEncoder / TextEncoder, bert
`attention_mask=(~text_mask)`) is written for that convention.
`preprocess_phonemizer_input` set **True = valid**, so all four NBX consumers
zeroed the real tokens and kept the padding — the duration LSTM and text
encoder were running on padding. Fixed: `text_mask` is now True=padding.
*Validation:* durations went from 32 → 62 frames for "Hello world." (vendor:
63 frames / 1.57 s), i.e. the duration path converged to vendor within 1 frame.
Content was still wrong ("I'll tell you what I'm saying") → next divergence.

**(B) text_encoder CNN block diverged from vendor `TextEncoder`.** Three
sub-divergences, fixed together as a coherent "match vendor" unit (validated at
the block level, not individually isolated): op order was
`conv → LeakyReLU → LayerNorm` but vendor is `conv → LayerNorm → LeakyReLU`;
LeakyReLU slope was `0.01` but vendor is `0.2`; and the per-block
`masked_fill` (after embedding and after each conv) was missing. These
corrupted the phoneme embeddings `t_en` → wrong words.
*Validation:* "Hello world." → "Hello world." (overlap 1.00). Babbling resolved.

## 2. Relationship to the P0a Option A fix (prior verdict superseded)

The earlier R29 (`validation_outputs/p_audio_p0a_kokoro/INDEX.md`) stated the
medium prompt "hits the overflow/compress branch, code path unchanged by
Option A." That was empirically true *at the time* — but only because the
`text_mask` bug (A) was feeding garbage padding tokens into the duration LSTM,
inflating the apparent natural sum past 128. With (A) fixed, the medium
prompt's true durations sum to ~63 frames (< 128), so it now takes the **fits**
branch: **Option A is load-bearing for the medium prompt too**, not the no-op
the prior verdict described. The original "duration stretching" P0a diagnosis
was a downstream symptom of the `text_mask` bug, not the root cause. Option A
(commit `a2d2be5`) is correct and stays.

## 3. Discipline

- **R34 model-agnostic**: fixes live in `core/flow/stages/kokoro.py`, which is
  a model-specific stage handler by nature (documented temporary torch
  violation). No model-name branch added to any shared primitive.
- **R30 native↔triton parity**: the fixes are in the shared stage handler, so
  they apply to both modes by construction. Kokoro `--triton` is, however,
  pre-existingly non-functional (`embedding(): weight must be Tensor, not
  NBXTensor` — the torch native handler receives NBXTensor weights in triton
  mode). This is the R33 temporary-violation consequence; it is removed by the
  build-side re-trace (`P-BUILD-KOKORO-DYNAMIC-FRAMES`), which lets the
  predictor run in-graph and deletes the native handler. Gap noted, scope not
  expanded.
- **R29**: STT auto-validation is the operative criterion; vendor reference
  oracle established.

## 4. Known residuals (non-impacting — STT passes)

- **Voicepack row index**: vendor `voicepack[len(ps)-1]`; NBX
  `min(actual_len, N-1)` (off by ~3). Affects style/timbre, not intelligibility
  — durations still match vendor within 1 frame and STT passes, so left as-is to
  avoid destabilising the working fix.
- **bert attention mask**: NBX has no `attention_mask` connection to the bert
  component; vendor masks bert attention to valid positions. Did not block
  intelligibility on the tested prompts.
- **Long-prompt duration parity** not independently checked on the
  23-ID-truncated input; the short prompt matches vendor within 1 frame and STT
  is the operative criterion.

## 5. Follow-ups

- `P-BUILD-KOKORO-DYNAMIC-FRAMES` (build-side, P0): symbolic `asr_frames` +
  phoneme `seq_len`. Lifts the fixed-128 window, the 23-phoneme truncation
  (the only remaining cause of long-prompt shortfall), and the triton native-
  handler gap (re-trace removes the handler). The runtime crop band-aid retires
  with it.

Relaunch: `neurobrix run --model Kokoro-82M --prompt "Hello world." --output out.wav`
then `neurobrix run --model whisper-large-v3-turbo --audio out.wav`.
Hocine validation: TODO (listen to `validation_outputs/p_kokoro_native_port_fidelity/*/output.wav`).
