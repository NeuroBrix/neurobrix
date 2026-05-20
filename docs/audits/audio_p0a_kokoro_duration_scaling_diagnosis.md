# Audio P0a — Kokoro phonemes diagnosis (2026-05-20)

§5.8 differential identified the structural bug behind the
Hocine-flagged "babbling phonemes / hey hey hey" symptom on
`Kokoro-82M`. STT auto-validation: pre-fix whisper-large
transcribes the WAV as empty string (Dette E baseline).

## Section 1 — Empirical signature

Runtime probe with `NBX_DUMP_TIDS=1` (CLAUDE.md §8 diagnostic):

```
[Phonemizer] 'Hello world, this is a test.' -> 29 phonemes -> 23 IDs (padded to 23)
[Voicepack]  Loaded 'af_heart' (ref_s=[1, 256])
[bert]         Done in 949ms
[bert_encoder] Done in 14ms
[text_encoder] output=[1, 512, 23]
[predictor]    asr=[1, 512, 128] F0=[1, 256] N=[1, 256]
[decoder]      Done in 1830ms
[output WAV]   3.2s (Hocine STT-auto-confirmed empty transcription).
```

Smoking-gun shape: `asr=[1, 512, 128]` — **128 frames of phoneme-
embedding** were generated for a 29-phoneme English prompt. At
the iSTFTNet stage handler's implied 80 ms/frame mel rate, 128
frames ≈ **10 seconds** of audio. The expected duration for
"Hello world, this is a test." is ≈ 2-3 s (≈ 25-40 frames).
The phoneme-to-frame alignment is **stretched ~4× longer** than
natural speech.

## Section 2 — Code path

`src/neurobrix/core/flow/stages/kokoro.py:_scale_kokoro_durations`
(line 181 caller, definition ~line 410). For each prompt the
function:

1. Predicts raw durations from the duration LSTM + projection
   (`raw_durations = sigmoid(dur_logits).sum(dim=-1) / speed`).
2. **Force-scales the predicted durations so they sum to exactly
   `target_asr_frames` (the trace-time fixed value, 128)**:
   ```
   scale = target / current_sum.item()
   durations[active] = round(durations[active] * scale).clamp(min=1)
   ```
3. Builds the alignment matrix `[B, num_phonemes, target_asr_frames]`
   with the scaled durations.

The trace-time `target_asr_frames` is read from the decoder
graph's `asr` tensor shape (`_get_kokoro_decoder_shapes`,
line 679). The build-time trace captured the decoder with a
**fixed** 128-frame asr input. The stage handler enforces that
fixed shape at runtime by scaling whatever natural durations
the predictor outputs.

Consequence: every phoneme gets `~128 / num_phonemes`-frame
allocation regardless of the natural duration the model
predicted. For a 29-phoneme prompt the average is ~4.4
frames/phoneme — already on the long side; for shorter prompts
(~10 phonemes) it climbs to ~12 frames/phoneme, **stretching
each phoneme 3-5× beyond natural**. The decoder synthesises a
plausible iSTFTNet waveform of the stretched phoneme sequence —
which sounds like garbled vowel-elongation / "hey hey hey"
repetition (Hocine's description).

The F0 and N curves are interpolated to `target_f0_len=256`
(half-rate of the audio-sample upsample). The interpolation
itself is correct; the input feature has the same stretching
defect.

## Section 3 — Why the trace baked a fixed frame count

Build-time, the build subtree captured Kokoro on a long-enough phoneme
sequence to produce a 128-frame decoder input. The decoder
graph captured that exact shape as a runtime expectation. The
phoneme path was traced with a `pad_to=23` mask so the input
phoneme tensor IS dynamic across runs; what's NOT dynamic is
the post-predictor `asr / F0 / N` decoder input shape.

This is a Kokoro-specific trace-shape problem. Most other
audio models (Voxtral, openaudio, chatterbox, parakeet,
whisper) carry the dynamic-duration shape through the decoder
naturally — Kokoro's stage handler hand-rolls the predictor
in Python torch (post-Ch3 native port), and that hand-rolled
path forces the decoder input shape.

## Section 4 — Fix design (3 options)

**Option A — Natural-duration alignment + post-decode crop (runtime
fix, smallest surface)**:
- Skip the duration scaling when `natural_sum ≤ target_asr_frames`.
  Keep the natural-rounded durations.
- Build alignment up to `pos = sum(natural_durations)`; frames
  `[pos, target_asr_frames)` get all-zero alignment → asr is
  zero past `pos`.
- The decoder synthesises audio across all 128 frames anyway,
  but with zero feature input past `pos` it produces silence
  / near-zero for the tail (iSTFTNet convolutional path).
- Post-decode, crop the WAV to `pos / target_asr_frames *
  waveform_len`.
- F0 / N: interpolate the natural-length curves to
  `pos / target_asr_frames * target_f0_len` and zero-pad the
  rest.
- Risk: the decoder's zero-input behaviour may not be exactly
  silent — could produce noise. STT auto-validation is the
  acceptance criterion.

**Option B — Re-trace the decoder with dynamic asr frame count
(build-side fix)**:
- Re-trace Kokoro with a stimulus that exercises the asr/F0/N
  frame count as a symbolic dim (similar to seq_len in LLMs).
- Update the stage handler to use natural durations directly.
- Out of P0a runtime scope; named follow-up
  **P-BUILD-KOKORO-DYNAMIC-FRAMES** for the build subtree.

**Option C — Trace-time-larger asr + always-crop**:
- Re-trace at, say, 256 frames asr (large enough for typical
  prompts) and have the stage handler always scale durations
  to fit but use only the natural-duration prefix.
- Same shape-vs-content reasoning as Option A but with more
  headroom.
- Still requires Option A's crop + zero-fill on top.

## Section 5 — Recommendation

Implement **Option A** as the audio-chantier P0a runtime fix
(scope: ~30 lines in `kokoro.py:_scale_kokoro_durations` +
maybe ~20 lines in the post-decode wav crop path). The fix is
hypothesis-testable via STT auto-validation: post-fix
whisper-large should transcribe "Hello world this is a test"
or close. If empirical content is wrong, the bug is elsewhere
(LSTM weights / AdaLN scaling / F0 interpolation) — Option A
narrows the diagnosis to the right surface.

Build-side **P-BUILD-KOKORO-DYNAMIC-FRAMES** (Option B) is the
proper long-term fix and goes to the build chantier backlog.

## Section 6 — Test plan post-fix (STT auto-validation)

For each test prompt:

1. `nbx run --model Kokoro-82M --prompt "<prompt>" --output /tmp/out.wav`
2. `nbx run --model whisper-large --audio /tmp/out.wav` → transcription.
3. Compare transcription to prompt via cosine / edit distance.

Test set:
- "Hello world." (short).
- "Hello world, this is a test." (medium).
- "The quick brown fox jumps over the lazy dog." (long).

Acceptance: each transcription should contain ≥ 80 % of the
prompt words (or close homophones). Hocine listens only if the
transcription is partial or ambiguous.

## Section 7 — Latent

The Kokoro stage handler at `kokoro.py:160` masks the LSTM
output via `text_mask.unsqueeze(1).to(device)` but doesn't mask
the asr beyond `pos` — that's the operational expression of the
stretching bug.

This file documents the diagnosis. The implementation lands in
a follow-up commit with the runtime fix (Option A) + unit test.
