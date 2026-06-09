# Dette E — TTS audio quality diagnosis (2026-05-20)

Branch `p-debt-settlement-batch-1`. State at start: Hocine flagged
Kokoro and openaudio-s1-mini R29 outputs as "incohérent" (random
sounds) and "TV test tone" (continuous tiiit). Audio quality is
not validated by exit-0 alone; R29 audio listening is the
discriminator.

## Section 1 — Spectral diagnosis (the differential)

Per-model audio stats vs natural-speech baseline:

| model | dur | rms | peak | ZCR | dom_freq | diagnosis |
|---|---|---|---|---|---|---|
| reference_speech.wav | 13.7 s | 0.055 | 0.46 | 0.150 | 96 Hz | baseline speech |
| Kokoro-82M | 3.2 s | 0.029 | 0.25 | 0.134 | 176 Hz | speech-LIKE spectral signature (ZCR ≈ ref, centroid ≈ ref), but Hocine confirmed unintelligible — model produces babbling phonemes |
| openaudio-s1-mini | 23.8 s | 0.006 | **0.01** | **0.029** | **689 Hz** | **quiet carrier tone, near-silent.** Peak 1% of full scale, ZCR 0.029 (5× lower than speech), dominant 689 Hz steady — the documented "TV mire tone" symptom |
| chatterbox | **82.3 s** | 0.085 | **0.99** | 0.062 | 213 Hz | 82 s output for short "Hello world" prompt, peak saturated to full scale — model runs unconstrained, possibly looping |

These are NOT regressions introduced by Ch7 or Ch8. The
fundamental defects predate both chantiers; what Ch7/Ch8 did was
clean up enough of the runtime that the R29 audio-listening
discipline now reveals them. Per Hocine's recadrage in the
mandate: "en enlevant les béquilles, on expose les vrais
problèmes sous-jacents qui étaient masqués". The "PASS sur stats
sans inspection sémantique" R29 case is precisely this.

## Section 2 — Per-model named follow-ups (the disposition)

Each model is a distinct defect class — different stages of the
TTS pipeline, different symptoms. Properly diagnosing and fixing
each requires the **dedicated audio-family chantier** Hocine
names at the end of the mandate ("ouvre le chantier de la famille
AUDIO"). Dette E's deliverable is **diagnosis + named follow-ups
+ R29 audio artefacts ready for that chantier**, not a fix in
Dette E itself.

- **P-AUDIO-OPENAUDIO-CARRIER-TONE** — openaudio-s1-mini codec
  decode produces a quiet ~689 Hz tone instead of speech. Quiet
  amplitude (peak 1%) and very low ZCR (0.029) are the signature
  of a near-pure sinusoid at the decode output. Investigation
  scope: the `codec.encoder → codec.quantizer → codec.decoder`
  pipeline (each is a separate component per the cached topology);
  the decoder's mel-spectrogram → waveform stage is the most
  likely site (vocoder kernel producing constant output).
- **P-AUDIO-KOKORO-PHONEMES** — Kokoro-82M produces audio with
  speech-like spectral characteristics (ZCR, centroid match
  reference within 10%) but is not intelligible. The acoustic
  envelope is plausible; the linguistic content is wrong.
  Investigation scope: text_encoder output → predictor
  (style + pitch) → decoder. Hypothesis: missing speaker /
  style condition in the runtime invocation chain, OR the
  phoneme-conditioning path is mis-wired post-Ch3
  cudnn-batch-norm fix. Note: Ch3 P-KOKORO-NATIVE-CUDNN-BATCH-NORM
  unblocked the **crash**, not the **content quality** — Ch3
  R29 audio inspection was missing.
- **P-AUDIO-CHATTERBOX-LOOP** — chatterbox produces 82 s of
  saturated audio for a short "Hello world" prompt. Length is
  unconstrained and the output peaks at full scale (clipping).
  Investigation scope: `tts_llm` flow stop-token / max-length
  logic; the decoder may be running until a hardcoded ceiling
  instead of stopping at the natural end-of-speech token.
- **P-HARNESS-STT-WAV-EXTENSION** — STT models (whisper-large,
  parakeet-tdt) write malformed `.wav` extension files because
  the CLI's family-aware output dispatch picks `.wav` for any
  `family=audio` model even when the actual output is a text
  transcription. Same root cause as the previously-named
  `P-HARNESS-AUDIO-LLM-OUTPUT-DISPATCH` follow-up; this is the
  STT branch of the same family-dispatch defect.

## Section 3 — Validation (Hocine-gated)

R29 audio artefacts: `validation_outputs/p_dette_e_tts_audio_diagnosis/`:

- `kokoro_random_sounds.wav` — listen to confirm babbling phonemes.
- `openaudio_carrier_tone.wav` — listen to confirm "tiiiit" tone.
- `chatterbox_82s_loop.wav` — listen to confirm 82 s saturated loop.
- `reference_speech.wav` — natural-speech baseline.
- `spectral_stats.json` — machine-readable summary.
- `INDEX.md` — full per-model discriminator table.

Hocine validation: **TODO**. Confirm the three symptom classes
match Hocine's pre-Dette-E observations (Kokoro random sounds,
openaudio tone, chatterbox loop). Once confirmed, the audio-
family chantier opens with these 4 named follow-ups as its
backlog.

## Section 4 — Why this is not a fix-in-Dette-E

Each of the three TTS bugs is a different vocoder/decoder stage
in a different model architecture. A proper fix requires:
(a) Op-by-op trace of the failing component to locate the
exact site producing the wrong output. (b) Comparison vs the
upstream vendor's reference implementation. (c) Validated R29
post-fix listening. Each model is plausibly multi-hour. The
audio-family chantier is the right container for that work — not
this debt-settlement pass.

The harness change from Dette C (IMAGE_PROMPT for image/video)
does not propagate to TTS: TTS prompt content is not the
discriminator (Kokoro babbles whatever text it gets;
openaudio's tone is amplitude-near-zero regardless of prompt).
The discriminator is the model-side acoustic decoder, not the
input.

## Section 5 — Latent observations

- The Ch3 P-KOKORO-NATIVE-CUDNN-BATCH-NORM closure verified
  Kokoro's `native` path no longer crashes on `aten::cudnn_batch_norm`
  but did NOT validate the audio content semantically (no R29
  audio inspection in that verdict's evidence). This is the
  canonical "PASS sur stats sans inspection sémantique" R29
  case the memory rule was added to prevent. Future TTS
  chantiers must ship R29 audio.
- The CLI's `--reference-audio` flag is dead code in `cmd_run`
  (the value is never consumed; only `--audio` is routed via
  `inputs["global.audio_path"]`). The harness uses `--audio`
  for TTS-with-ref flows (chatterbox, openaudio), so the
  reference DOES reach the model in the harness path. Manual
  CLI users who invoke `--reference-audio` get a silent no-op.
  Named follow-up:
  **P-CLI-DEAD-REFERENCE-AUDIO-FLAG** — either consume the
  flag (alias to `--audio` with TTS-with-ref family) or remove
  the dead flag from the argparse surface.
