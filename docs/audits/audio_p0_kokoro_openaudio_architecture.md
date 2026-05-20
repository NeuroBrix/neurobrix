# Audio chantier P0 — Kokoro & openaudio architecture prep (2026-05-20)

Read-only prep for the upcoming audio-family chantier's Layer P0
(quality unblock for Kokoro phonemes + openaudio carrier tone).
Component op-counts captured from each model's cached
`components/<comp>/graph.json`.

## Kokoro-82M (StyleTTS2-class iSTFTNet vocoder)

5 components per the cached topology, flow=audio:

| component | ops | top-5 op_types |
|---|---:|---|
| `bert` | 561 | view(194), t(74), addmm(74), add(50), transpose(48) |
| `bert_encoder` | 4 | view(2), t(1), addmm(1) |
| `text_encoder` | 63 | transpose(10), masked_fill(5), slice(5), index_select(4), _weight_norm_interface(3) |
| `predictor` | 231 | transpose(42), slice(20), index_select(16), _to_copy(12), zeros(12) |
| `decoder` | **1347** | mul(221), add(203), view(178), repeat(116), convolution(72) |

`defaults.json` carries `phoneme_vocab` + `phoneme_lang` — the
phoneme path is part of the model identity.

Pipeline (per StyleTTS2 paper class):

  text → phonemizer → phoneme tokens
                                    ↘
  text → bert → bert_encoder → style features
                                    ↘
  phoneme tokens → text_encoder → phoneme embeddings
                                    ↘
  (style + phoneme embeds) → predictor → duration / F0 / style
                                                                    ↘
  (phoneme embeds + duration + F0 + style) → decoder (iSTFTNet) → waveform

**Symptom**: "babbling phonemes" — speech-like spectral signature
(ZCR 0.134 ≈ reference 0.150, centroid 2080 Hz ≈ reference 2577 Hz)
but unintelligible content. Whisper-large STT auto-transcribes
to empty string.

**Hypothesis classes** for Layer-P0 op-by-op diff:
1. Phoneme tokenisation produces correct IDs but the
   `phoneme_vocab` mapping at runtime is mismatched (off-by-one
   on special tokens / language-code prefix / boundary tokens).
2. Style embedding is unset or set wrong (Kokoro carries
   per-voice style vectors — `phoneme_lang` selects which).
3. Duration predictor outputs are wrong → phoneme expansion
   produces wrong frame alignment.
4. iSTFTNet decoder runs correctly but its conditioning input
   is degenerate (zero, constant, or wrong-shape).

**Investigation plan**:
- Tap each component output via the `NBX_DUMP_TIDS` per-op
  dump (CLAUDE.md §8 retained diagnostic).
- Compare runtime tensor stats at the boundaries (bert output,
  text_encoder output, predictor outputs for duration/F0/style,
  decoder input) vs the vendor reference using the same input
  phoneme sequence.
- The decoder's 116 `aten::repeat` ops are the typical
  iSTFTNet upsample / frame expansion pattern — a wrong
  duration prediction propagates through these directly.

## openaudio-s1-mini (Fish-Speech / OpenAudio S1 codec-LM class)

4 components per the cached topology, flow=dual_ar:

| component | ops | top-5 op_types |
|---|---:|---|
| `codec.encoder` | 783 | mul(129), view(127), add(96), select(64), constant_pad_nd(34) |
| `codec.quantizer` | **2206** | mul(322), view(310), select(256), add(157), t(108) |
| `codec.decoder` | 368 | add(70), view(58), mul(58), _weight_norm_interface(30), convolution(30) |
| `model` | **3116** | view(589), select(460), mul(451), add(208), _unsafe_view(197) |

Architecture pattern: a DAC/SNAC-class neural audio codec
(encoder ↔ quantizer ↔ decoder) plus an autoregressive
language-modelling head over the discrete codec codes.

Pipeline:

  text + reference voice → model (autoregressive) → audio codes
                                                                    ↘
  audio codes → codec.decoder (30 convolution + 30 weight_norm
                ops) → waveform

**Symptom**: quiet ~689 Hz carrier tone, peak amplitude 1 % of
full scale, ZCR 0.029. Whisper-large STT auto-transcribes to
empty string. WAV is 23.8 s — the model generates tokens for a
long time but the decoded output is near-silent with a residual
sinusoid.

**Hypothesis classes**:
1. The autoregressive `model` produces a degenerate code sequence
   (all the same code, or all silence/EOS codes, or quasi-random
   codes that the codec.decoder maps to near-zero).
2. The reference voice conditioning is not reaching the model
   (the harness passes `--audio` which routes to
   `inputs["global.audio_path"]`; needs verification that
   the codec.encoder consumes it for voice-style extraction).
3. codec.decoder runs but with wrong weights / wrong dtype path,
   producing the 689 Hz carrier as a kind of "no-input ringing"
   (its 30 convolution ops with `_weight_norm_interface` form
   the upsample stack; corrupted weight-norm could produce this).
4. The dual_ar flow handler doesn't route the codec.decoder
   output correctly — the WAV may be reading from a
   wrong slot (silence buffer vs the actual decoded audio).

**Investigation plan**:
- Inspect the autoregressive token sequence produced by `model`
  (dump `global.generated_token_ids` and look for code
  diversity / sane distribution).
- If codes look reasonable, dump `codec.decoder` input and
  output tensor stats; compare to the vendor reference for the
  same code sequence.
- If codes look degenerate, walk back into `model` — likely a
  reference-voice conditioning issue.

## R16 — external references (read in the chantier)

- StyleTTS2 paper (NeurIPS 2023) — Kokoro-82M is built on this
  architecture. The phoneme + style + duration + iSTFTNet
  arrangement is documented there.
- Kokoro official repo (HuggingFace `hexgrad/Kokoro-82M`) —
  reference implementation for the runtime expected input shapes
  and the phoneme vocabulary table.
- Fish-Speech / OpenAudio S1 repo — the dual_ar codec-LM
  architecture, reference voice conditioning protocol, and the
  expected token-distribution diagnostics.
- DAC / SNAC papers — the codec.decoder class.

## Cross-cutting

Both bugs are pre-Ch7/Ch8 — the audio decoders were never
content-validated in any prior chantier. Ch3 P-KOKORO closure
fixed the **crash** path (cudnn_batch_norm undefined-tensor), not
the **content quality**. The R29 audio rule (audio-listening
mandatory, not just exit-0) postdates that closure; this
audit-prep is the doc that frames the audio chantier's P0
catch-up.

STT auto-validation (whisper-large transcription) is the
correctness oracle for the fix. Target post-fix: whisper-large
transcribes the prompt back (or close enough — TTS-roundtrip
quality on "Hello world" should give "hello world" or a very
near homophone).
