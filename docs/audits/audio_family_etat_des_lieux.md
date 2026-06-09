# Audio family — état des lieux factuel (2026-05-20)

Factual inventory per the mandate. Built post-Dette-F closure as
the basis for the upcoming audio-family chantier. STT
auto-validation (whisper-large transcription on cached outputs)
applied per the new collaboration rule.

## Section 1 — Table: 11 cached audio models

Legend
- **Cached**: dir present at `~/.neurobrix/cache/<model>/`.
- **Traced**: implicit-yes when cached (the cache is the
  post-trace artefact).
- **Hub published**: cross-ref with the 15-model Prisma DB
  inventory on `10.0.0.39` (Dette G Section 1).
- **Quality**: validated R29 / cassé+bug nommé / non testé.
  TTS quality = STT-auto transcription via whisper-large; STT
  quality = transcription vs known reference; audio_llm
  quality = manifest-prompted output coherence.
- **Modes** = harness Ch8 results after Dette B updates
  (PASSED / XFAIL / FAILED).
- **Blockers**: named follow-ups from `docs/follow-ups/INDEX.md`.

### TTS sub-family

| model | cached | hub published | quality | native | triton | blockers |
|---|---|---|---|---|---|---|
| **Kokoro-82M** | YES | NO | **CASSÉ** — whisper-large transcribes EMPTY string on the cached output (audio is babbling phonemes, not speech). | PASSED | XFAIL (`_execute_native_text_encoder` NBXTensor→F.embedding boundary) | **P-AUDIO-KOKORO-PHONEMES** (P0). Ch3 fixed crash; content quality never R29-validated. |
| **VibeVoice-1.5B** | YES | NO | non testé (XFAIL both modes; harness CLI profile doesn't match the speaker-script format). | XFAIL | XFAIL | TensorDAG contract violation — DDPM + ConvNext1d outside graph; needs build-side re-trace. |
| **chatterbox** | YES | **YES** (`resemble-ai/Chatterbox`) | **CASSÉ** — whisper-large transcribes EMPTY on the 82 s saturated output. | PASSED | XFAIL (triton tts_llm doesn't wire `audio_path` reference voice) | **P-AUDIO-CHATTERBOX-LOOP** (P1) — stop-token / max-length logic. + reference-voice triton plumbing. |
| **orpheus-3b-0.1-ft** | YES | **YES** (`canopylabs/Orpheus-3B`) | **CASSÉ** — pre-existing FAIL both modes in Ch8 harness (declared xfail in Dette B). | FAILED→XFAIL | FAILED→XFAIL | audio-chantier follow-up (no specific name yet — diagnose first). |
| **openaudio-s1-mini** | YES | NO | **CASSÉ** — whisper-large transcribes EMPTY on the 23.8 s 689 Hz carrier-tone output (peak amplitude 1%). | PASSED | XFAIL (triton dual_ar doesn't wire reference voice) | **P-AUDIO-OPENAUDIO-CARRIER-TONE** (P0) — codec.decoder mel→waveform suspect. + reference-voice triton plumbing. |

### STT sub-family

| model | cached | hub published | quality | native | triton | blockers |
|---|---|---|---|---|---|---|
| **whisper-large** | YES | NO | **OK** — transcribes `test_speech_ref.wav` as `going along slushy country roads and speaking to damp audiences in drafty school rooms day after da…` (matches expected English speech). | PASSED | XFAIL (triton encoder_decoder audio flow not validated end-to-end) | triton encoder_decoder flow port. |
| **whisper-large-v3-turbo** | YES | **YES** (`openai/Whisper-V3-Turbo`) | **OK** — transcribes same reference as `going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after d…` (slight word-variant difference vs whisper-large, semantically equivalent). | PASSED | XFAIL (same triton encoder_decoder flow port) | triton encoder_decoder flow port. |
| **parakeet-tdt-1.1b** | YES | NO | **PARTIAL** — encoder runs, output shape `[1, 1024, 375]`, transcription header printed but actual text content TBD (output truncated by grep in initial probe; needs a clean re-run for confirmation). | PASSED | XFAIL (triton rnnt flow not validated end-to-end) | rnnt triton flow port. |

### audio_llm sub-family

| model | cached | hub published | quality | native | triton | blockers |
|---|---|---|---|---|---|---|
| **Voxtral-Mini-3B-2507** | YES | **YES** (`mistralai/Voxtral-Mini-3B`) | **OK runtime / CASSÉ behaviour** — Ch5 byte-identical compiled↔triton transcription (string-identical 112 B). Content is conversational ("I'm sorry, I didn't quite catch that…") rather than transcription — the model answers as a chat partner. Build-side processor wiring. | PASSED | PASSED | **P-VOXTRAL-HALLUCINATION** (P2) — build-side multimodal-processor wiring. **P-TRITON-PERF-AUDIO-LLM** (P2) — triton decode 50× slower than compiled. |
| **canary-qwen-2.5b** | YES | NO | non-fully-tested (initial probe started — token generation began; full output capture pending). Ch5 era output was `<think>` (model-side processor pattern). | PASSED | XFAIL | **P-TRITON-NBXTENSOR-REPEAT-MISSING** (P1) — encoder hits `aten::repeat` → `NBXTensor.repeat` missing (`kernels/dispatch.py:161`). |
| **granite-speech-3.3-8b** | YES | NO | **CASSÉ** — output is literal `erer` (4 tokens in 3.4 s). Same pattern across Ch5 + Dette E + état des lieux probe. Build-side processor wiring or model loading issue. | PASSED (exit 0 but content wrong) | XFAIL | **P-TRITON-SAFE-SOFTMAX-MISSING** (P1) — triton compile aborts on `aten::_safe_softmax`. Plus a build-side processor issue producing the `erer` output independent of the triton block. |

## Section 2 — Aggregate state

11 cached models / 4 hub-published (Chatterbox, Orpheus, Voxtral,
Whisper-V3-Turbo). Of those 4 published:
- 2 are quality-CASSÉ (Chatterbox loop, Orpheus harness-fail).
- 1 is runtime-OK but build-quality-CASSÉ (Voxtral chats instead
  of transcribing — Hocine flagged in Ch5).
- 1 is OK end-to-end (Whisper-V3-Turbo).

Of the 7 NOT-published:
- Kokoro: CASSÉ phonemes (publishing blocked by audio chantier).
- VibeVoice: structural TensorDAG contract violation (publishing
  requires build-side re-trace).
- openaudio: CASSÉ carrier tone (publishing blocked by audio
  chantier).
- whisper-large: OK runtime; not yet published.
- parakeet: needs full transcription confirmation; not yet
  published.
- canary-qwen / granite-speech: triton kernel-op gaps;
  granite has build-side `erer` issue.

## Section 3 — Audio-chantier scope (the actual work)

Per the mandate, this état des lieux feeds the dedicated audio
chantier. The named bugs that require fixing inside it:

**P0 (blocks broad TTS publication)**:
- P-AUDIO-OPENAUDIO-CARRIER-TONE — codec.decoder vocoder.
- P-AUDIO-KOKORO-PHONEMES — phoneme/style condition path.

**P1**:
- P-AUDIO-CHATTERBOX-LOOP — tts_llm stop-token.
- P-TRITON-NBXTENSOR-REPEAT-MISSING — canary-qwen triton.
- P-TRITON-SAFE-SOFTMAX-MISSING — granite triton + audio
  encoder_decoder / rnnt triton flow ports.

**P2 (already-known, separate workstream)**:
- P-VOXTRAL-HALLUCINATION — build-side processor wiring.
- P-TRITON-PERF-AUDIO-LLM — 50× decode slowdown.

**Coverage-future**:
- VibeVoice TensorDAG contract violation — build-side re-trace.

## Section 4 — Recommended next chantier framing

The audio chantier opens with a four-layer scope:

1. **TTS quality unblock** (P-AUDIO-* P0) — diagnose Kokoro and
   openaudio at the vocoder stage. Likely 1-2 op-by-op
   differentials each, against the model-vendor reference
   implementation. The STT-auto validation rule confirms whether
   the fix lands (whisper-large transcription should match the
   prompt).
2. **TTS stop-token fix** (P-AUDIO-CHATTERBOX-LOOP) — locate the
   length cap in `core/flow/tts_llm.py`.
3. **STT + audio_llm triton coverage** (P-TRITON-NBXTENSOR-REPEAT,
   P-TRITON-SAFE-SOFTMAX, triton encoder_decoder, triton rnnt) —
   kernel-op + flow ports.
4. **Build-side hygiene** (P-VOXTRAL-HALLUCINATION,
   P-CHATTERBOX-DECODING-CHARABIA, granite `erer`) — out of
   runtime scope; flag for the build-side chantier that owns
   model processor wiring.

The harness now (post-Dette-B) gives the audio chantier a clean
xfail/pass profile to bisect against. STT-auto-validation gives
the closing R29 without escalating each transcription to Hocine.
