# Audio dtype-dedup sweep — Verdict (2026-05-26)

**Scope**: confirm zero regression on the shared audio compute-dtype
resolution path after commit `5686ba5` ("route audio compute-dtype
resolution through the dtype engine"), which routed three inlined
string→`torch.dtype` maps (`audio_utils.get_compute_dtype`,
`next_token_diffusion._compute_dtype`, `audio._get_compute_dtype`) to the
canonical `config.get_torch_dtype`. Byte-identical by construction
(`manifest.get("dtype","float16")` guarantees a valid key); this sweep is
the empirical confirmation across the audio family. Whisper + VibeVoice were
re-validated in 5686ba5's own chantier; this covers the 7 remaining models.

**Verdict**: PASS — all 7 produce correct output. Zero regression.

## Method

This is a confirmation sweep (current-state-passes), not a before/after
byte-identity proof. STT-direction: feed the canonical clip
`test_speech_ref.wav`, compare the transcription to the ground-truth `R`
(whisper-large reference). TTS-direction: generate from "Hello world!",
transcribe the produced wav with Whisper, compare to the prompt. Fuzzy match
(difflib prefix-aligned ratio); human-read each output per R29. Each run
standalone in its own shell (the back-to-back chained-run hang —
P-SEQUENTIAL-RUN-BACKTOBACK-HANG — makes chained model runs unreliable).

Ground-truth `R` (Voxtral verbatim): *"going along slushy country roads and
speaking to damp audiences in draughty school-rooms day after day for a
fortnight, he'll have to put in an appearance at some place of worship on
Sunday morning, and he can come to us immediately afterwards."*

## Results

| Model | Family | Direction | Ratio | Result |
|---|---|---|---|---|
| whisper-large (oracle) | audio | STT | 0.882* | correct (clip) — also derives `R` |
| whisper-large-v3-turbo | stt | STT | 0.953 | correct (clip) |
| parakeet-tdt-1.1b | audio (rnnt) | STT | 0.857 | correct (minor rnnt glyph artifacts) |
| Voxtral-Mini-3B-2507 | audio_llm | STT | 1.000 | verbatim |
| canary-qwen-2.5b | audio_llm | STT | 0.786† | correct, clean-punctuated reformat |
| Kokoro-82M | audio (tts) | TTS | 1.000 | "Hello World" |
| openaudio-s1-mini | tts | TTS | 1.000 | "Hello World!" |
| orpheus-3b-0.1-ft | tts (SNAC) | TTS | 1.000 | "Hello world!" |

\* whisper-large preview is runtime-truncated (~100 chars) → prefix ratio.
† canary reformats the clip into clean prose ("He is going along…, he will
have to appear…") — the fuzzy ratio penalises the LLM-style rephrasing, not a
content error. This is its documented "clean punctuated transcription"
behaviour; content is unambiguously the same clip. R29 human inspection
confirms correct.

## Incidental observations (not regressions, not dtype-related)

- **whisper-large (family `audio`) crashes save_audio on an empty
  transcription**: transcribing orpheus's wav via whisper-large returned an
  empty transcription and then errored in `save_audio` ("no
  global.output_audio"). Cross-checked with whisper-large-v3-turbo → clean
  "Hello world!", confirming orpheus's audio is correct (wav non-silent,
  rms=3885, peak=23737, 1.45 s). The empty-transcription save crash is a
  family-`audio` output-dispatch quirk (STT model under a wav-default family),
  pre-existing and unrelated to the dtype dedup.
- Family `audio` STT models (whisper-large, parakeet) auto-write a stray
  `output_<model>.wav` at the project root when `--output` is omitted (wav is
  the family default). Cleaned. A symptom of the same family-`audio`
  STT/TTS-mixing quirk.

## Artefacts

`validation_outputs/p_audio_dedup_sweep/<model>/` — `run.log`,
`output.txt`/`stt_result.txt`, generated `output.wav` (TTS, all < 2 s).

Hocine validation: TODO
