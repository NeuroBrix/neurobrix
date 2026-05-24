# Audio Family P0 — Consolidated Verdict (2026-05-24)

Scope: bring the entire audio family to functional + STT-validated, or park
genuine missing capabilities as named chantiers. 11 models.

## Final tally — 8 functional/STT-validated, 3 parked capabilities

### Functional + STT-validated (8)
| Model | Family | Validation (generate → whisper-large STT → expected text) |
|---|---|---|
| whisper-large | stt | reference transcriber (used to validate every TTS model) |
| whisper-large-v3-turbo | stt | transcribes the canonical ref clip ✓ |
| Voxtral-Mini-3B-2507 | audio_llm | verbatim transcription ✓ |
| canary-qwen-2.5b | audio_llm | clean punctuated transcription ✓ (re-confirmed this session) |
| parakeet-tdt-1.1b | stt (rnnt) | transcribes the ref clip ✓ |
| openaudio-s1-mini | tts (dual-AR) | "Hello world!" ✓ |
| Kokoro-82M | tts | "Hello world." (1.00) ✓ |
| orpheus-3b-0.1-ft | tts (SNAC) | **"Hello World!" ✓ (CLOSED this session)** |

### Parked capability chantiers (3) — missing capabilities, NOT finishable bugs
- **chatterbox** — conditioning RESOLVED (runs the embedded conditioning-encoder
  component incl. the Perceiver resampler → vendor-exact [1,34,1024]; the speech
  LM no longer over-generates). Vocoder blocked on **P-SYMBOLIC-ITEM-TRACKING**:
  its pad mask is `arange(token_len.max().item())` where `token_len` is a VALUE
  sum of the `*_len` inputs; symbolizing it needs value-symbolic tracking
  (value↔shape correspondence carried through `.item()`), a matrix-wide build
  capability. Rejected shortcuts: re-enabling the deprecated expression-value
  match (false-match risk), or a runtime mask re-derivation (fragilises the
  general mechanism).
- **granite-speech-3.3-8b** — **P-CEIL-PAD-WINDOW**: Q-Former window projector
  needs `ceil(seq/W)` + dynamic pad-to-multiple; a trace-time pad is baked.
- **VibeVoice-1.5B** — **P-VIBEVOICE-NEXT-TOKEN-DIFFUSION-FLOW**: needs a complete
  next-token-diffusion generation flow (LatentLM-style); the local vendor ships
  only components, no generation-loop oracle.

## Bugs fixed this chantier — root causes

### orpheus (5-fix chain, was non-functional → "Hello World!")
1. **LM head config never emitted** — the build applied the registry flow
   overlay (which refines a coarse trace flow label to autoregressive_generation)
   AFTER defaults generation, so `lm_config` (num_heads/head_dim/num_layers) was
   skipped and the KV cache allocated a zero head_dim buffer. Fix: overlay before
   defaults gen.
2. **Two-arg `arange(0, seq_len)` decode shift** — the KV-cache decode position
   shift assumed `arange(end)`; the two-arg form gave `arange(cache_len,
   cache_len)` = empty range at decode.
3. **`expand`-size promotion of the wrapped-list form** — the symbolic promotion
   unwrapped the `{type:list,value:[…]}` size for `view`/`reshape` but not
   `expand`, so the RoPE position-expand kept its trace seq_len literal and
   broadcast the single decode position to 23. THE root of the RoPE degeneracy.
4. **SNAC token de-interleaving** — the 7-per-frame hierarchical layout
   (L0:0, L1:1,4, L2:2,3,5,6) was split contiguously → noise. The breakthrough to
   intelligible speech.
5. **Speech-end EOS** — orpheus marks end-of-speech with 128258, not the Llama
   text EOS 128001; the loop ran to max_tokens (13.4 s ramble). Registry now sets
   the stop, and registry generation params override config-derived defaults.

### chatterbox conditioning (was 43 s of garbage → vendor-exact conditioning)
The runtime hand-rolled T3CondEnc and skipped the Perceiver (2-token conditioning
vs the vendor's 34). Re-architected to run the embedded conditioning-encoder
component data-driven; the speech LM stops over-generating (1052+ → ~24-48 tokens).

## Shared-primitive fixes — R23 non-regression proven

- **SDPA cross-attention K/V transpose detection** — keyed the "is K transposed"
  test on the head-dim axis instead of the seq axis. Self-attention is
  byte-identical; only the previously-crashing cross-attention case (seq_q≠seq_k)
  changes. Proven: TinyLlama (self-attn) coherent & unchanged; whisper-large
  (encoder-decoder cross-attn) transcribes correctly; canary-qwen (audio_llm)
  verbatim; chatterbox Perceiver unblocked. No triton mirror needed (the triton
  path has no equivalent reassembly heuristic).
- **Two-arg arange decode shift** + **wrapped-list expand-size promotion** —
  both fix previously-broken cases only; single-arg / raw-list paths unchanged.
  Validated: TinyLlama coherent; the promotion path runs only for seq_len models
  (diffusion/image skip it).
- (prior session) **DtypeEngine fp16 square-mul guard** — RMSNorm `x*x` overflow
  on fp16 hardware, R23 byte-identical on TinyLlama/Sana.

## Perf / capability follow-ups (named, NOT in this loop)
- **P-SYMBOLIC-ITEM-TRACKING** — chatterbox vocoder (above).
- **P-CEIL-PAD-WINDOW** — granite (above).
- **P-VIBEVOICE-NEXT-TOKEN-DIFFUSION-FLOW** — VibeVoice (above).
- **P-DUALAR-KVCACHE-3D** — dual-AR (openaudio) KV-cache 3D layout perf.
- **P-BUILD-KOKORO-DYNAMIC-FRAMES** — Kokoro symbolic `asr_frames`/phoneme
  seq_len; removes the fixed 128-frame window, the 23-phoneme truncation, and
  unblocks Kokoro `--triton`.
- **P-DTYPE-CENTRALIZATION** — push remaining local dtype handling into the
  DtypeEngine.
- **P-TRITON-PERF-AUDIO-LLM** — triton audio_llm decode ~50× slower than compiled.

## Validation method
Per model: generate → whisper-large STT → compare to expected text; verify
symbolic shapes, standardized weight keys, and DtypeEngine applied. Artefacts
under `validation_outputs/audio_family/`. Hocine audio validation: TODO.
