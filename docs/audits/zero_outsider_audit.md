# Zero Outsider — full runtime vendor-dependency audit (2026-06-08)

## Resolution status (2026-06-08)

| phase | scope | status |
|-------|-------|--------|
| ZO-1 | pure-Python tokenizer runners (drop tokenizers/sentencepiece/tiktoken/mistral_common) | **DONE** — byte-exact vs vendor, clean-room green |
| ZO-2 | compiled mel via shared numpy `mel_dsp` (drop transformers WhisperFeatureExtractor/torchaudio) | **DONE** |
| ZO-4 | numpy mel filterbank (drop librosa) | **DONE** — parity 1.86e-9..0.0 |
| RNNT | parakeet flow mel+decode (drop torchaudio/sentencepiece) | **DONE** |
| ZO-5 | image processors (Janus/swin2SR) | **N/A** — `preprocessor_config.json` read as JSON, no `AutoImageProcessor` import (already clean) |
| Manifest | split — 14 vendor libs → `[adapter]` extra; runtime deps = torch/numpy/infra/IO | **DONE** |
| ZO-0 | orpheus SNAC decoder into the `.nbx` | **COMPILED DONE** — SNAC traced into a new `orpheus-3b-0.1-ft-snac` build as a `codec.decoder` component (NoiseBlock `randn`→`randn_like` so the noise length stays symbolic); the flow redistributes the 7-tokens/frame stream and runs the traced codec — no `snac`, no HF download. Clean-room compiled → STT "Hello world!". **Triton remaining**: a triton weight-load issue on the model prefill of the 3-component build (the ZO-0 codec mechanism is proven in compiled). Production `orpheus-3b-0.1-ft` untouched. |
| ZO-3 | Kokoro g2p internal/embedded (drop phonemizer/espeakng_loader) | **OPEN — escalated** — faithful g2p means replacing espeak (compiled binary dict + rule engine; Kokoro trained on espeak IPA) AND espeak-ng is GPL-3.0 (distilling its data into this Apache-2.0 repo is a maintainer licensing decision). The clean-room raises a clean ZERO-TORCH-g2p error rather than silently using torch. |

`TokenizerFactory._load_tokenizer` (`factory.py`) still references `transformers`
but is **dead at runtime** (not called by core/cli/serving/triton — the runtime
uses `load_tokenizer_from_path`); its lazy import never fires in the clean room.

Outcome: ~28/30 models run from their `.nbx` in the clean venv (torch/triton only,
none of the 14) across compiled + triton. Residuals = orpheus (SNAC, build-side)
and Kokoro (g2p, GPL/espeak). Full per-model matrix:
`validation_outputs/zo/matrix/RESULTS.txt`; bilan: `validation_outputs/zo/BILAN.md`.

---

**Goal (R34):** at inference, the NeuroBrix engine depends **only on the `.nbx`**.
Every model component — backbone, **frontend** (tokenizer / g2p / mel), and
**decoder** (vocoder / codec / VAE) — is either present in the `.nbx` graph as
ATen ops or implemented as a NeuroBrix-internal class. **No vendor library is
imported to compute or to load a model.**

Definitions used below:
- **frontend** = input pre-processing (text→ids = tokenizer; text→phonemes = g2p;
  audio→mel = feature extractor).
- **decoder** = output post-processing that is itself a neural net (vocoder /
  codec / VAE).
- **backbone** = the network present in the `.nbx`.

## Exhaustive finding: only **3** `from_pretrained` exist in the whole runtime

```
core/module/tokenizer/factory.py:269   tokenizer_cls.from_pretrained(tokenizer_path)        # LOCAL path (.nbx), via transformers class
core/module/audio/input_processor.py:135  WhisperFeatureExtractor.from_pretrained(model_path) # LOCAL config, via transformers (compiled mel)
core/module/audio/output_processor.py:117 snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")# HF REPO ID -> DOWNLOADS, decoder NOT in .nbx
```

Full external-import surface of `core/` + `triton/` (excluding stdlib / torch /
triton / numpy / neurobrix / obsolete `triton_kernels_ref/`):

| import | role | verdict |
|--------|------|---------|
| `transformers`, `tokenizers`, `sentencepiece`, `tiktoken`, `mistral_common` | tokenizer runners | **VIOLATION** (frontend) |
| `phonemizer`, `espeakng_loader`, `kokoro` | g2p | **VIOLATION** (frontend) |
| `torchaudio`, `librosa` | mel / resample / filterbank | **VIOLATION** (frontend DSP) |
| `snac` | RVQ codec decoder | **VIOLATION** (decoder, + HF download) |
| `safetensors` | reads the `.nbx` weight shards | infra — reads the `.nbx` itself, accepted (could be internalised, low prio) |
| `yaml`, `pydantic` | config / schema parsing | infra — not a model, accepted |
| `jinja2` | chat-template rendering (prompt string) | infra — text templating, accepted |
| `soundfile`, `wave`, `PIL`, `imageio_ffmpeg` | media file read/write at the CLI boundary | boundary I/O — accepted |

(The `from_pretrained` set is provably complete — the grep above is exhaustive.)

## Per-model pollution matrix (all 30 cached builds)

Legend: **T** = tokenizer frontend (`transformers`/`sentencepiece`/`tiktoken`/
`mistral_common`), **M** = mel frontend (compiled: `transformers`
WhisperFeatureExtractor + `torchaudio`/`librosa`; triton already numpy-internal),
**G** = g2p frontend (`phonemizer`/`espeakng_loader`/`kokoro`), **S** = untraced
**decoder** run from a vendor lib (`snac`, + HF download), **OK** = clean.

| family | model | components in the `.nbx` | pollutants |
|--------|-------|-------------------|------------|
| LLM | TinyLlama-1.1B | lm_head, model | **T** |
| LLM | deepseek-moe-16b | lm_head, model | **T** |
| LLM (MoE) | Qwen3-30B-A3B | lm_head, model | **T** |
| image | Sana-1024 / Sana-4Kpx | text_encoder, transformer, **vae** | **T** (vae in `.nbx` OK) |
| image | PixArt-XL / PixArt-Sigma | text_encoder, transformer, **vae** | **T** |
| image | Flex.1-alpha | text_encoder x2, transformer, **vae** | **T** |
| video | SANA-Video-2B | text_encoder, transformer, **vae** | **T** |
| multimodal | Janus-Pro-7B | vision_model, language_model, gen_* | **T** + image processor (verify) |
| STT | whisper-large, whisper-large-v3-turbo | model.encoder, **model.decoder** | **T**, **M** |
| audio_llm | Voxtral-Mini-3B | audio_tower, language_model, projector | **T**, **M** |
| audio_llm | canary-qwen-2.5b | perception, llm, embed_tokens | **T**, **M** |
| audio_llm | granite-speech-3.3-8b | encoder, language_model, projector | **T**, **M** |
| STT (rnnt) | parakeet-tdt-1.1b | encoder, **decoder**, joint | **T**, **M** |
| TTS | Kokoro-82M | bert, text_encoder, predictor, **decoder** | **G** (g2p) |
| TTS | chatterbox | cond_enc, t3_cfg, ve, **s3gen** | **T** (s3gen decoder in `.nbx` OK) |
| TTS | openaudio-s1-mini | model, model.fast, **codec.decoder**, codec.quantizer | **T** (decoder in `.nbx` OK) |
| TTS | VibeVoice-1.5B | language_model, **acoustic_tokenizer**, prediction_head, … | **T** (decoder in `.nbx` OK) |
| TTS | **orpheus-3b** | lm_head, model | **T** + **S** <- only untraced decoder (SNAC, HF download) |
| upscaler | real-esrgan x2/x4/x8 | model | **OK clean** |
| upscaler | swinir-classical x2/x4 | model | **OK clean** |
| upscaler | hat-l/hat-s | model | **OK clean** |
| upscaler | swin2SR x2/x4 (x3) | swin2sr, upsample | processor module (no `from_pretrained` -> verify, likely internal normalisation) |

### Direct answers

- **LLM polluted?** Yes — by the **tokenizer** only (`lm_head`+`model` present; no
  untraced decoder). TinyLlama, deepseek-moe, Qwen3 all hit `transformers.from_pretrained`.
- **Image polluted?** Yes — by the **tokenizer** only. The `text_encoder`,
  `transformer` and **`vae` decoder are all in the `.nbx`**; the only vendor
  dependency is the prompt tokenizer. Same for Flex, PixArt, SANA-Video, Janus.
- **Only untraced DECODER across all 30 models = SNAC (orpheus)** — and it is the
  worst case (downloads from HF at inference).
- **Already Zero-Outsider-clean:** the pure image upscalers (real-esrgan, swinir,
  hat) — image->image, no text, no tokenizer, no decoder import. (swin2SR has a
  `processor` module to confirm.)

## Severity ranking

1. **SNAC / orpheus** — a neural decoder OUTSIDE the `.nbx`, **fetched from the
   internet** at inference. Hardest breach of "depend only on the `.nbx`".
2. **Tokenizer** — widest blast radius (~24 of 30 models, both modes). The data is
   already embedded in `.nbx modules/tokenizer`; only the *runner* is the vendor lib.
3. **mel (compiled)** — 5 audio-input models. The triton side is already
   numpy-internal; compiled still uses `transformers`/`torchaudio`/`librosa`.
4. **g2p (Kokoro)** — 1 model, both modes.
5. **librosa filterbank** — shared math helper used by the mel front-ends.

## Zero Outsider — phased remediation plan

| phase | scope | removes | models unblocked |
|-------|-------|---------|------------------|
| **ZO-0** | put SNAC into the orpheus `.nbx` (or `@triton.jit` RVQ+convT) | `snac` + HF download | orpheus |
| **ZO-1** | NeuroBrix-internal tokenizer runner over the embedded `.nbx` tokenizer data | `transformers`/`sentencepiece`/`tiktoken`/`tokenizers`/`mistral_common` | ~24 models, both modes |
| **ZO-2** | route compiled mel through the NeuroBrix mel class (reuse `audio_frontend` math) | `transformers WhisperFeatureExtractor`, `torchaudio` | whisper x2, voxtral, canary, granite, parakeet (compiled) |
| **ZO-3** | embed g2p in the `.nbx` (like the tokenizer) or NeuroBrix-internal g2p | `phonemizer`/`espeakng_loader`/`kokoro` | Kokoro |
| **ZO-4** | replace `librosa.filters.mel` with a NeuroBrix mel-filterbank function | `librosa` | all mel models, both modes |
| **ZO-5** | verify/internalise the image `processor` (Janus, swin2SR) | image processor lib (if any) | Janus, swin2SR |
| **ZO-6** | decide on infra deps (`safetensors` container reader; `yaml`/`pydantic`/`jinja2`) | optional / keep as infra | — |

**End state:** the engine RUNS any model with only `torch` (compiled) and
`triton`+`NBXTensor` (triton); every frontend datum and every decoder lives in the
`.nbx` or in NeuroBrix code. Order by leverage: **ZO-0 (gravest) -> ZO-1 (widest)**
first.

## Method note

This audit is static: the `from_pretrained` set is closed (3 sites, grep-exhaustive)
and the external-import surface is fully enumerated, cross-referenced against the
`components/` and embedded `modules/` of each build. A runtime import-logging probe
per model would be the gold-standard confirmation and is the first validation step
inside each ZO phase.
