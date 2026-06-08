# R34 — Engine Import Purity Audit (2026-06-08)

## The rule (R34)

The NeuroBrix runtime **engine** (`core/`, `triton/`, `kernels/`) treats every
model component — backbone, encoder, **decoder, codec, vocoder**, and every other
neural module — exactly the way it treats a transformer: the component is either

1. **present in the `.nbx` graph as ATen ops** and executed by the engine, or
2. **implemented as NeuroBrix-internal compute** in a dedicated class/function under
   `core/flow|module` (compiled) or `triton/flow|module` (triton).

The engine **never imports a third-party model / codec / DSP package at runtime**
to perform compute or to load a model. The end state: a NeuroBrix install needs
**only `torch` (mode 1)** and **only `triton` + `NBXTensor` (mode 2)** to RUN any
model. Every `from_pretrained`, every vendor codec, every DSP library is either
baked into the `.nbx` at trace time or re-implemented inside NeuroBrix.

R34 is the generalisation of **R33** (which sealed `triton/` against `torch`) to
**all external imports, in both modes**.

### Allowed runtime imports

| mode | allowed | glue |
|------|---------|------|
| compiled | `torch` (+ the cuDNN/cuBLAS/cuFFT it bridges) | stdlib, file-I/O libs at the boundary |
| triton | `triton` + `NBXTensor` | `numpy` (CPU glue), stdlib, file-I/O libs at the boundary |

Media file-I/O at the CLI boundary (`soundfile` read/write WAV, `Pillow` PNG,
`imageio` MP4) is **boundary I/O, not compute** — accepted, like reading the input
file. The rule targets **neural compute and model loading**, not file I/O.

## Audit — where R34 is violated today (runtime engine only)

The offline trace that produces the `.nbx` is a **separate system** from the
runtime engine and is *expected* to import vendor packages (that is how a component
gets into the `.nbx` graph in the first place). This audit covers the **runtime
engine** only.

| # | import | file:line | mode(s) | model(s) that trigger it | tier | remediation chantier |
|---|--------|-----------|---------|--------------------------|------|----------------------|
| 1 | `snac` | `core/module/audio/output_processor.py:116` | compiled **and** triton | **orpheus-3b** (audio token IDs → 24 kHz) | 1 — external neural codec | **P-ORPHEUS-SNAC-TRITON** (get SNAC into the orpheus `.nbx`, or `@triton.jit` RVQ+convT) |
| 2 | `kokoro` (`KPipeline`) | `core/flow/stages/kokoro.py:272` | compiled | **Kokoro-82M** g2p | 1 — vendor model/g2p | **P-G2P-EMBED** |
| 3 | `phonemizer` | `core/flow/stages/kokoro.py:280` (compiled) · `triton/audio_frontend.py:257` (triton) | both | **Kokoro-82M** g2p | 2 — text frontend | **P-G2P-EMBED** |
| 4 | `espeakng_loader` | `triton/audio_frontend.py:249` | triton | **Kokoro-82M** g2p | 2 — text frontend (ships espeak `.so`) | **P-G2P-EMBED** |
| 5 | `transformers` (tokenizer class) | `core/module/tokenizer/factory.py:257` | both | **all** models with an HF tokenizer (TinyLlama, deepseek-moe, Voxtral, canary, granite, orpheus, openaudio, chatterbox) | 2 — tokenizer runner | **P-TOKENIZER-INTERNALIZE** |
| 6 | `transformers` (`WhisperFeatureExtractor`) | `core/module/audio/input_processor.py:133` | compiled | **whisper-large**, **whisper-large-v3-turbo**, **Voxtral** (whisper mel) | 2/3 — mel frontend | **P-MEL-FRONTEND-COMPILED** |
| 7 | `sentencepiece` | `core/module/tokenizer/sp_tokenizer.py:53` · `core/flow/rnnt.py:502` · `triton/flow/rnnt.py:394` | both | **parakeet-tdt** + SP-tokenizer models | 2 — tokenizer runner | **P-TOKENIZER-INTERNALIZE** |
| 8 | `tiktoken` | `core/module/tokenizer/sp_tokenizer.py:1224` | both | tiktoken-vocab models | 2 — tokenizer runner | **P-TOKENIZER-INTERNALIZE** |
| 9 | `torchaudio` | `core/module/audio/input_processor.py:163,264,381` · `core/flow/rnnt.py:161` | compiled | **whisper×2, canary, parakeet, granite** (mel / resample) | 3 — DSP | **P-MEL-FRONTEND-COMPILED** |
| 10 | `librosa` | `core/module/audio/input_processor.py:171` (compiled) · `triton/audio_frontend.py` (triton, filterbank) | both | mel filterbank (whisper/voxtral/canary/granite/parakeet) | 3 — DSP math | **P-MEL-FRONTEND-INTERNALIZE** (replace `librosa.filters.mel` with a NeuroBrix mel-filterbank function) |
| — | `soundfile` | several (both modes) | both | all audio models | I/O boundary | **accepted** (file read/write, not compute) |
| — | `transformers.models.*` | `kernels/triton_kernels_ref/unsloth/...` | — | none | **DEAD** | `triton_kernels_ref/` is the OBSOLETE reference tree, not imported by the live engine — no action |

### Tier legend

- **Tier 1 — external neural model/codec** run as a vendor package (SNAC, kokoro
  model). Hard violation: this is model compute outside the `.nbx`/engine.
- **Tier 2 — text frontend / tokenizer / mel-extractor loader** imported at
  runtime. Should be embedded in the `.nbx` (the tokenizer *data* already is — the
  *runner* still imports the vendor lib) or computed by a NeuroBrix class.
- **Tier 3 — DSP math** (`torchaudio`, `librosa`): deterministic signal-processing
  helpers; re-implementable as NeuroBrix-internal functions (the triton numpy mel
  front-end already proves this for extraction; `librosa.filters.mel` is the last
  filterbank helper).

## What this chantier already fixed (triton side, 2026-06-08)

- Mel **extraction** in triton is now NeuroBrix-internal numpy (`triton/audio_frontend.py`)
  — `torchaudio`/`transformers` removed from the triton mel path (rows 6, 9 closed
  for **triton**; still open for **compiled**).
- Kokoro g2p in triton uses the torch-free phonemizer and **raises** rather than
  importing the kokoro/misaki model (row 2 closed for **triton**; rows 3/4 reduced
  to the espeak frontend, still an import → P-G2P-EMBED).
- `triton/` has **zero direct `torch` imports** (R33 grep clean).

## Remediation roadmap (named chantiers, not orphan TODOs)

1. **P-ORPHEUS-SNAC-TRITON** — get SNAC into the orpheus `.nbx` so the codec runs
   as engine ATen/triton kernels; or port SNAC (RVQ quantiser + transposed-conv
   decoder) as `@triton.jit`. Removes row 1. Highest priority (it is the only
   Tier-1 *neural codec* on a closed model's runtime path).
2. **P-G2P-EMBED** — embed phoneme conversion in the `.nbx` the way the tokenizer
   is embedded (g2p table / espeak rules baked at trace), with a NeuroBrix-internal
   g2p runner. Removes rows 2, 3, 4. The engine then needs no `kokoro`/`phonemizer`/
   `espeakng_loader` at runtime.
3. **P-TOKENIZER-INTERNALIZE** — make the tokenizer runner NeuroBrix-internal
   (`SPTokenizer` already exists) so loading the embedded tokenizer needs no
   `transformers`/`sentencepiece`/`tiktoken` at runtime. Removes rows 5, 7, 8.
4. **P-MEL-FRONTEND-COMPILED** — route the **compiled** mel/resample through a
   NeuroBrix mel class (reuse the audio_frontend math, torch tensors instead of
   numpy). Removes rows 6, 9 for compiled.
5. **P-MEL-FRONTEND-INTERNALIZE** — replace `librosa.filters.mel` with a NeuroBrix
   mel-filterbank function (both modes). Removes row 10.

After 1–5, the runtime engine imports only `torch` (compiled) and `triton`+`NBXTensor`
(triton) for compute — R34 fully satisfied.

## Detection grep

```bash
# Tier 1 — vendor neural models/codecs in the runtime
grep -rnE "import (snac|kokoro|misaki|diffusers|vocos|encodec)\b|from (snac|kokoro|misaki|diffusers|vocos|encodec) " \
  src/neurobrix/core src/neurobrix/triton src/neurobrix/kernels | grep -vE "test_|triton_kernels_ref"

# Tier 2/3 — vendor frontends / DSP in the runtime
grep -rnE "import (transformers|tiktoken|sentencepiece|mistral_common|phonemizer|espeakng_loader|torchaudio|librosa)\b" \
  src/neurobrix/core src/neurobrix/triton src/neurobrix/kernels | grep -vE "test_|triton_kernels_ref"
```

Any hit on a **live runtime** path is an R34 violation pending one of the chantiers above.
