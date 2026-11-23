# NeuroBrix CLI — Usage by Family

NeuroBrix runtime supports 9 model families with a uniform CLI. The
required and optional flags per family are declared in
`src/neurobrix/config/families/<family>.yml` and validated at runtime —
the CLI itself stays small.

This page lists canonical invocations per family with models from the
hub. Run `neurobrix run --help` for the full flag list, or read the
[CLI reference](reference/cli.md).

## Common flags

| Flag | Purpose |
|------|---------|
| `--model <name>` | The model's hub name, `<org>/<name>`, as `neurobrix import` recorded it (`neurobrix list` shows it). Can be omitted while a `serve` daemon is running |
| `--hardware <id>` | Hardware profile (auto-detect if omitted) |
| `--output <path>` | Save destination. Auto-extension if extension omitted; strict mismatch error otherwise |
| `--seed <int>` | Seeds the sampler and any random initialisation |
| `--compiled` / `--sequential` / `--triton` / `--triton-sequential` | Execution mode. No flag: `--compiled` where PyTorch is installed, `--triton` where it is not; the launch line says which |
| `--explain-plan` | Print the placement plan — strategy, why it won and what it beat, every component with its device and memory — and exit without loading anything |

## Execution modes — two branches, four modes

Every family runs through two independent compute branches, each in a
sequential (op-by-op) and a compiled (fused hot-loop) variant. With no
flag, a run takes `--compiled` where PyTorch is installed and `--triton`
where it is not.

| Flag | Branch | Substrate | Role |
|------|--------|-----------|------|
| `--compiled` *(default with PyTorch)* | PyTorch | `torch` + cuDNN/cuBLAS | production path |
| `--sequential` | PyTorch | `torch` ATen, op-by-op | op-by-op reference / debug path |
| `--triton` *(default without PyTorch)* | Triton | NeuroBrix Triton kernels + `NBXTensor` (nbxtensor), no `torch`/cuDNN | production path without PyTorch |
| `--triton-sequential` | Triton | NeuroBrix kernels, op-by-op | op-by-op debug path |

Use the default to just run a model, `--triton` for the path without
PyTorch, and the two `*-sequential` modes op-by-op to debug a numerical
discrepancy. Only one mode flag can be passed at a time. The flags are
also accepted by `neurobrix serve` and `neurobrix upscale`.

## llm — Text Generation

Examples on the hub: TinyLlama, DeepSeek-MoE, Granite (IBM), Qwen3 (Alibaba).

```bash
# Basic completion
neurobrix run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "What is 2+2?" --max-tokens 30

# Chat with system prompt
neurobrix run --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --system "You are a careful math tutor." \
    --prompt "Explain 7×8 to a child" \
    --max-tokens 200 --chat

# Several turns as one session (a JSON list of user messages)
neurobrix run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --turns turns.json --max-tokens 64
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` or `--turns` | one of them | Input text, or a JSON file of user messages played as one chat session |
| `--system` | no | System message (chat mode) |
| `--chat` / `--no-chat` | no | Force chat template on/off |
| `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--max-tokens` | no | Sampling params (order and defaults: [Sampling](#sampling)) |

Output: `.txt`.

## vlm — Vision-Language Models

Example on the hub: GLM-4.1V (Zhipu AI).

```bash
neurobrix run --model zai-org/GLM-4.1V-9B-Thinking \
    --input-image cat.jpg \
    --prompt "What animal is in this image?" \
    --max-tokens 100
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Question / instruction |
| `--input-image`, `--input-video`, `--audio` | no | Media for the model to read; with none, the language model answers the prompt alone |
| `--system`, `--chat` | no | Chat-style framing |

Output: `.txt`.

## multimodal — Unified Models (text, image or audio output)

Reference: DeepSeek Janus-Pro (`multimodal_understanding` ↔ `generate`).
**`--mode` is mandatory** because the .nbx is built for one head.

```bash
# Image generation (requires a Janus container built for image)
neurobrix run --model deepseek-ai/Janus-Pro-7B \
    --mode image --prompt "a red cat sitting on a couch"

# Text understanding (requires a Janus container built for text)
neurobrix run --model deepseek-ai/Janus-Pro-7B \
    --mode text --input-image cat.jpg --prompt "describe this image"

# Omni understanding builds: --mode text over image / audio / video inputs
neurobrix run --model openbmb/MiniCPM-o-4_5 \
    --mode text --input-image fruit.png --prompt "What fruit is this, and what colour is it?"
```

A `.nbx` built for one mode refuses the other by name:

```
this build serves only --mode image (its generation type is 'autoregressive_image');
a container built for --mode text serves that mode
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Always |
| `--mode {text,image,audio}` | yes | Multimodal-strict — no auto-deduction |
| `--input-image` | only `--mode text` | Image to describe |

Output: `.txt` (text mode), `.png` (image mode) or `.wav` (audio mode).

## tts — Text-to-Speech

References: Chatterbox, Orpheus, OpenAudio-S1, Kokoro, VibeVoice.

```bash
# Basic
neurobrix run --model ResembleAI/chatterbox --prompt "Hello world"

# With a speaker preset
neurobrix run --model hexgrad/Kokoro-82M \
    --prompt "Hello world" --speaker af_heart \
    --output greeting.wav
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Text to speak |
| `--speaker` | no | Speaker preset id (model-specific) |

`--reference-audio` (cloning a voice from a recording) is not wired in
this version: it is refused by name.

Output: `.wav` (24 kHz mono unless model overrides).

## stt — Speech-to-Text

References: Whisper (OpenAI), Parakeet TDT (NVIDIA).

```bash
neurobrix run --model openai/whisper-large-v3-turbo \
    --audio interview.wav \
    --output transcript.txt
```

| Flag | Required | Notes |
|------|----------|-------|
| `--audio` | yes | Input wav |
| `--prompt` | no | Optional context hint |

Output: `.txt`.

## audio_llm — Audio-Conditioned Language Models

References: Voxtral (Mistral), Canary-Qwen (NVIDIA), Granite Speech (IBM).

```bash
neurobrix run --model mistralai/Voxtral-Mini-3B-2507 \
    --audio meeting.wav \
    --prompt "summarize the discussion" \
    --max-tokens 200
```

| Flag | Required | Notes |
|------|----------|-------|
| `--audio` | yes | Input wav |
| `--prompt` | yes | Question / instruction |
| `--system`, `--chat`, `--max-tokens` | no | Same as llm |

Output: `.txt`.

## image — Image Generation

References: PixArt-XL, PixArt-Sigma, Sana (NVlabs), Flex.1.
The family declares `t2i` (default), `img2img` and `inpainting`, chosen
from the inputs.

```bash
# Text-to-image (default)
neurobrix run --model PixArt-alpha/PixArt-XL-2-1024-MS \
    --prompt "a red apple on a white plate" --steps 20
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Text prompt |
| `--input-image` | no | Triggers `img2img` mode |
| `--mask-image` | no | Requires `--input-image`, triggers `inpainting` |
| `--steps`, `--cfg`, `--height`, `--width` | no | Diffusion params |

`--reference-image` (a style reference) is not wired in this version:
it is refused by name.

Output: `.png`.

## upscaler — Image Super-Resolution

References: Swin2SR (classical = clean images, realworld = degraded
photos), Real-ESRGAN, SwinIR, HAT.

Dedicated subcommand:

```bash
neurobrix upscale \
    --model caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr \
    --input low_res.png \
    --output upscaled.png \
    --mode compiled
```

| Flag | Required | Notes |
|------|----------|-------|
| `--model` | yes | Upscaler model name |
| `--input` (alias `--input-image`) | yes | Source image (PNG/JPEG) |
| `--output` | yes | Destination image path (PNG) |
| `--mode` | no | `compiled` / `sequential` / `triton` / `triton-sequential`, the same choice as the engine flags (`--mode triton` = `--triton`). No mode: the same default as `run` |
| `--hardware` | no | Hardware profile id (auto-detected if omitted) |

Output: `.png` at `input × upscale_factor` (the factor is
intrinsic to each model, e.g. x2 / x4).

Model guide:

| Model | Scale | Best for |
|-------|-------|----------|
| swin2SR-classical-sr-x2-64 | 2× | clean source images |
| swin2SR-classical-sr-x4-64 | 4× | clean source images |
| swin2SR-realworld-sr-x4-64-bsrgan-psnr | 4× | degraded / real-world photos |

## video — Video Generation

References: Wan2.1/Wan2.2 (Alibaba), CogVideoX (Zhipu AI), Mochi (Genmo),
Open-Sora, Allegro (Rhymes AI), SANA-Video (NVIDIA).
The family declares `t2v` (default), `i2v` and `v2v`, chosen from the inputs.

```bash
# Text-to-video
neurobrix run --model Efficient-Large-Model/SANA-Video_2B_720p_diffusers \
    --prompt "ocean waves at sunset" --num-frames 24

# Image-to-video (auto: --input-image present)
neurobrix run --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --input-image first_frame.png \
    --prompt "camera pans left" --num-frames 32
```

| Flag | Required | Notes |
|------|----------|-------|
| `--prompt` | yes | Text prompt |
| `--input-image` | no | Triggers `i2v` mode |
| `--video` | no | Triggers `v2v` mode |
| `--num-frames`, `--fps` | no | Length / playback rate |
| `--steps`, `--cfg`, `--height`, `--width` | no | Diffusion params |

A flag left out takes the value the model's container declares: frame
count, step count, guidance scale and frame rate come from the
container, never from an engine default. On `main` (not in 0.6.16), when
neither the request nor the container gives one of them, the run is
refused by name.

Output: `.mp4` (H.264, yuv420p).

## Mode resolution rules

For each invocation:

1. If `--mode <X>` is passed: validate against `modes.supported`, use it.
2. Otherwise auto-deduce from optional flags' `triggers_mode` (e.g.
   `--input-image` triggers `img2img` for image family).
3. Otherwise fall back to `modes.default`.
4. If the family is `multimodal_strict` (currently only `multimodal`)
   and no mode is set, raise `--mode is required`.

## Output extension policy

- If `--output` is omitted: auto-name `output_<model>.<ext>` where
  `<ext>` comes from `output.default_extension_per_mode[mode]`.
- If `--output result.<ext>` is passed with a matching extension: use as-is.
- If the extension does not match the resolved mode: error clearly,
  never silently write wrong content.
- If `--output result` (no extension): append the auto-extension.

## Sampling

Every text, vlm and audio_llm decode samples the same way, on both engines (`--compiled` and `--triton`): one sampler
applies the repetition penalty, the temperature, top-k and top-p, then draws (or takes the most likely token when the
temperature is 0). Each of the four values is taken from the first source that sets it:

1. the request: the flag (`--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`), or the same key in a `serve` request (the warm path forwards all four and
   `--max-tokens`);
2. the model's container: its `runtime/defaults.json`, the vendor's generation config as the container holds it;
3. the family's documented defaults, in `config/families/<family>.yml` under `defaults:`. For `llm`, `vlm`,
   `audio_llm` and `multimodal`: temperature 0.7, top-k 50, top-p 0.9, repetition penalty 1.0.

A value the container leaves empty (`null`) lets the family's value through. A value no source sets stops the run with
an error naming it; the engine never substitutes one. The repetition penalty counts the prompt's tokens and the
generated ones for a text decode, and the generated ones for a decode fed with image or audio embeddings.

## What a command prints

stdout carries the command's result only: the generated text, the transcript, `SAVED: <path>` for a file, a table,
or — with `--json` — one JSON record on one line. stderr carries a few status lines (the engine in use, the run
time, a warning, an error with its remedy) and stays empty under `--json` unless something is wrong.

On the first launch on a machine that has no hardware profile, NeuroBrix creates that profile once; every later
launch reads it. A profile given with `--hardware` is read as it is.

| Option / variable | Effect |
|----------|--------|
| `-v`, `--verbose` (before or after the command) | Adds the engine's developer lines on stderr — the plan (strategy, devices, memory), the loaders, the replay, the kernel directory, the `[Timings]` / `[Peak]` lines |
| `-vv` | The developer lines and the full Python error detail of an error |
| `NBX_LOG=debug\|info\|warning\|error` | The level from the environment (`debug` = `-vv`): the knob for scripts and daemons that parse the developer lines. A serving daemon's log (`daemon.log` in the engine's home) always holds every line. Every `NBX_*` diagnostic variable below logs at the debug level |

## Storage layout

```
~/.neurobrix/
├── store/              Downloaded .nbx archives (transport format)
│   └── model-name.nbx
└── cache/              Extracted models (runtime reads from here)
    └── model-name/
        ├── manifest.json
        ├── topology.json
        ├── name.json          the hub name `neurobrix import` recorded
        ├── runtime/
        │   └── defaults.json
        └── ...
```

## Update check

Once a day, in a short separate process the command never waits for, the engine asks PyPI for the latest `neurobrix` and
`nbxtensor` and keeps the answer in `update_check.json` in the engine home. When the installed release is behind, the
next command prints one notice on stderr with the command that updates it for the way it was installed (pip, uv,
pipx, or a source checkout). The installed `nbxtensor` is also checked against the version this `neurobrix` requires,
locally, without the network. The check never refuses a command, says nothing when the network does not answer, and
prints nothing under `--json`. The update check is on `main`, not in 0.6.16.

| Variable | Effect |
|----------|--------|
| `NEUROBRIX_NO_UPDATE_CHECK=1` | No network call and no notice |
| `NEUROBRIX_PYPI_INDEX=<url>` | The package index asked (default `https://pypi.org/pypi`) |

## Debug environment variables

| Variable | Effect |
|----------|--------|
| `NBX_DEBUG=1` | Computes debug statistics, shown with `-v` / `NBX_LOG=debug`; slower, as it forces a GPU sync |
| `NBX_NAN_GUARD=1` | PyTorch compiled mode: find NaN/Inf after every op and replace it (expensive, debugging only) |

## Adding a new family

1. Create `src/neurobrix/config/families/<family>.yml` with the
   uniform schema (`output_processing`, `execution.has_kv_cache`,
   `inputs.required` / `inputs.optional`, `modes.supported` /
   `modes.default` / `modes.multimodal_strict`,
   `output.default_extension_per_mode`).
2. No CLI / runtime code changes required if the writer is one of
   `txt` / `wav` / `png` / `mp4`. For new formats, extend
   `core/runtime/output_dispatch.save_output()`.
