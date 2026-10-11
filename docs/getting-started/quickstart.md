# Quick Start

## 1. Import a model

Browse available models:

```bash
neurobrix hub
neurobrix hub --category LLM
```

Import one by its hub identifier (`<org>/<name>`):

```bash
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

This downloads the `.nbx` container from the NeuroBrix Hub and extracts it to
`~/.neurobrix/cache/TinyLlama-1.1B-Chat-v1.0`. The `<name>` part is what `--model` takes.

## 2. Single-shot run

```bash
neurobrix run --model TinyLlama-1.1B-Chat-v1.0 --prompt "Explain recursion." --max-tokens 256
```

Hardware is detected automatically. To see where each component will run without loading
anything:

```bash
neurobrix run --model TinyLlama-1.1B-Chat-v1.0 --prompt "Explain recursion." --explain-plan
```

!!! note
    Single-shot mode loads and unloads the weights on every run. Use serve mode for repeated
    requests.

## 3. Serve mode

Serve mode loads the weights once and keeps them resident:

```bash
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0
neurobrix chat                          # interactive multi-turn chat (language models)
neurobrix run --prompt "Hello"          # --model can be omitted while a daemon runs
neurobrix stop                          # stop the daemon and free memory
```

!!! tip "Targeting specific hardware"
    By default NeuroBrix plans against the whole machine. `--hardware <profile-id>` targets a
    specific hardware profile instead (see [Hardware Profiles](../guide/hardware.md)).

## 4. Other families

```bash
# Speech-to-text
neurobrix import openai/whisper-large-v3-turbo
neurobrix run --model whisper-large-v3-turbo --audio recording.wav

# Video; --input-image switches an image-to-video model to its first frame
neurobrix import zai-org/CogVideoX-2b
neurobrix run --model CogVideoX-2b --prompt "a hot air balloon over mountains" \
    --num-frames 17 --seed 42 --output balloon.mp4

# Upscaling
neurobrix import xinntao/Real-ESRGAN-RealESRGAN_x4plus
neurobrix upscale --model Real-ESRGAN-RealESRGAN_x4plus --input photo.png --output photo_x4.png
```

## What happened?

1. `neurobrix import` downloaded a `.nbx` container: a self-contained archive with the model's
   graph, weights and topology.
2. `neurobrix run` or `neurobrix serve` detected your hardware and planned where each component
   runs.
3. The runtime executed the graph read from the container; the engine has no per-model pipeline.
