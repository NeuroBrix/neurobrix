# Quick Start

## 1. Import a Model

Browse available models:

```bash
neurobrix hub
```

Import one:

```bash
neurobrix import sana/1600m-1024
```

This downloads the `.nbx` container from the NeuroBrix Hub and extracts it to the local cache.

## 2. Serve the Model (Recommended)

Serve mode loads weights into VRAM once and keeps them resident for fast repeated inference:

```bash
neurobrix serve --model 1600m-1024 --hardware v100-32g
```

Then generate:

```bash
neurobrix run --prompt "A cyberpunk cityscape at night"
```

Stop when done:

```bash
neurobrix stop
```

## 3. Single-Shot Mode

For one-off inference without a persistent daemon:

```bash
neurobrix run --model 1600m-1024 --hardware v100-32g --prompt "A mountain lake at sunset"
```

!!! note
    Single-shot mode loads and unloads weights each time. Use serve mode for repeated inference.

## 4. LLM Chat

For language models, use the interactive chat:

```bash
neurobrix serve --model deepseek-moe-16b-chat --hardware v100-32g
neurobrix chat
```

```
You: What is the capital of France?
DeepSeek: The capital of France is Paris...
You: /quit
```

## What Happened?

1. `neurobrix import` downloaded a `.nbx` container — a self-contained archive with the model's graph, weights, and topology
2. `neurobrix serve` loaded the model using the **Prism solver**, which automatically determined the best execution strategy for your hardware
3. `neurobrix run` executed the graph — the runtime has zero knowledge of what kind of model it is; it only sees tensors and operations
