# Model Hub

The NeuroBrix Hub hosts pre-built `.nbx` containers ready for inference.

> **Important:** NeuroBrix is an inference engine — it does not create or own any model. All models are the work of their respective authors and subject to their original licenses. Users must review and accept each model's license before use.

## Browse Models

```bash
neurobrix hub
```

Filter by category:

```bash
neurobrix hub --category IMAGE
neurobrix hub --category LLM
neurobrix hub --category AUDIO
neurobrix hub --category VIDEO
```

Search:

```bash
neurobrix hub --search sana
```

## Available Models

### Image Generation

| Model | Author | License | Description |
|-------|--------|---------|-------------|
| Flex.1-alpha | Ostris | Apache 2.0 | Flexible image generation |
| Sana 1600M 4K | NVIDIA / MIT | Apache 2.0 | 4K resolution, 1.6B params |
| PixArt-Sigma-XL-2-1024-MS | PixArt | OpenRAIL++ | Multi-subject generation |
| PixArt-XL-2-1024-MS | PixArt | OpenRAIL++ | XL generation |
| Janus-Pro-7B | DeepSeek | MIT | VQ autoregressive images |

### Video Generation

| Model | Author | License | Description |
|-------|--------|---------|-------------|
| SANA-Video 2B 720p | NVIDIA / MIT | Apache 2.0 | 720p video, 81 frames |

### Audio (Speech-to-Text + Text-to-Speech)

| Model | Author | License | Type |
|-------|--------|---------|------|
| Whisper Large | OpenAI | MIT | STT |
| Whisper Large V3 Turbo | OpenAI | MIT | STT |
| Parakeet TDT 1.1B | NVIDIA | CC-BY-4.0 | STT |
| Canary-Qwen 2.5B | NVIDIA | CC-BY-4.0 | STT |
| Voxtral Mini 3B | Mistral AI | Apache 2.0 | STT |
| Orpheus 3B | Canopy Labs | Apache 2.0 | TTS |
| Kokoro 82M | Hexgrad | Apache 2.0 | TTS |
| VibeVoice 1.5B | Will Held | Apache 2.0 | TTS |
| OpenAudio S1 Mini | Fish Audio | CC-BY-NC-SA-4.0 | TTS |
| Chatterbox | Resemble AI | MIT | TTS |

> **Non-commercial:** OpenAudio S1 Mini uses CC-BY-NC-SA-4.0 — non-commercial use only.

### Large Language Models

| Model | Author | License | Description |
|-------|--------|---------|-------------|
| DeepSeek-MoE-16B | DeepSeek | MIT | 64-expert MoE |
| Qwen3-30B-A3B-Thinking | Alibaba / Qwen | Apache 2.0 | 30B/3B active, reasoning |
| TinyLlama 1.1B | TinyLlama | Apache 2.0 | Compact, fast |

## Import a Model

```bash
neurobrix import <org>/<model>
```

Example:

```bash
neurobrix import sana/1600m-1024
```

The model is downloaded and extracted to `~/.neurobrix/cache/`.

## Manage Models

```bash
# List installed models
neurobrix list

# Remove a model
neurobrix remove 1600m-1024

# Clean all
neurobrix clean --all -y
```
