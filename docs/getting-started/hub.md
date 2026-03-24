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
| [Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) | Ostris | [Apache 2.0](https://huggingface.co/ostris/Flex.1-alpha/blob/main/LICENSE) | Flexible image generation |
| [Sana 1600M 4K](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16) | NVIDIA / MIT | [Apache 2.0](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16/blob/main/LICENSE) | 4K resolution, 1.6B params |
| [PixArt-Sigma-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | PixArt | [OpenRAIL++](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/blob/main/LICENSE) | Multi-subject generation |
| [PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | PixArt | [OpenRAIL++](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/blob/main/LICENSE) | XL generation |
| [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | DeepSeek | [MIT](https://huggingface.co/deepseek-ai/Janus-Pro-7B/blob/main/LICENSE) | VQ autoregressive images |

### Video Generation

| Model | Author | License | Description |
|-------|--------|---------|-------------|
| [SANA-Video 2B 720p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p) | NVIDIA / MIT | [Apache 2.0](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p/blob/main/LICENSE) | 720p video, 81 frames |

### Audio (Speech-to-Text + Text-to-Speech)

| Model | Author | License | Type |
|-------|--------|---------|------|
| [Whisper Large](https://huggingface.co/openai/whisper-large) | OpenAI | [MIT](https://huggingface.co/openai/whisper-large/blob/main/LICENSE) | STT |
| [Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI | [MIT](https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/LICENSE) | STT |
| [Parakeet TDT 1.1B](https://huggingface.co/nvidia/parakeet-tdt-1.1b) | NVIDIA | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-1.1b) | STT |
| [Canary-Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) | NVIDIA | [CC-BY-4.0](https://huggingface.co/nvidia/canary-qwen-2.5b) | STT |
| [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Mistral AI | [Apache 2.0](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/blob/main/LICENSE) | STT |
| [Orpheus 3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | Canopy Labs | [Apache 2.0](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | TTS |
| [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) | Hexgrad | [Apache 2.0](https://huggingface.co/hexgrad/Kokoro-82M) | TTS |
| [VibeVoice 1.5B](https://huggingface.co/WillHeld/VibeVoice-1.5B) | Will Held | [Apache 2.0](https://huggingface.co/WillHeld/VibeVoice-1.5B) | TTS |
| [OpenAudio S1 Mini](https://huggingface.co/FishAudio/OpenAudio-S1-Mini) | Fish Audio | [CC-BY-NC-SA-4.0](https://huggingface.co/FishAudio/OpenAudio-S1-Mini) | TTS |
| [Chatterbox](https://huggingface.co/resemble-ai/chatterbox) | Resemble AI | [MIT](https://huggingface.co/resemble-ai/chatterbox) | TTS |

> **Non-commercial:** OpenAudio S1 Mini uses CC-BY-NC-SA-4.0 — non-commercial use only.

### Large Language Models

| Model | Author | License | Description |
|-------|--------|---------|-------------|
| [DeepSeek-MoE-16B](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) | DeepSeek | [MIT](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/blob/main/LICENSE) | 64-expert MoE |
| [Qwen3-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Alibaba / Qwen | [Apache 2.0](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/LICENSE) | 30B/3B active, reasoning |
| [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | TinyLlama | [Apache 2.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/LICENSE) | Compact, fast |

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
