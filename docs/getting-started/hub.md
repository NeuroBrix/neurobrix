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
| PixArt-Sigma-XL-1024 | PixArt | OpenRAIL++ | Multi-subject generation |
| PixArt-XL-1024 | PixArt | OpenRAIL++ | XL generation |
| Sana-1600M-MultiLing | NVIDIA | NVIDIA Open Model License | 1024px, multilingual prompts |
| Sana-1600M-4Kpx-BF16 | NVIDIA | NVIDIA Open Model License | Up to 4K resolution, 1.6B params |
| Flex.1-alpha | Ostris | Apache 2.0 | Flexible image generation |

### Audio (Speech-to-Text, Speech Understanding, Text-to-Speech)

| Model | Author | License | Type |
|-------|--------|---------|------|
| Whisper-Large-V3 | OpenAI | MIT | STT |
| Whisper-V3-Turbo | OpenAI | MIT | STT |
| Parakeet-TDT-1.1B | NVIDIA | CC-BY-4.0 | STT |
| Canary-Qwen-2.5B | NVIDIA | CC-BY-4.0 | audio_llm |
| Voxtral-Mini-3B | Mistral AI | Apache 2.0 | audio_llm |
| Granite-Speech-3.3-8B | IBM | Apache 2.0 | audio_llm |
| Orpheus-3B | Canopy Labs | Apache 2.0 | TTS |
| Kokoro-82M | Hexgrad | Apache 2.0 | TTS |
| VibeVoice-1.5B | Microsoft | MIT | TTS |
| OpenAudio-S1-Mini | Fish Audio | CC-BY-NC-SA-4.0 | TTS |
| Chatterbox | Resemble AI | MIT | TTS |

> **Non-commercial:** OpenAudio S1 Mini uses CC-BY-NC-SA-4.0 — non-commercial use only.

### Image Upscalers (Super-Resolution)

| Model | Author | License | Scale |
|-------|--------|---------|-------|
| HAT-L-x4 | XPixelGroup | Apache 2.0 | 4x |
| Real-ESRGAN-x4 | Xintao Wang et al. | BSD-3-Clause | 4x |
| SwinIR-Classical-x4 | Jingyun Liang et al. | Apache 2.0 | 4x |
| Swin2SR-Classical-x4 | Marcos V. Conde et al. | Apache 2.0 | 4x |

### Large Language Models

| Model | Author | License | Description |
|-------|--------|---------|-------------|
| DeepSeek-MoE-16B-Chat | DeepSeek | DeepSeek License | 64-expert MoE |
| Qwen3-30B-A3B-Thinking | Alibaba / Qwen | Apache 2.0 | 30B/3B active, reasoning |
| TinyLlama-1.1B-Chat | TinyLlama | Apache 2.0 | Compact, fast |

### Video Generation

The complete video family — validated in all four execution modes, being published alongside the v0.3.0 release:

| Model | Author | License | Type |
|-------|--------|---------|------|
| Wan-AI/Wan2.1-T2V-1.3B | Alibaba | Apache 2.0 | text-to-video |
| Wan-AI/Wan2.1-VACE-1.3B | Alibaba | Apache 2.0 | video creation & editing |
| Wan-AI/Wan2.1-I2V-14B-480P | Alibaba | Apache 2.0 | image-to-video |
| Wan-AI/Wan2.2-I2V-A14B | Alibaba | Apache 2.0 | image-to-video, 28B dual-denoiser |
| THUDM/CogVideoX-2b | Zhipu AI | Apache 2.0 | text-to-video |
| THUDM/CogVideoX-5b-I2V | Zhipu AI | CogVideoX License | image-to-video |
| genmo/Mochi-1-preview | Genmo | Apache 2.0 | text-to-video |
| hpcai-tech/Open-Sora-v2 | HPC-AI Tech | Apache 2.0 | text-to-video |
| rhymes-ai/Allegro | Rhymes AI | Apache 2.0 | text-to-video, native 720×1280, 88 frames |
| rhymes-ai/Allegro-TI2V | Rhymes AI | Apache 2.0 | image-to-video |
| Efficient-Large-Model/SANA-Video-2B-720p | NVIDIA | NVIDIA Open Model License | text-to-video, 720p |

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
