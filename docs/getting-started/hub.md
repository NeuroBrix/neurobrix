# Model Hub

The NeuroBrix Hub hosts pre-built `.nbx` containers ready for inference.

> **Important:** NeuroBrix is an inference engine — it does not create or own any model. All models are the work of their respective authors and subject to their original licenses. Users must review and accept each model's license before use.

## Browse Models

```bash
neurobrix hub
```

Filter by category:

```bash
neurobrix hub --category LLM
neurobrix hub --category VIDEO
neurobrix hub --category TTS
```

Search:

```bash
neurobrix hub --search whisper
```

## Available Models

The hub lists 49 models. The hub is the live list: run
`neurobrix hub` or see [neurobrix.org/models](https://neurobrix.org/models). Being on the hub means a
container can be downloaded. Some licenses (for example `cogvideox-license`, `deepseek`, `openrail++`,
`cc-by-nc-sa-4.0`) ask you to accept them on the model's hub page.

### Language models (`LLM`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `deepseek-ai/deepseek-moe-16b-chat` | deepseek | 30.5 GiB |
| `ibm-granite/granite-3.1-1b-a400m-instruct` | apache-2.0 | 2.6 GiB |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | apache-2.0 | 56.9 GiB |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | apache-2.0 | 2.1 GiB |

### Code (`CODE`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | deepseek | 29.9 GiB |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | apache-2.0 | 56.9 GiB |

### Vision-language (`VLM`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `zai-org/GLM-4.1V-9B-Thinking` | mit | 19.2 GiB |

### Multimodal (`MULTIMODAL`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `deepseek-ai/Janus-Pro-7B` | mit | 13.8 GiB |
| `inclusionAI/Ming-Lite-Omni-1.5` | mit | 51.1 GiB |
| `openbmb/MiniCPM-o-4_5` | apache-2.0 | 19.3 GiB |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | apache-2.0 | 65.8 GiB |
| `Qwen/Qwen3-VL-30B-A3B-Thinking` | apache-2.0 | 57.9 GiB |

### Speech-to-text (`STT`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `nvidia/parakeet-tdt-1.1b` | cc-by-4.0 | 4.0 GiB |
| `openai/whisper-large` | apache-2.0 | 5.8 GiB |
| `openai/whisper-large-v3-turbo` | mit | 1.5 GiB |

### Speech understanding (`AUDIO_LLM`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `ibm-granite/granite-speech-3.3-8b` | apache-2.0 | 16.1 GiB |
| `mistralai/Voxtral-Mini-3B-2507` | apache-2.0 | 8.7 GiB |
| `nvidia/canary-qwen-2.5b` | cc-by-4.0 | 4.7 GiB |

### Text-to-speech (`TTS`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `canopylabs/orpheus-3b-0.1-ft` | apache-2.0 | 14.1 GiB |
| `fishaudio/s1-mini` | cc-by-nc-sa-4.0 | 2.6 GiB |
| `hexgrad/Kokoro-82M` | apache-2.0 | 352.9 MiB |
| `microsoft/VibeVoice-1.5B` | mit | 5.0 GiB |
| `ResembleAI/chatterbox` | mit | 2.0 GiB |

### Image generation (`IMAGE`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `ostris/Flex.1-alpha` | apache-2.0 | 24.5 GiB |
| `PixArt-alpha/PixArt-XL-2-1024-MS` | openrail++ | 20.3 GiB |
| `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | openrail++ | 20.3 GiB |
| `Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers` | apache-2.0 | 9.0 GiB |
| `Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers` | apache-2.0 | 12.0 GiB |

### Video generation (`VIDEO`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | apache-2.0 | 17.1 GiB |
| `genmo/mochi-1-preview` | apache-2.0 | 38.2 GiB |
| `hpcai-tech/Open-Sora-v2` | apache-2.0 | 42.4 GiB |
| `rhymes-ai/Allegro` | apache-2.0 | 23.6 GiB |
| `rhymes-ai/Allegro-TI2V` | apache-2.0 | 24.3 GiB |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | apache-2.0 | 84.4 GiB |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | apache-2.0 | 26.9 GiB |
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | apache-2.0 | 18.2 GiB |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | apache-2.0 | 118.0 GiB |
| `zai-org/CogVideoX-2b` | apache-2.0 | 12.9 GiB |
| `zai-org/CogVideoX-5b-I2V` | cogvideox-license | 21.0 GiB |

### Upscaling (`UPSCALER`)

| `neurobrix import` | License (as listed on the hub) | Size (as listed on the hub) |
|---|---|---:|
| `ai-forever/Real-ESRGAN-RealESRGAN_x2` | bsd-3-clause | 63.9 MiB |
| `ai-forever/Real-ESRGAN-RealESRGAN_x8` | bsd-3-clause | 64.0 MiB |
| `caidas/swin2SR-classical-sr-x2-64` | apache-2.0 | 47.1 MiB |
| `caidas/swin2SR-classical-sr-x4-64` | apache-2.0 | 47.6 MiB |
| `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | apache-2.0 | 46.9 MiB |
| `XPixelGroup/HAT-HAT-L_SRx4_ImageNet-pretrain` | apache-2.0 | 145.8 MiB |
| `XPixelGroup/HAT-HAT-S_SRx4` | apache-2.0 | 34.8 MiB |
| `JingyunLiang/SwinIR-001_classicalSR_DF2K_s64w8_SwinIR-M_x2` | apache-2.0 | 42.7 MiB |
| `JingyunLiang/SwinIR-001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` | apache-2.0 | 43.2 MiB |
| `xinntao/Real-ESRGAN-RealESRGAN_x4plus` | bsd-3-clause | 63.9 MiB |

> **Non-commercial:** `fishaudio/s1-mini` is licensed CC-BY-NC-SA-4.0, non-commercial use only.

## Import a Model

```bash
neurobrix import <org>/<model>
```

Example:

```bash
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0 --accept-license
```

The model is downloaded and extracted under `~/.neurobrix/cache/`; `--model` takes the same `<org>/<model>` name.

## Manage Models

```bash
# List installed models
neurobrix list

# Remove a model
neurobrix remove TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Clean all
neurobrix clean --all -y
```
