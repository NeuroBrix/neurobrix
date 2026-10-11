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

The hub listed 45 public models on 2026-10-12. The hub is the live list: run
`neurobrix hub` or see [neurobrix.es/models](https://neurobrix.es/models). Being on the hub means a
container can be downloaded; each release's notes list the models and modes known not to work.

### Language models (`LLM`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `deepseek-ai/deepseek-moe-16b-chat` | deepseek | 32.8 GB |
| `ibm-granite/granite-3.1-1b-a400m-instruct` | apache-2.0 | 2.8 GB |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | apache-2.0 | 61.1 GB |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | apache-2.0 | 2.2 GB |

### Code (`CODE`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | deepseek | 32.1 GB |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | apache-2.0 | 61.1 GB |

### Vision-language (`VLM`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `zai-org/GLM-4.1V-9B-Thinking` | mit | 20.6 GB |

### Multimodal (`MULTIMODAL`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `deepseek-ai/Janus-Pro-7B` | mit | 14.8 GB |
| `inclusionAI/Ming-Lite-Omni-1.5` | mit | 54.9 GB |
| `openbmb/MiniCPM-o-4_5` | apache-2.0 | 20.7 GB |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | apache-2.0 | 70.6 GB |
| `Qwen/Qwen3-VL-30B-A3B-Thinking` | apache-2.0 | 62.1 GB |

### Speech-to-text (`STT`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `nvidia/parakeet-tdt-1.1b` | cc-by-4.0 | 4.3 GB |
| `openai/whisper-large` | mit | 6.2 GB |
| `openai/whisper-large-v3-turbo` | mit | 1.6 GB |

### Speech understanding (`AUDIO_LLM`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `ibm-granite/granite-speech-3.3-8b` | apache-2.0 | 17.3 GB |
| `mistralai/Voxtral-Mini-3B-2507` | apache-2.0 | 9.4 GB |
| `nvidia/canary-qwen-2.5b` | cc-by-4.0 | 5.1 GB |

### Text-to-speech (`TTS`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `canopylabs/orpheus-3b-0.1-ft` | apache-2.0 | 15.2 GB |
| `fishaudio/s1-mini` | cc-by-nc-sa-4.0 | 3.6 GB |
| `hexgrad/Kokoro-82M` | apache-2.0 | 370 MB |
| `microsoft/VibeVoice-1.5B` | mit | 5.4 GB |
| `ResembleAI/chatterbox` | mit | 2.2 GB |

### Image generation (`IMAGE`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `ostris/Flex.1-alpha` | apache-2.0 | 26.3 GB |
| `PixArt-alpha/PixArt-XL-2-1024-MS` | openrail++ | 21.8 GB |

### Video generation (`VIDEO`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | nvidia-open-model-license | 18.4 GB |
| `genmo/mochi-1-preview` | apache-2.0 | 41.0 GB |
| `hpcai-tech/Open-Sora-v2` | apache-2.0 | 45.6 GB |
| `rhymes-ai/Allegro` | apache-2.0 | 25.3 GB |
| `rhymes-ai/Allegro-TI2V` | apache-2.0 | 26.1 GB |
| `Wan-AI/Wan2.1-I2V-14B-480P` | apache-2.0 | 90.6 GB |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | apache-2.0 | 28.9 GB |
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | apache-2.0 | 19.5 GB |
| `Wan-AI/Wan2.2-I2V-A14B` | apache-2.0 | 126.7 GB |
| `zai-org/CogVideoX-2b` | apache-2.0 | 13.8 GB |
| `zai-org/CogVideoX-5b-I2V` | cogvideox-license | 22.6 GB |

### Upscaling (`UPSCALER`)

| `neurobrix import` | License (as listed on the hub) | Size |
|---|---|---:|
| `ai-forever/Real-ESRGAN-RealESRGAN_x2` | see the hub page | 67 MB |
| `ai-forever/Real-ESRGAN-RealESRGAN_x8` | see the hub page | 67 MB |
| `caidas/swin2SR-classical-sr-x2-64` | apache-2.0 | 49 MB |
| `caidas/swin2SR-classical-sr-x4-64` | apache-2.0 | 50 MB |
| `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | apache-2.0 | 49 MB |
| `jaideepsingh/upscale_models-HAT-L_SRx4_ImageNet-pretrain` | apache-2.0 | 166 MB |
| `JingyunLiang/SwinIR-001_classicalSR_DF2K_s64w8_SwinIR-M_x2` | apache-2.0 | 68 MB |
| `JingyunLiang/SwinIR-001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` | apache-2.0 | 60 MB |
| `xinntao/Real-ESRGAN-RealESRGAN_x4plus` | bsd-3-clause | 67 MB |

> **Non-commercial:** `fishaudio/s1-mini` is licensed CC-BY-NC-SA-4.0, non-commercial use only.

## Import a Model

```bash
neurobrix import <org>/<model>
```

Example:

```bash
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The model is downloaded and extracted to `~/.neurobrix/cache/<name>`; `--model` takes that `<name>`.

## Manage Models

```bash
# List installed models
neurobrix list

# Remove a model
neurobrix remove TinyLlama-1.1B-Chat-v1.0

# Clean all
neurobrix clean --all -y
```
