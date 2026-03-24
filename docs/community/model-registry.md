# Model Registry Program

## Publish Your Model on NeuroBrix Hub

The [NeuroBrix Hub](https://neurobrix.es/models) is the official model registry for the NeuroBrix runtime. It hosts AI models packaged in the universal `.nbx` container format — ready to run on any supported hardware with zero configuration.

We invite AI model developers, research labs, and organizations to publish their models in the NeuroBrix Hub.

---

## Why Publish on NeuroBrix Hub

| Benefit | Description |
|---------|-------------|
| **Universal Execution** | Your model runs on any hardware supported by NeuroBrix — no porting needed |
| **Single Command Install** | Users download and run your model with `neurobrix import` + `neurobrix serve` |
| **Hardware Optimization** | The Prism solver automatically finds the best execution strategy for each user's setup |
| **No Framework Lock-in** | The `.nbx` format is framework-independent — your model is not tied to any specific runtime |
| **Enterprise Distribution** | Reach enterprise users who deploy AI at scale across heterogeneous hardware |
| **Permanent Availability** | Your model is hosted and served from NeuroBrix infrastructure |

---

## Supported Model Families

| Family | Examples | Status |
|--------|----------|--------|
| **Image Generation** | Diffusion transformers, VQ-based generators, rectified flow models | Available |
| **Large Language Models** | Autoregressive transformers, Mixture-of-Experts | Available |
| **Audio** | Speech-to-text (Whisper, Parakeet, Canary, Voxtral), text-to-speech (Orpheus, Kokoro, VibeVoice, OpenAudio, Chatterbox) | Available |
| **Video** | Video generation (SANA-Video) | Available |
| **Multimodal** | Vision-language, any-to-any | Roadmap |

---

## How It Works

### The NBX Container

NeuroBrix uses a proprietary tracing technology to convert your model into a `.nbx` container. This process:

1. **Traces the computation graph** at a low level, capturing exact execution semantics
2. **Serializes weights** in safetensors format for fast loading
3. **Records topology** — execution flow (iterative, autoregressive, forward)
4. **Packages everything** into a single self-describing archive

The resulting `.nbx` file is completely self-contained. The NeuroBrix runtime reads it and executes the graph mechanically — no model-specific code required.

### What We Need From You

To trace and publish your model, we need:

| Requirement | Description |
|-------------|-------------|
| **Model weights** | Hosted publicly (Hugging Face, your servers, etc.) or shared privately |
| **Inference code** | A working inference script or pipeline we can trace |
| **Example inputs** | Sample prompts, images, or other inputs for validation |
| **License confirmation** | Your model's license must permit redistribution in `.nbx` format |
| **Model card** | Description, capabilities, limitations, intended use |

---

## Publishing Tiers

### Community Submission

**For:** Open-source models with permissive licenses

- Submit a Model Request at [models@neurobrix.es](mailto:models@neurobrix.es) or via [GitLab Issues](https://gitlab.com/neurobrix/neurobrix/-/issues)
- Our team evaluates and traces the model
- Published on the Hub with full attribution
- Community priority queue

### Partner Publication

**For:** Organizations that want priority publishing and co-branding

- Direct engineering collaboration for tracing and optimization
- Priority queue — faster turnaround
- Co-branded listing on the Hub
- Joint announcements and case studies
- Ongoing model updates and maintenance

Contact [models@neurobrix.es](mailto:models@neurobrix.es) for partner publication.

### Enterprise Distribution

**For:** Proprietary models with controlled distribution

- Private Hub listings with access controls
- Custom licensing terms
- SLA-backed availability
- Dedicated support channel

Contact [enterprise@neurobrix.es](mailto:enterprise@neurobrix.es) for enterprise distribution.

---

## Current Hub Catalog

All models are the work of their respective authors and subject to their original licenses. Users must review each model's license before use.

### Image Generation

| Model | Author | License |
|-------|--------|---------|
| [Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) | Ostris | [Apache 2.0](https://huggingface.co/ostris/Flex.1-alpha/blob/main/LICENSE) |
| [Sana 1600M](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16) | NVIDIA / MIT | [Apache 2.0](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16/blob/main/LICENSE) |
| [PixArt-Sigma-XL](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | PixArt | [OpenRAIL++](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/blob/main/LICENSE) |
| [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) | DeepSeek | [MIT](https://huggingface.co/deepseek-ai/Janus-Pro-7B/blob/main/LICENSE) |

### Video Generation

| Model | Author | License |
|-------|--------|---------|
| [SANA-Video 2B](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p) | NVIDIA / MIT | [Apache 2.0](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p/blob/main/LICENSE) |

### Audio (11 models)

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

### Large Language Models

| Model | Author | License |
|-------|--------|---------|
| [DeepSeek-MoE-16B](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) | DeepSeek | [MIT](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/blob/main/LICENSE) |
| [Qwen3-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Alibaba / Qwen | [Apache 2.0](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/LICENSE) |
| [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | TinyLlama | [Apache 2.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/LICENSE) |

> **Non-commercial:** OpenAudio S1 Mini uses CC-BY-NC-SA-4.0 — non-commercial use only.

Browse the full catalog: [neurobrix.es/models](https://neurobrix.es/models)

---

## Submission Process

1. **Contact us** at [models@neurobrix.es](mailto:models@neurobrix.es) with:
   - Model name and architecture
   - Link to weights and inference code
   - License type
   - Target family (image, LLM, audio, video)

2. **Technical review** — We assess compatibility with the NeuroBrix tracing pipeline.

3. **Tracing and validation** — We trace the model, validate outputs against reference, and optimize execution.

4. **Hub publication** — Model is listed on [neurobrix.es/models](https://neurobrix.es/models) with documentation and quick-start instructions.

5. **Announcement** — Joint publication announcement (optional).

---

## Contact

**Model Registry Program**
Email: [models@neurobrix.es](mailto:models@neurobrix.es)
Web: [neurobrix.es/models](https://neurobrix.es/models)
GitLab: [gitlab.com/neurobrix/neurobrix/-/issues](https://gitlab.com/neurobrix/neurobrix/-/issues)

---

NeuroBrix is developed and maintained by **WizWorks OÜ**, a property of **Neural Networks Holding LTD**.
