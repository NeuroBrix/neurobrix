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
| **Audio** | Speech-to-text (Whisper, Parakeet), speech understanding (Canary-Qwen, Voxtral, Granite-Speech), text-to-speech (Orpheus, Kokoro, VibeVoice, OpenAudio, Chatterbox) | Available |
| **Image Upscalers** | Super-resolution (HAT, Real-ESRGAN, SwinIR, Swin2SR) | Available |
| **Video** | Video generation, text-to-video + image-to-video (Wan, CogVideoX, Mochi, Open-Sora, Allegro, SANA-Video) | Available |
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

- Submit a Model Request at [models@neurobrix.es](mailto:models@neurobrix.es) or via the issue tracker ([GitHub](https://github.com/NeuroBrix/neurobrix/issues) | [GitLab](https://gitlab.com/neurobrix/neurobrix/-/issues))
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
| PixArt-Sigma-XL-1024 | PixArt | OpenRAIL++ |
| PixArt-XL-1024 | PixArt | OpenRAIL++ |
| Sana-1600M-MultiLing | NVIDIA | NVIDIA Open Model License |
| Sana-1600M-4Kpx-BF16 | NVIDIA | NVIDIA Open Model License |
| Flex.1-alpha | Ostris | Apache 2.0 |

### Audio (11 models)

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

### Image Upscalers

| Model | Author | License |
|-------|--------|---------|
| HAT-L-x4 | XPixelGroup | Apache 2.0 |
| Real-ESRGAN-x4 | Xintao Wang et al. | BSD-3-Clause |
| SwinIR-Classical-x4 | Jingyun Liang et al. | Apache 2.0 |
| Swin2SR-Classical-x4 | Marcos V. Conde et al. | Apache 2.0 |

### Large Language Models

| Model | Author | License |
|-------|--------|---------|
| DeepSeek-MoE-16B-Chat | DeepSeek | DeepSeek License |
| Qwen3-30B-A3B-Thinking | Alibaba / Qwen | Apache 2.0 |
| TinyLlama-1.1B-Chat | TinyLlama | Apache 2.0 |

> **Non-commercial:** OpenAudio S1 Mini uses CC-BY-NC-SA-4.0 — non-commercial use only.

### Video Generation

The complete video family (validated in all four execution modes, publishing alongside the v0.3.0 release):

| Model | Author | License |
|-------|--------|---------|
| Wan-AI/Wan2.1-T2V-1.3B | Alibaba | Apache 2.0 |
| Wan-AI/Wan2.1-VACE-1.3B | Alibaba | Apache 2.0 |
| Wan-AI/Wan2.1-I2V-14B-480P | Alibaba | Apache 2.0 |
| Wan-AI/Wan2.2-I2V-A14B | Alibaba | Apache 2.0 |
| THUDM/CogVideoX-2b | Zhipu AI | Apache 2.0 |
| THUDM/CogVideoX-5b-I2V | Zhipu AI | CogVideoX License |
| genmo/Mochi-1-preview | Genmo | Apache 2.0 |
| hpcai-tech/Open-Sora-v2 | HPC-AI Tech | Apache 2.0 |
| rhymes-ai/Allegro | Rhymes AI | Apache 2.0 |
| rhymes-ai/Allegro-TI2V | Rhymes AI | Apache 2.0 |
| Efficient-Large-Model/SANA-Video-2B-720p | NVIDIA | NVIDIA Open Model License |

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
Issues: [GitHub](https://github.com/NeuroBrix/neurobrix/issues) | [GitLab](https://gitlab.com/neurobrix/neurobrix/-/issues)

---

NeuroBrix is developed and maintained by **WizWorks OÜ**, a property of **Neural Networks Holding LTD**.
