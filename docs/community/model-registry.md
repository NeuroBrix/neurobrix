# Model Registry Program

## Publish Your Model on NeuroBrix Hub

The [NeuroBrix Hub](https://neurobrix.org/models) is the official model registry for the NeuroBrix runtime. It hosts AI models packaged in the universal `.nbx` container format — ready to run on any supported hardware.

We invite AI model developers, research labs, and organizations to publish their models in the NeuroBrix Hub.

---

## Why Publish on NeuroBrix Hub

| Benefit | Description |
|---------|-------------|
| **Universal Execution** | Your model runs on any hardware supported by NeuroBrix — no porting needed |
| **Single Command Install** | Users download and run your model with `neurobrix import` + `neurobrix serve` |
| **Planned per machine** | Prism plans each run from the user's hardware profile |
| **No Framework Lock-in** | The `.nbx` format is framework-independent — your model is not tied to any specific runtime |
| **Enterprise Distribution** | Reach enterprise users who deploy AI at scale across heterogeneous hardware |
| **Hosted Distribution** | Your model is hosted and served from NeuroBrix infrastructure |

---

## Supported Model Families

| Family | Examples | Status |
|--------|----------|--------|
| **Image Generation** | Diffusion transformers, VQ-based generators, rectified flow models | Available |
| **Large Language Models** | Autoregressive transformers, Mixture-of-Experts | Available |
| **Audio** | Speech-to-text (Whisper, Parakeet), speech understanding (Canary-Qwen, Voxtral, Granite-Speech), text-to-speech (Orpheus, Kokoro, VibeVoice, OpenAudio, Chatterbox) | Available |
| **Image Upscalers** | Super-resolution (HAT, Real-ESRGAN, SwinIR, Swin2SR) | Available |
| **Video** | Video generation, text-to-video + image-to-video (Wan, CogVideoX, Mochi, Open-Sora, Allegro, SANA-Video) | Available |
| **Multimodal** | Vision-language (GLM-4.1V, Qwen3-VL), any-to-any (Qwen3-Omni, MiniCPM-o, Janus-Pro) | Available |

---

## How It Works

### The NBX Container

The NeuroBrix team builds your model's `.nbx` container. The container holds:

1. **The computation graph** of every component
2. **The weights**, in safetensors format
3. **The topology**: the execution flow (iterative, autoregressive, forward, ...)
4. **The configs**: everything the runtime needs, in one self-describing archive

The resulting `.nbx` file is self-contained. The NeuroBrix runtime reads it and executes the graph; no
model-specific code is needed.

### What We Need From You

To build your model's container and publish it, we need:

| Requirement | Description |
|-------------|-------------|
| **Model weights** | Hosted publicly (Hugging Face, your servers, etc.) or shared privately |
| **Inference code** | A working inference script or pipeline: the reference the container's outputs are checked against |
| **Example inputs** | Sample prompts, images, or other inputs for validation |
| **License confirmation** | Your model's license must permit redistribution in `.nbx` format |
| **Model card** | Description, capabilities, limitations, intended use |

---

## Publishing Tiers

### Community Submission

**For:** Open-source models with permissive licenses

- Submit a Model Request at [models@neurobrix.org](mailto:models@neurobrix.org) or via the issue tracker ([GitHub](https://github.com/NeuroBrix/neurobrix/issues) | [GitLab](https://gitlab.com/neurobrix/Neurobrix/-/issues))
- Our team evaluates the model and builds its container
- Published on the Hub with full attribution
- Community priority queue

### Partner Publication

**For:** Organizations that want priority publishing and co-branding

- Direct engineering collaboration for container building and optimization
- Priority queue — faster turnaround
- Co-branded listing on the Hub
- Joint announcements and case studies
- Ongoing model updates and maintenance

Contact [models@neurobrix.org](mailto:models@neurobrix.org) for partner publication.

### Enterprise Distribution

**For:** Proprietary models with controlled distribution

- Private Hub listings with access controls
- Custom licensing terms
- SLA-backed availability
- Dedicated support channel

Contact [enterprise@neurobrix.org](mailto:enterprise@neurobrix.org) for enterprise distribution.

---

## Current Hub Catalog

All models are the work of their respective authors and subject to their original licenses. Users must
review each model's license on its hub page before use. Some models carry a non-commercial license.

Examples by family:

| Family | Examples |
|--------|----------|
| Image generation | PixArt-Sigma-XL-2-1024-MS, PixArt-XL-2-1024-MS, Sana_1600M_1024px_MultiLing, Sana_1600M_4Kpx_BF16, Flex.1-alpha |
| Language models | TinyLlama-1.1B-Chat-v1.0, Qwen3-30B-A3B-Thinking-2507, Qwen3-Coder-30B-A3B-Instruct, DeepSeek-Coder-V2-Lite-Instruct, deepseek-moe-16b-chat, granite-3.1-1b-a400m-instruct |
| Vision-language and multimodal | GLM-4.1V-9B-Thinking, Qwen3-VL-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Instruct, MiniCPM-o-4_5, Janus-Pro-7B |
| Speech-to-text and speech understanding | whisper-large-v3-turbo, whisper-large, parakeet-tdt-1.1b, canary-qwen-2.5b, Voxtral-Mini-3B-2507, granite-speech-3.3-8b |
| Text-to-speech | Kokoro-82M, orpheus-3b-0.1-ft, VibeVoice-1.5B, chatterbox, s1-mini (OpenAudio) |
| Image upscalers | HAT, Real-ESRGAN (x2, x4plus, x8), SwinIR (x2, x4), Swin2SR (x2, x4, real-world x4) |
| Video | Wan2.1-T2V-1.3B, Wan2.1-I2V-14B-480P, Wan2.2-I2V-A14B, CogVideoX-2b, mochi-1-preview, Open-Sora-v2, SANA-Video_2B_720p |

The hub lists 49 models. Browse the full catalog, with each model's author and license: [neurobrix.org/models](https://neurobrix.org/models)

---

## Submission Process

1. **Contact us** at [models@neurobrix.org](mailto:models@neurobrix.org) with:
   - Model name and architecture
   - Link to weights and inference code
   - License type
   - Target family (image, LLM, audio, video)

2. **Technical review** — We assess whether the engine can run your model's architecture.

3. **Container build and validation** — We build the model's container and check its outputs against the vendor's reference.

4. **Hub publication** — Model is listed on [neurobrix.org/models](https://neurobrix.org/models) with documentation and quick-start instructions.

5. **Announcement** — Joint publication announcement (optional).

---

## Contact

**Model Registry Program**
Email: [models@neurobrix.org](mailto:models@neurobrix.org)
Web: [neurobrix.org/models](https://neurobrix.org/models)
Issues: [GitHub](https://github.com/NeuroBrix/neurobrix/issues) | [GitLab](https://gitlab.com/neurobrix/Neurobrix/-/issues)

---

Developed by Hocine Benkelaya · Supported and funded by [Neural Network Holding Ltd](https://neuralnetworkholding.com) · Commercial distribution by [WizWorks OÜ](https://wizworks.io).
