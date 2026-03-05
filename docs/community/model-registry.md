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
| **Audio** | Speech-to-text, text-to-speech | Roadmap |
| **Video** | Video generation, video understanding | Roadmap |
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

- Submit a [Model Request](https://github.com/NeuroBrix/neurobrix/issues/new?template=model_request.yml) on GitHub
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

### Image Generation

| Model | Organization | Size |
|-------|-------------|------|
| Flex.1-alpha | Ostris | 24.5 GB |
| Sana 1600M (4K + Multilingual) | Sana | 12.1 GB |
| PixArt-Sigma-XL | PixArt | 20.3 GB |
| Janus-Pro-7B | DeepSeek AI | 13.8 GB |

### Large Language Models

| Model | Organization | Size |
|-------|-------------|------|
| DeepSeek-MoE-16B-Chat | DeepSeek AI | 30.6 GB |
| Qwen3-30B-A3B | Qwen | 57.1 GB |

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
GitHub: [github.com/NeuroBrix/neurobrix/issues](https://github.com/NeuroBrix/neurobrix/issues)

---

NeuroBrix is developed and maintained by **WizWorks OÜ**, a property of **Neural Networks Holding LTD**.
