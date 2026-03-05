# NeuroBrix Roadmap

**Last updated: February 2026**

NeuroBrix is a universal deep learning inference engine. One runtime, any model, any hardware. This roadmap outlines where we are today, where we are headed, and how you can help shape the future of open AI infrastructure.

---

## Vision

NeuroBrix aims to become the universal runtime for neural network inference — capable of running any modern AI model on any hardware, with zero proprietary dependencies, zero silent failures, and a clean, portable architecture.

Think of it as a universal operating system for neural networks.

---

## Why NeuroBrix?

The AI inference landscape is fragmented. Each tool solves a piece of the puzzle, but none solves the whole thing.

| Capability | Ollama | ComfyUI | vLLM | llama.cpp | **NeuroBrix** |
|---|---|---|---|---|---|
| LLM inference | Yes | No | Yes | Yes | **Yes** |
| Image generation | No | Yes | No | No | **Yes** |
| Audio models | No | No | No | Yes (whisper.cpp) | **Yes** |
| Video generation | No | Yes (plugins) | No | No | **Yes** |
| Multimodal (VQ) | No | No | No | No | **Yes** |
| Multi-GPU parallelism | No | Limited | Yes | Partial | **Yes** |
| Hardware-agnostic | No | NVIDIA only | NVIDIA focused | CPU + NVIDIA | **Any GPU** |
| Custom hardware profiles | No | No | No | No | **Yes** |
| Zero silent fallbacks | No | No | No | No | **Yes** |
| One unified CLI | Yes | No (GUI) | Yes | Yes | **Yes** |
| Model registry | Yes | No | No | No | **Yes** |
| pip installable | No | Yes | Yes | No | **Yes** |

NeuroBrix is the first runtime that treats all model families — diffusion, language, audio, video, multimodal — as first-class citizens under one unified architecture.

---

## Architecture Overview

NeuroBrix is designed around a simple principle: **the model container is the single source of truth**.

```
Install:   pip install neurobrix
Import:    neurobrix import sana/1600m-1024
Run:       neurobrix run --model 1600m-1024 --hardware v100-32g --prompt "A sunset over the ocean"
```

When you run a model, NeuroBrix:

1. **Reads the model container** — a self-describing `.nbx` package containing the computation graph, weights, metadata, and execution pipeline
2. **Selects an execution strategy** — based on your hardware profile (single GPU, multi-GPU with NVLink, CPU offload, etc.)
3. **Executes deterministically** — no guessing, no silent fallbacks, no hardcoded values. If something is wrong, you get a clear error, not a corrupted output

### Core Design Principles

- **Zero Hardcode** — All dimensions, configurations, and parameters are derived from the model container. Nothing is assumed.
- **Zero Fallback** — The system crashes explicitly if data is missing. No silent defaults that produce wrong results.
- **Zero Semantic** — The runtime has no concept of "image", "latent", or "text". It sees tensors and operations. This is what makes it truly universal.

### Hardware Strategy Engine (Prism)

NeuroBrix automatically determines the best execution strategy for your hardware:

- **Single GPU** — Standard execution with intelligent memory management
- **Pipeline Parallel** — Per-layer sequential fill across GPUs (like Accelerate `device_map="auto"`)
- **Component Placement** — Distributes whole components (text_encoder, transformer, vae) across GPUs
- **Block Scatter** — Block-level distribution across GPUs for very large components
- **Weight Sharding** — Weight-file round-robin distribution across GPUs
- **CPU Offload** — DeepSpeed-style offloading when GPU memory is insufficient

You describe your hardware once in a YAML profile, and NeuroBrix handles the rest.

---

## Supported Models

### Available Now

**Image Generation (Diffusion)**
- PixArt-Alpha (multiple resolutions)
- PixArt-Sigma XL 1024
- Sana 1024px and 4K

**Large Language Models**
- DeepSeek-MoE-16B-Chat
- Llama 3 family
- Mistral
- TinyLlama (lightweight)

**Multimodal**
- Janus-Pro-7B (vision + autoregressive image generation)

### Coming Soon

**Audio**
- Whisper (speech-to-text, all model sizes)

**Video**
- CogVideoX (text-to-video generation)

**Additional LLMs**
- DeepSeek-V2 and V3
- Llama 3.1 and 3.2
- Mixtral MoE
- Qwen 2.5

**Additional Image Models**
- Stable Diffusion XL
- Stable Diffusion 3

---

## Phased Roadmap

### Phase 1 — Foundation & Community (Q1-Q2 2026)

**Status: In Progress**

Phase 1 is about hardening what works and building the community foundation.

- **LoRA support** — Load and apply LoRA adapters at runtime. This is critical for the community and is a top priority for Phase 1.
- **Multi-hardware validation** — Testing on AMD ROCm (MI100/MI250), Apple Silicon (MPS), and Intel Arc GPUs.
- **Community hardware profiles program** — Submit your GPU configuration as a YAML profile and help NeuroBrix run on more hardware. See [Contributing Hardware Profiles](#contributing-hardware-profiles) below.
- **Audio models** — Complete Whisper support across all model sizes.
- **API documentation** — Full public API reference for developers building on NeuroBrix.
- **Profiler** — Built-in `--profile` flag to measure time and memory per operation.
- **Model registry expansion** — Grow the registry to 50+ models across all families.

### Phase 2 — Performance & Scale (Q3-Q4 2026)

Phase 2 is about making NeuroBrix fast and efficient.

- **Quantization** — INT4 and FP8 quantization for reduced memory usage and faster inference. Support for common community formats (AWQ, GPTQ).
- **Benchmarking system** — Reproducible benchmarks comparing NeuroBrix performance across hardware configurations and against other runtimes. Published results with methodology.
- **Fused kernels** — Custom Triton kernels for fused operations (LayerNorm+Linear, GELU+MatMul) to reduce memory bandwidth.
- **Flash Attention** — Native Flash Attention support for faster and more memory-efficient inference.
- **Video models** — Full CogVideoX support and other text-to-video architectures.
- **KV cache quantization** — INT8/FP8 KV cache for longer context windows on limited memory.
- **Graph visualizer** — Interactive web-based visualization of the execution graph (what is actually happening inside your model).

### Phase 3 — Universal Runtime (2027)

Phase 3 is the long-term vision: NeuroBrix as the standard runtime for AI inference.

- **6+ GPU architectures** — NVIDIA, AMD, Intel, Apple Silicon, ARM (Jetson, Snapdragon), and RISC-V.
- **95% Triton kernel coverage** — Custom kernels for the vast majority of operations, reducing PyTorch dependency.
- **Graph debugger** — Set breakpoints in the computation graph, inspect intermediate tensors, step through execution.
- **SDK for integrations** — Stable Python API for embedding NeuroBrix in other applications and services.
- **100+ models in registry** — Comprehensive coverage across diffusion, LLM, multimodal, audio, and video.

---

## Mobile Strategy

Let us be transparent about mobile and edge deployment.

NeuroBrix is built on Python and Triton. These are not technologies that run natively on phones or microcontrollers. We will not pretend otherwise, and we will not ship a half-working mobile solution just to check a box.

Our mobile strategy is honest:

1. **Server-side inference is the primary target.** Most production AI workloads run on servers, and that is where NeuroBrix delivers the most value today.

2. **Edge GPUs are a real target.** NVIDIA Jetson (Orin, AGX) runs full Python + Triton and can run NeuroBrix today. ARM server GPUs are on our roadmap.

3. **Apple Silicon Macs are a target.** MPS backend support is planned for Phase 1. Many developers use Macs locally, and we want to support that workflow.

4. **Phones and browsers are not a target.** We will not compile NeuroBrix to WASM or ship an iOS framework. If you need on-device inference for mobile apps, tools like Core ML, TensorFlow Lite, or ONNX Runtime Mobile are better choices. We would rather point you to the right tool than give you a bad experience.

5. **If the landscape changes, we adapt.** If Triton gains genuine mobile support, or if WebGPU matures for serious inference, we will revisit this position. But we will not chase hype.

This is a deliberate technical decision, not a limitation we are ignoring. We believe that doing fewer things exceptionally well is more valuable than doing everything poorly.

---

## Timeline

```
2026 Q1  [███████░░░░░]  Foundation: LoRA, Whisper, API docs, hardware validation
2026 Q2  [░░░░░░░░░░░░]  Community: Hardware profiles, model registry, profiler
2026 Q3  [░░░░░░░░░░░░]  Performance: INT4/FP8 quantization, fused kernels, benchmarks
2026 Q4  [░░░░░░░░░░░░]  Scale: Flash Attention, video models, KV cache quantization
2027 H1  [░░░░░░░░░░░░]  Universal: 6+ GPU architectures, graph debugger, SDK
2027 H2  [░░░░░░░░░░░░]  Maturity: 100+ models, 95% kernel coverage, stable API
```

---

## Success Metrics

We will measure progress against concrete, public metrics:

| Metric | Current (Feb 2026) | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|---|---|---|---|---|
| Supported model families | 3 (image, LLM, multimodal) | 4 (+audio) | 5 (+video) | 5+ |
| Models in registry | ~15 | 50+ | 100+ | 150+ |
| Tested GPU architectures | 1 (NVIDIA) | 3 (+AMD, +Apple) | 4 (+Intel) | 6+ |
| Triton kernel coverage | Experimental | Stable core ops | Fused ops | 95%+ |
| Community hardware profiles | 5 | 20+ | 50+ | 100+ |
| Quantization formats | None | None | INT4, FP8, AWQ, GPTQ | Full suite |
| LoRA support | No | Yes | Yes (multi-LoRA) | Yes |

---

## Get Involved

NeuroBrix is open source, and we are building it in public. There are many ways to contribute:

### Contributing Models

Every model added to the NeuroBrix registry makes the ecosystem more valuable. If you work with a model family we do not yet support, open an issue describing the architecture — we prioritize based on community demand.

### Contributing Hardware Profiles

This is one of the most impactful contributions you can make. NeuroBrix uses YAML hardware profiles to understand your GPU's capabilities (memory, compute, interconnect). If you have hardware we have not tested on:

1. Run NeuroBrix on your setup
2. Create a hardware profile YAML describing your configuration
3. Submit a pull request with your profile and test results

We especially need profiles for:
- AMD Instinct (MI100, MI250, MI300)
- AMD Radeon (RX 7900 XTX)
- Intel Arc (A770, A780)
- Apple Silicon (M1/M2/M3/M4 Pro, Max, Ultra)
- NVIDIA consumer GPUs (RTX 3090, 4090)
- Multi-GPU configurations (2x, 4x, 8x)

### Bug Reports & Testing

Run NeuroBrix on your models and hardware. When something fails, the error messages are designed to be useful — include them in your bug report. Every crash report helps us improve.

### Documentation

Help us improve guides, tutorials, and examples. Clear documentation lowers the barrier for everyone.

### Code Contributions

Check the [issues page](https://github.com/NeuroBrix/neurobrix/issues) for good first issues. The codebase follows strict principles (zero hardcode, zero fallback, zero dead code) — read the contributing guide before submitting a PR.

---

## Links

- **GitHub**: [github.com/NeuroBrix/neurobrix](https://github.com/NeuroBrix/neurobrix)
- **Model Registry**: [neurobrix.es](https://neurobrix.es)
- **Install**: `pip install neurobrix`

---

*This roadmap is a living document. It reflects our current plans and priorities, which may evolve based on community feedback, technical discoveries, and the rapidly changing AI landscape. We update it regularly and welcome input.*
