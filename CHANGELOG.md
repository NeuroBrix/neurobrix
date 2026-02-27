# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Universal hardware auto-detection — `--hardware` flag is now optional
- OS-first detection architecture: Linux, macOS, and Windows support
- CPU detection in hardware profiles (model, cores, threads, RAM, instruction set features)
- CPU-only machine support for laptops, VMs, and edge devices without GPU
- GPU detection for 10 vendors: NVIDIA, AMD, Intel, Apple, Tenstorrent, Moore Threads, Biren, Iluvatar, Hygon DCU, Cambricon

### Changed
- Hardware profiles standardized to clean block-style YAML format

### Fixed
- CUDA fork error when running `neurobrix serve` without `--hardware`
- NVLink bandwidth calculation (300 GB/s correct, was 100 GB/s)

## [0.1.0-alpha.4] - 2026-02-26

### Added
- 12 new GPU hardware profiles (RTX 20/30/40 series, A10, A100, H100, L40S, T4)
- Pipeline parallel strategy (per-layer sequential fill)
- Full 11-strategy allocation system with industry-standard names
- MoE fusion for all execution modes (compiled + native)
- Enterprise-grade documentation system (MkDocs Material)
- Benchmarking infrastructure

### Changed
- Strategy names renamed: `component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding`
- Dtype maps consolidated to single source of truth

### Fixed
- V100 preferred dtype set to float16 — Sana 2.2x faster
- MoE dispatch: eliminated 6,144 GPU syncs per token
- FGP multi-device execution stability
- Serve mode KV cache overestimation for large-context models

### Security
- Enforced `weights_only=True` for torch.load (prevents pickle RCE)
- Zip-slip path traversal validation in registry import
- Safe arithmetic parser replacing `eval()` in shape resolver
- Path traversal check in NBX parallel extractor

## [0.1.0-alpha.3] - 2026-02-25

### Added
- Persistent model serving: `neurobrix serve`, `neurobrix chat`, `neurobrix stop`
- Multi-turn conversation with context management and automatic summarization
- `--max-tokens` CLI argument for chat
- Weight pre-warming at serve time
- Non-LLM output saving from warm serve path (images saved by daemon)

### Changed
- Autoregressive flow rewritten with Strategy Pattern (TextStrategy + ImageStrategy)
- Loading mode strategy-driven — FGP/PP strategies now correctly warm-serve

### Fixed
- Chat tokenization double-BOS (TinyLlama gibberish)
- Jinja2 chat template rendering matching HuggingFace standard
- RoPE table slice corruption during decode (TinyLlama)
- Absolute position IDs for pre-computed RoPE models
- Repetition penalty scope includes prompt context
- RoPE buffer loss on cold serving repeat requests
- Non-blocking daemon stop with escalation

### Removed
- All non-error print statements — zero terminal noise during inference

## [0.1.0-alpha.2] - 2026-02-24

### Added
- Universal model registry — single source of truth for all model configs
- Transformers vendor isolation: each model traces with its exact version

### Changed
- ZERO FALLBACK enforced: removed all silent config defaults

### Fixed
- Janus custom model loader (5 bugs)
- Tokenizer priority for chat templates
- Image-to-image input validation with clear error messages

## [0.1.0-alpha.1] - 2026-02-15

### Added
- Initial release of NeuroBrix universal deep learning inference engine
- NBX container format with TensorDAG, topology, manifest
- Prism hardware solver with multi-GPU allocation
- CompiledSequence zero-overhead execution engine
- DtypeEngine with automatic mixed precision
- 4 model families: image (diffusion + VQ), LLM, audio, video
- CLI: `run`, `hub`, `import`, `list`, `inspect`, `validate`
- MoE (Mixture of Experts) fused dispatch with NOP propagation
- KV cache with data-driven sizing
- Triton GPU kernel framework
- NeuroBrix registry at neurobrix.es
- Support for 9 models: Sana, PixArt-Alpha, PixArt-Sigma, FLUX.2-dev, Flex.1-alpha, Janus-Pro-7B, DeepSeek-MoE-16B, Qwen3-30B-A3B, TinyLlama-1.1B

[Unreleased]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a4...HEAD
[0.1.0-alpha.4]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a3...v0.1.0a4
[0.1.0-alpha.3]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a2...v0.1.0a3
[0.1.0-alpha.2]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a1...v0.1.0a2
[0.1.0-alpha.1]: https://github.com/Benkelaya/NeuroBrix/releases/tag/v0.1.0a1
