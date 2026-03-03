# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- DtypeEngine simplified to standard PyTorch AMP — removed custom overflow protection (add/sub clamping), _safe_downcast clamping, matmul output inf clamp, and bmm FP32 override
- bmm moved from AMP_FP32_OPS to AMP_FP16_OPS (matches PyTorch classification)
- amp_cast_result() is now a no-op (standard PyTorch AMP does no output clamping)

### Added
- Forge tracer: audio family support — 5 models at 100% trace (Whisper, Whisper V3 Turbo, Voxtral, Orpheus, Granite Speech)
- Forge tracer: universal .pth/native model loading — non-HuggingFace models (fish-speech, Kokoro) load via native constructors
- Forge tracer: Conformer/Linear encoder stimulus detection alongside Conv1d (Whisper)
- Forge tracer: multi-argument stimulus generation for models with complex forward() signatures
- Forge registry: audio model entries (dual_ar, kokoro, vibevoice, chatterbox) with loader configs

### Fixed
- Fix Forge VMM pool destruction before unified tracing — transformers models (TinyLlama, Qwen3) crashed with "CUDA error: invalid argument" because pool was destroyed after Phase A
- Fix Forge bf16 safety scan using wrong GPU — scan used `inventory.find_largest_gpu()` instead of VMM primary device, causing CUDA errors and false fp32 fallback for FLUX transformer
- Fix Forge bf16 scan CPU fallback — generic exceptions now fall back to CPU scan instead of immediately failing to fp32 (prevented FLUX transformer from loading as fp16)
- Fix Forge scipy vendoring crash — vendored scipy 1.12.0 (compiled against numpy 1.x) broke with numpy 2.x; now uses system scipy with dist-info version shim
- Retrace all 9 pre-VMM models with clean VMM graphs — zero meta tensors across PixArt-XL, PixArt-Sigma, TinyLlama, Sana 1024/4K, Janus-Pro-7B, Flex.1-alpha, FLUX.2-dev, Qwen3-30B-A3B
- Fix Prism double-capacity reduction in pipeline_parallel and block_scatter strategies — `fgp_target` was applied to already-reduced device capacities (0.95 × 0.92 = 0.874x), causing large models like Qwen3-30B to incorrectly fall back to zero3
- Fix SDPA multi-device crash in lazy_sequential — kwargs containing list/tuple tensors (attn_bias) were not moved to target device during cross-GPU transfers
- Eliminate redundant DtypeEngine clamp/copy operations — skip clamp when source dtype matches target, skip output clamp when no input downcast occurred, skip _to_copy when dtype already correct
- Eliminate weight transpose ops at compile time — pre-transpose weight tensors in bind_weights, removing ~5K aten::t ops per token from the hot loop

### Added
- Universal hardware auto-detection — `--hardware` flag is now optional
- OS-first detection architecture: Linux, macOS, and Windows support
- CPU detection in hardware profiles (model, cores, threads, RAM, instruction set features)
- CPU-only machine support for laptops, VMs, and edge devices without GPU
- GPU detection for 10 vendors: NVIDIA, AMD, Intel, Apple, Tenstorrent, Moore Threads, Biren, Iluvatar, Hygon DCU, Cambricon
- Data-driven CPU optimization: thread pinning, DNNL ISA selection, dtype validation from profile
- CPU RAM budget validation for zero3/lazy_sequential offload strategies
- Smart `pin_memory` decision based on available RAM vs weight size
- Proper `LazySequentialStrategy` with mixed per-component strategies (no longer a `SingleGPUStrategy` alias)
- `Zero3Strategy` now pins CPU weights and uses non-blocking DMA for per-op transfers
- Dedicated hardware profiles documentation with full YAML field reference

### Changed
- Hardware profiles standardized to clean block-style YAML format
- `PrismProfile.cpu` is now a `CPUConfig` dataclass (replaces `cpu_memory_gb: float`)
- `ExecutionPlan` carries `cpu_ram_mb` for runtime offload decisions
- CompiledSequence multi-device transfers use `non_blocking=True` for pinned CPU tensors
- All docs updated to reflect `--hardware` as optional

### Removed
- Dead `Zero3Executor` block-level code (incompatible with CompiledSequence pre-compiled ops)
- `_execute_zero3_subcomponent()` bypass in executor (replaced by strategy dispatch)

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
