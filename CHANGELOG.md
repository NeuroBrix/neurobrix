# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apple Silicon (MPS) support — M1 through M5 Ultra, unified memory, auto-detection
- `DeviceBrand.APPLE` with `"mps"` device prefix in Prism hardware abstraction
- Apple Silicon chip database (20 variants: M1-M5 base/Pro/Max/Ultra with GPU cores, bandwidth, memory)
- `device_utils.py` — unified device abstraction (`device_sync`, `device_empty_cache`, `device_seed`, `device_memory_stats`, `device_multinomial`)
- No-silent-fallback guardrail hook (blocks `PYTORCH_ENABLE_MPS_FALLBACK` and try/except device swallowing)
- Single-GPU strategy shortcut in Prism solver (skips multi-GPU cascade for 1-device hardware)
- `neurobrix doctor` command with OS-specific PATH fix instructions
- GitLab CI/CD pipeline for PyPI publishing (OIDC trusted publisher + API token fallback)

### Changed
- All `torch.cuda.empty_cache()` calls replaced with device-agnostic `device_empty_cache()` (26 call sites across flow handlers, strategies, graph executor, serving engine)
- All `torch.cuda.synchronize()` for timing replaced with `device_sync()` (serving engine, strategy base)
- All `torch.cuda.manual_seed_all()` replaced with `device_seed()` (serving engine)
- VRAM reporting in serving engine uses `device_memory_stats()` (supports CUDA + MPS)
- `torch.multinomial` replaced with `device_multinomial()` — CPU round-trip on MPS (9 call sites)
- Removed hardcoded `"cuda:0"` defaults from loaders and strategies — crash explicitly if Prism provides no device
- All repository URLs migrated from GitHub to GitLab (`gitlab.com/neurobrix/Neurobrix`)
- Dependencies updated: added `pydantic`, `packaging`, `torchaudio`, `snac`, `phonemizer`, `imageio-ffmpeg`, `transformers`, `mistral-common`, `tiktoken` — all families work out of the box
- bf16 dtype support gated by Apple chip generation (M2+ with macOS 14+)

### Removed
- `licenses.py` — hardcoded license classifications deleted. Hub is the single source of truth.

### Fixed
- License gating desync between CLI and hub — CLI now reads `gated`/`licenseName`/`licenseUrl` from hub API
- Serving engine crash on `ExecutionPlan.allocations` — use `primary_device` property
- Prism profile loader mapped unknown brands to NVIDIA silently — Apple got `cuda:0` instead of `mps:0`. Now crashes on unknown brand.
- Weight loader only transferred weights to CUDA GPUs — MPS weights stayed on CPU, triggering multi-device path. Now transfers to any GPU device.
- macOS daemon used `os.fork()` + `os.setsid()` which breaks Metal GPU access (MTLCompilerService is per-session). Now uses `subprocess.Popen` like Windows.
- False `avx2` ISA warning on Apple Silicon — ARM chips use NEON, not x86 ISA. Skip check for arm64.
- Apple M2+ now prefers bf16 (not fp16) — bf16 has fp32 exponent range, prevents overflow in matmul/conv accumulation that caused blurry image output
- Disable AMP on MPS with bf16 — MPS crashes on mixed-dtype ops (`add(f16, f32)`). bf16 doesn't need AMP overflow protection anyway (fp32 exponent range).
- Add `force_uniform_dtype` to DtypeEngine — on MPS, `_to_copy` fp32 targets remapped to compute_dtype, all constants and graph inputs forced to match. Metal requires uniform dtype across all operands.
- SNAC audio decoder had silent `except ImportError` fallback returning zeros — now crashes explicitly
- `python -m neurobrix` shows PATH hint when CLI not on PATH

## [0.1.0] - 2026-03-26

First stable release of NeuroBrix — universal deep learning inference engine.

### Added
- NBX container format with TensorDAG, topology, manifest
- Prism hardware solver with multi-GPU allocation (11 strategies: single_gpu through zero3)
- CompiledSequence zero-overhead execution engine (eliminates all Python dict lookups)
- DtypeEngine with automatic mixed precision (standard PyTorch AMP rules)
- 4 model families: image (diffusion + VQ), LLM, audio, video
- CLI commands: `run`, `serve`, `chat`, `stop`, `hub`, `import`, `list`, `remove`, `clean`, `inspect`, `validate`, `info`, `doctor`
- MoE (Mixture of Experts) fused dispatch with NOP propagation
- KV cache with data-driven sizing and on-demand growth
- Triton GPU kernel framework
- NeuroBrix model registry at neurobrix.es
- Support for 34 models across 4 families (LLM, image, audio, video)
- Audio family: all 11 models working — Whisper, Whisper V3 Turbo, Parakeet, Orpheus, Canary-Qwen, Kokoro-82M, VibeVoice-1.5B, Voxtral, OpenAudio-S1, Granite Speech, Chatterbox
- Audio flow handlers: encoder_decoder, audio_llm, dual_ar, rnnt, tts_llm
- Universal AudioEngine with data-driven flow routing
- SANA-Video 720p support (video generation)
- Persistent model serving: `neurobrix serve`, `neurobrix chat`, `neurobrix stop`
- Multi-turn conversation with context management and automatic summarization
- Universal hardware auto-detection — `--hardware` flag is optional
- Cross-platform support: Windows, macOS, and Linux
- Platform-adaptive IPC: AF_UNIX on Unix/macOS, TCP localhost on Windows
- Universal TilingEngine — data-driven per-component tiling with accumulate-and-divide blending
- Symbolic spatial dims in compiled graphs — view/reshape ops use expression trees for multi-resolution
- ExprArg in CompiledSequence — runtime resolves symbolic expressions
- 12+ GPU hardware profiles (RTX 20/30/40 series, A10, A100, H100, L40S, T4, V100)
- Pipeline parallel, block scatter, weight sharding allocation strategies
- Prism hot/cold budget split for serve vs run mode
- Zero3 layer-wise pipelining with dual CUDA streams
- License system for model distribution with acceptance flow in `neurobrix import`
- Enterprise-grade documentation system (MkDocs Material)
- `neurobrix doctor` command for diagnosing PATH and installation issues

### Security
- Enforced `weights_only=True` for torch.load (prevents pickle RCE)
- Zip-slip path traversal validation in registry import
- Safe arithmetic parser replacing `eval()` in shape resolver

[Unreleased]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.0...main
[0.1.0]: https://gitlab.com/neurobrix/Neurobrix/-/releases/v0.1.0
