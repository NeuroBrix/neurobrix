# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Triton compiled mode: `update_seq_dependent_constants()` — narrows RoPE cos/sin buffers to actual seq_len at runtime, fixing decode garbage
- Triton decode optimization: `skip_kills` parameter — skip kill_slots during decode (intermediates reuse same-size arena slots), eliminates cudaFree + sync overhead
- Deferred kill_slots with GPU sync at end of prefill — prevents use-after-free from async kernel reads
- `DeviceAllocator.sync_device()` — GPU synchronization via driver API (cudaDeviceSynchronize / hipDeviceSynchronize)
- New Triton modules: `moe.py` (fused MoE execution), `promotion.py` (symbolic promotion), `cfg/engine.py` (classifier-free guidance)
- New kernels: `dtype_convert.py` (bf16→fp16 GPU conversion), `floor.py`, `fused_moe.py`
- Weight loader bf16→fp16 GPU conversion via Triton kernel (zero CPU round-trip)

### Changed
- Refactor dtype to string everywhere above engine boundary — Prism, factory, and shared code use dtype strings ("float16", "bfloat16"), engines convert internally
- TritonDtypeEngine: remove mm/bmm/addmm from `_FP16_NEED_FP32` — Triton kernels accumulate in fp32 internally, upcast was causing fp16 overflow in KV cache
- Default `uses_absolute_position=True` when graph has position_ids input — fixes decode position tracking for all LLMs

### Fixed
- Triton compiled decode producing garbage — missing `update_seq_dependent_constants()` call in TritonExecutor.run() and _run_triton_compiled()
- Position IDs stuck at [[0]] during decode — absolute position default was False, should be True for any model with position_ids graph input

### Removed
- Eliminate `str(self.dtype).replace('torch.', '')` hacks from graph_executor and triton flow

## [0.1.2] - 2026-04-03

### Added
- `--triton` mode — compiled Triton kernel inference (136 kernel files, 128 dispatch entries)
- `--triton-sequential` mode — sequential Triton execution for kernel debugging
- `TritonSequentialDispatcher` — extends NativeATenDispatcher, routes compute ops to Triton kernels
- 136 pure `@triton.jit` kernel files extracted from FlagGems, attorch, Liger-Kernel, Flash-Attention (Dao-AILab)
- NBXTensor — lightweight tensor descriptor for zero-PyTorch metadata ops
- Universal launch layer: `_prepare_binary`, `_prepare_comparison` — broadcasting, scalar handling, device context for all ops
- `_cuda_guard` in dispatch — handles multi-GPU + Zero3 CPU offloading transparently
- Metal GPU detection: `--triton` on Apple Silicon shows "not compatible" message
- CPU Triton backend: auto-enables `TRITON_CPU_BACKEND=1` on CPU-only machines
- Symbolic shape patching for sequential mode: `_patch_seq_len_in_ops` resolves trace-time seq_len in creation ops
- Pure Triton inference mode for LLM autoregressive generation (`--triton` flag)
- Zero-torch flow handler: autoregressive.py, samplers.py, generator.py, session.py
- Triton sequential debug mode (`--triton-sequential` flag)
- KV cache with GQA support for Triton decode (O(1) per token)
- Strided scatter kernel for non-contiguous KV cache writes
- NBXTensor boundary functions: nbx_to_torch(), nbx_dtype_to_torch()

### Changed
- DtypeEngine: merged `FP16_PRECISION_OPS` into `_FP16_NEED_FP32` (subset of `AMP_FP16_OPS`), eliminated duplicate sets
- DtypeEngine: `amp_cast_inputs()` now handles `_FP16_NEED_FP32` (was only in `compile_op`)
- Conv2d kernel: replaced FlagGems with attorch (V100 `num_stages` compatibility)
- Conv1d: routes through conv2d via unsqueeze (V100 safe)
- `pow` kernel: uses `libdevice.pow` for negative base handling (was `exp(e*log(x))` → NaN)
- `compiled_ops.py` enforces: missing Triton kernel for compute op = crash with descriptive error
- Remove @triton.autotune from 100+ element-wise kernels (fixed BLOCK_SIZE=1024)
- Remove @triton.autotune from 36 compute-bound kernels (fixed conservative configs)
- Cold start reduced from 8+ minutes to ~5 seconds

### Fixed
- NBXTensor.cat() called is_contiguous as attribute instead of method — corrupted RoPE
- Symbolic promotion skipped when multiple symbols share trace_value (s1/s3 ambiguity)
- SDPA double-masking when graph passes explicit causal mask with is_causal=False
- NBXTensor.__setitem__ used flat copy_kernel on non-contiguous narrow view — corrupted KV cache
- NBXTensor.contiguous() used memcpy instead of strided copy for non-contiguous views

### Removed
- `kernels/adapter.py` (1181 lines) — replaced by `dispatch.py` + `wrappers.py`
- `kernels/mapping.py` (155 lines), `kernels/resolver.py` (316 lines), `kernels/registry.py` (68 lines), `kernels/exceptions.py` (15 lines)
- `kernels/ops_legacy/` directory, `kernels/arch/` directory, `kernels/spec.py`
- `_execute_triton_op`, `_precompile_dispatch_table`, `_exec_type_map` from graph_executor.py
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
- MPS dtype flow: AMP stays ON (same rules as CUDA). fp32 precision chain flows through single-input ops (pow, mean, rsqrt) safely. Multi-input ops (mm, addmm) cast inputs to compute_dtype via AMP FP16 wrappers. No mixed dtype at multi-input op boundaries.
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
