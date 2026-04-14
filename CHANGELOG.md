# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2026-04-14

### Changed
- DeepSeek benchmark script now requires `HF_TOKEN` to be provided via the shell environment or a gitignored `.env` file; no token is ever hardcoded. This replaces the previous version of the same file, which shipped with a hardcoded token — users who pulled `0.1.3` from the sdist should upgrade.
- GitLab CI `publish-pypi` stage switched to `when: manual`. New version tags no longer trigger an automatic PyPI upload; an operator now reviews the build artefacts on GitLab and clicks the job explicitly.

### Security
- Rotate the credential that was shipped in the `neurobrix-0.1.3.tar.gz` sdist on PyPI (hardcoded HuggingFace access token in `benchmarks/profile_hf_deepseek.py`). The sdist has been yanked; `pip install neurobrix` now resolves to `0.1.4` by default. Users who installed `0.1.3` from the sdist (not the wheel — the wheel does not include benchmarks) should upgrade to `0.1.4`.

## [0.1.3] - 2026-04-14

### Added
- `--triton` mode: DeepSeek-MoE-16B now supported end-to-end (greedy output `" Hello! How can I help you today?"` on `"Hello"`, matching the native path semantically). Joins TinyLlama-1.1B and Qwen3-30B-A3B as fully working LLMs in the Triton runtime.
- `--triton` decode speedups for LLMs across the board (V100 numbers, fp16):
  - TinyLlama-1.1B: full decode step 443 → 160 ms (2.8× faster).
  - Per-matmul on the decode hot path: 2.02 → 0.20 ms (10–18× depending on shape).
  - Per-SDPA on decode with GQA: 1.58 → 0.94 ms (1.7×).
- Decode-aware output precision for matrix multiplication: when running one token at a time, accumulation now lands in fp32, preventing silent overflow on very deep MoE stacks (Qwen3-30B observed crash → now stable). Prefill and image/video spatial matmuls keep their fp16 output — no memory regression on diffusion.
- Decode-aware attention: short-query attention now uses a compact block size (no more 99% wasted compute when generating one token at a time). GQA models compute in place — K/V are no longer expanded to the Q head count in front of every attention call.
- Triton profiling harness (opt-in, off by default):
  - `NBX_TRITON_PROF=1` — per-category ms/op breakdown (matmul / sdpa / elem / meta / embed / other) for every run.
  - `NBX_DUMP_TIDS=<path>` + `NBX_DUMP_TIDS_FILTER=<substrings>` — dump any op output as JSON for side-by-side native vs Triton numerical diff.
  - `NBX_MOE_DIAG=1` — dump MoE routing intermediates on the first forward pass.
- New benchmark: `benchmarks/profile_hf_deepseek.py` — reference timings against the HuggingFace + Accelerate device_map=auto baseline on the same hardware.

### Changed
- Triton dtype policy simplified: only `div` still forces an fp32 input upcast. Matmul ops (`mm`, `bmm`, `addmm`) now cooperate with the new decode-aware output precision instead of forcing every input to fp32 per call — removes a ~3.5 GB per-decode-step weight-copy cost that was silently capping throughput.
- Triton graph-load pipeline is more permissive about trace-shaped vs declared-shaped buffers: models that ship position-indexed lookup tables sized from the trace sample (DeepSeek's per-block rotary cache is the reference case) now load instead of crashing on a shape mismatch.
- Embedding wrapper accepts any scalar index dtype and casts internally. Fixes diffusion timestep → embedding paths (PixArt-Sigma and similar DiTs) that used to crash with a pointer/float type error.

### Fixed
- DeepSeek-MoE-16B `--triton`: previously produced gibberish at decode (`"роко"` / `"!!!!!"`). The three root causes were all addressed in this release:
  1. The model's MoE routing normalisation flag was silently ignored in Triton mode (defaulted to the Qwen3 convention), collapsing routed-expert magnitudes ~20×.
  2. Top-k selection over a softmax with non-power-of-two k returned a corrupted tail — the fix skips a redundant sort stage when the input fits in a single chunk. Side effect: one fewer kernel launch per MoE layer for every model whose expert count fits in that chunk.
  3. RoPE position indexing collapsed after the first decode step when cos/sin were recomputed per forward (DeepSeek's pattern). The runtime now pins the RoPE chain at its traced size so subsequent decode positions stay in-bounds.
- Qwen3-30B-A3B `--triton` now runs noticeably faster at decode from the new attention block-size heuristic and the GQA-in-place kernel path.
- TinyLlama-1.1B `--triton` decode is faster end-to-end (2.8× step time) and keeps the same output as before on greedy runs.
- PixArt-Sigma `--triton` no longer crashes on the embedding kernel (timestep dtype) or on the first `aten::add` after the timestep path (computable buffers now enter the Triton runtime in the expected tensor type). Further progress is blocked by an SDPA VRAM allocation failure partway through the transformer on 16 GB V100s — tracked as a separate issue; the native path has an unrelated config bug on the same setup.

### Removed
- Per-attention-call GQA materialization (`unsqueeze → expand → reshape → contiguous` of K and V). Replaced by kernel-native stride indirection, active only when the model has GQA; non-GQA models are bit-identical to before.

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
