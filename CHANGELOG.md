# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.12] - 2026-03-20

### Fixed
- Windows: `os.kill(pid, 0)` crash — replaced with `ctypes.windll.kernel32.OpenProcess()` for daemon process check
- Windows: `signal.SIGTERM` handler registration crash — guarded with `hasattr(signal, 'SIGTERM')`
- Windows: ffmpeg video pipe deadlock — replaced manual `stdin.write()`/`close()`/`wait()` with `communicate()`

## [0.1.0-alpha.11] - 2026-03-19

### Fixed
- Windows path separator bug: weight loader and NBX container used `str(path)` which gives backslashes on Windows — replaced with `.as_posix()` for cross-platform path matching
- Runtime reaching outside NBX cache for tokenizer files — removed hardcoded dev paths, engine now reads exclusively from `~/.neurobrix/cache/`
- TTS output hardcoded to absolute Linux path — now writes to current working directory

## [0.1.0-alpha.10] - 2026-03-19

### Fixed
- Add all missing package dependencies: `sentencepiece`, `tokenizers`, `jinja2`, `soundfile`, `Pillow`
- Make `sentencepiece` import lazy (was top-level, crashed if not installed)
- Add optional dependency groups: `audio` (librosa), `mistral`, `tiktoken`, `full`

## [0.1.0-alpha.9] - 2026-03-19

### Added
- **Cross-platform support: Windows and macOS** — NeuroBrix now runs on Windows, macOS, and Linux
- Platform-adaptive IPC: AF_UNIX domain socket on Unix/macOS, TCP localhost:19384 on Windows
- Windows background daemon via `subprocess.Popen` with `CREATE_NEW_PROCESS_GROUP` (replaces Unix `os.fork()`)
- Windows force-kill via `taskkill /F` (replaces Unix `SIGKILL`)

### Changed
- Video output layout corrected to CTHW (diffusers standard: [B,C,T,H,W])
- Force-unload pre_loop components before VAE post_loop to free VRAM

### Fixed
- `os.fork()` crash on Windows (Unix-only syscall) — replaced with platform-adaptive daemonization
- `os.setsid()` crash on Windows — only called on Unix/macOS
- `socket.AF_UNIX` crash on Windows — replaced with platform-adaptive IPC transport
- `signal.SIGKILL` undefined on Windows — replaced with `taskkill /F` for force termination
- Stale socket cleanup crash on Windows (no Unix socket file to unlink)
- `import triton` crash on Windows — Triton imported lazily, only loaded for `--triton` mode (Linux-only)
- Add `torch` to package dependencies (was missing — users had to install manually)
- Rename optional dependency group `cuda` → `triton` for clarity

## [0.1.0-alpha.7] - 2026-03-19

### Added
- SANA-Video 720p support (1/10 video models): 1280×704, 81 frames, 16fps video generation
- Per-channel latent denormalization in VAE handler for models with `latents_mean`/`latents_std` buffers (LTX2Video)

### Fixed
- Video output green screen on QuickTime: replaced mp4v (MPEG-4 Part 2) codec with H.264 via ffmpeg/libx264
- SANA-Video VAE producing posterized output: missing pre-decode latent denormalization (`latents / std + mean`)
- `aten::copy` using non-in-place semantics in CompiledSequence, breaking view aliasing in RoPE patterns — caused all-NaN transformer output
- Ambiguous seq_len symbol promotion in CompiledSequence: skip promotion when multiple symbols share the same trace_value to prevent wrong symbol binding
- NBX build missing per-channel VAE normalization buffers for diffusers video models

## [0.1.0-alpha.6] - 2026-03-19

### Added
- **Audio family: all 11/11 models working** — Whisper, Whisper V3 Turbo, Parakeet, Orpheus, Canary-Qwen, Kokoro-82M, VibeVoice-1.5B, Voxtral, OpenAudio-S1, Granite Speech, Chatterbox
- `encoder_decoder` flow handler: Whisper-style encoder → cross-attention autoregressive decoder
- `audio_llm` flow handler: audio-conditioned LLM (Voxtral, Granite Speech, Canary-Qwen)
- `dual_ar` flow handler: DualAR semantic token generation (OpenAudio-S1)
- `rnnt` flow handler: frame-by-frame greedy TDT decode with NeMo-compatible mel preprocessing (Parakeet)
- `tts_llm` flow handler: text → LM → DDPM diffusion → acoustic decoder (VibeVoice)
- `audio_utils.py`: shared audio preprocessing/postprocessing utilities
- Universal AudioEngine — data-driven flow handler for all audio models (STT/TTS)
- AudioInputProcessor — routes audio preprocessing by topology (mel_spectrogram, raw_waveform, conformer)
- AudioOutputProcessor — token decode (STT) and waveform save (TTS)
- VibeVoice-1.5B TTS pipeline: LM forward + DDPM diffusion (20-step DDIM, cosine schedule, v_prediction) + native acoustic decoder (ConvNext1d)
- Kokoro-82M TTS pipeline: native text_encoder (BiLSTM), predictor (DurationEncoder + F0/N prosody), compiled decoder (iSTFTNet)
- DDIMSchedulerConfig: separate config validation for DDIM/DDPM schedulers
- NeMo .nemo archive support — auto-extraction and weight conversion
- NeMo mel spectrogram preprocessing (n_fft=512, per-feature normalize, dither, preemphasis)
- SNAC codec decode for Orpheus TTS audio token output (24kHz waveform)
- Delta feature extraction for Conformer models
- LLM-style text tokenization for TTS/audio models
- CLI `--audio` argument for speech-to-text input
- Serving daemon audio support
- lm_head component execution in audio flow for TTS models with separate lm_head (Orpheus)
- Universal hardware auto-detection — `--hardware` flag is now optional
- OS-first detection: Linux, macOS, Windows support
- CPU detection (model, cores, threads, RAM, ISA features) + CPU-only machine support
- GPU detection for 10 vendors: NVIDIA, AMD, Intel, Apple, Tenstorrent, Moore Threads, Biren, Iluvatar, Hygon DCU, Cambricon
- Universal TilingEngine — data-driven per-component tiling with accumulate-and-divide blending
- Symbolic spatial dims in compiled graphs — view/reshape ops use expression trees for multi-resolution
- ExprArg in CompiledSequence — runtime resolves symbolic expressions for view/reshape spatial dims
- `detect_vendor_dtype()` — discovers actual weight dtype from safetensors, .pth, .ckpt, or .nemo
- .pth, .nemo, .ckpt → safetensors weight conversion in NBX build
- DtypeEngine `amp_enabled` wired through full compilation chain
- Data-driven audio preprocessing from graph.json input shapes
- Proper `LazySequentialStrategy` with mixed per-component strategies
- `Zero3Strategy` with pinned CPU weights and non-blocking DMA for per-op transfers
- Symbolic shape injection for expand/repeat ops (algebraic propagation from shape tracker)

### Changed
- Audio flow decomposition: monolithic audio.py split into encoder_decoder, audio_llm, dual_ar, audio (pipeline) — each named after vendor execution pattern
- Orpheus uses autoregressive_generation flow (it IS a Llama-3 LLM, not a custom audio flow)
- Audio flow ZERO FALLBACK enforcement: direction, preprocessing, max_tokens, temperature, eos_token_id, decoder_start_token_id, latent_shape, decoder shapes, sample_rate all crash if missing
- Audio flow ZERO HARDCODE cleanup: replaced all hardcoded dtypes (float32/bfloat16) with data-driven compute dtype
- Audio flow ZERO SEMANTIC cleanup: component routing by topology native_subtype instead of component name matching
- DtypeEngine simplified to standard PyTorch AMP — removed custom overflow protection (add/sub clamping, _safe_downcast, matmul output inf clamp)
- CompiledSequence runtime promotion respects algebraic symbolization for view/reshape ops
- InstanceNorm affine default initialization in CompiledSequence for params not in checkpoint (weight=1, bias=0)
- Complex number literal parsing in CompiledSequence (e.g., `1j` for iSTFT ops)
- AdainResBlk1d support: learned skip connection (conv1x1), upsample (ConvTranspose1d + interpolate), rsqrt(2) scaling
- Hardware profiles standardized to clean block-style YAML format
- `PrismProfile.cpu` is now a `CPUConfig` dataclass (replaces `cpu_memory_gb: float`)
- `ExecutionPlan` carries `cpu_ram_mb` for runtime offload decisions
- CompiledSequence multi-device transfers use `non_blocking=True` for pinned CPU tensors

### Fixed
- OpenAudio-S1 codec decoder producing incorrect output due to missing snake activation compute
- OpenAudio-S1 end-to-end TTS pipeline now working (DualAR backbone → codec decoder → WAV)
- Chatterbox s3gen vocoder symbolic dimension recovery: item() breaks symbolic propagation, registry-driven post-trace recovery restores expressions
- Symbolic shape aliasing: value-based symbol lookup disabled for shape lists — symbols propagate through algebraic rules only, preventing head_dim/seq_len confusion
- Shape propagation _reduce (mean) rule handles dim as list, not just int
- SDPA: detect undersized causal mask from trace time and use is_causal=True
- LoRA weight matching: strip base_layer/base_model PEFT wrapper tokens before suffix matching
- Fix symbolic shape rebinding in AR loops — `bind_from_inputs()` clears `_runtime_values` before rebinding
- Fix view/reshape dimension collision when trace-time seq_len equals head_dim/2 — prevents batch dim corruption (Voxtral bmm crash)
- Fix input tensor detection in `_get_component_input_shape()` — match by `input_name` field and `input::` prefix
- Fix preprocessing auto-correction from graph input shape — detects mel_spectrogram vs conformer from layout
- Fix frame padding/truncation for encoders with non-symbolic position encoding
- Fix tokenizer encode compatibility for TTS models — fallback to manual tensor wrapping (orpheus-3b)
- AudioEngine AR loop: forced_decoder_ids applied during generation, fixing language detection for whisper-large
- AudioEngine AR loop: max_tokens treated as total sequence length limit (prevents position embedding overflow)
- Fix grid artifacts in diffusion VAE output (Sana 1K/4K, PixArt Sigma/Alpha) — clean model graphs + runtime TilingEngine
- Fix vae_scale_factor computation for AutoencoderDC — fallback to encoder/decoder_block_out_channels
- Fix Prism double-capacity reduction in pipeline_parallel and block_scatter strategies
- Fix SDPA multi-device crash in lazy_sequential — kwargs with list/tuple tensors not moved to target device
- Eliminate redundant DtypeEngine clamp/copy operations
- Eliminate weight transpose ops at compile time — pre-transpose in bind_weights
- DtypeEngine: added batch_norm, instance_norm to AMP FP32 ops — Conformer models work in fp16
- CUDA fork error when running `neurobrix serve` without `--hardware`
- NVLink bandwidth calculation (300 GB/s correct, was 100 GB/s)

### Removed
- VAETilingStrategy — replaced by universal TilingEngine
- EncoderDecoderAudioHandler — replaced by universal AudioEngine
- Whisper-specific AudioProcessor — replaced by AudioInputProcessor
- NBX builder no longer converts weight dtype — preserves vendor-original dtype faithfully
- Dead `Zero3Executor` block-level code
- `_execute_zero3_subcomponent()` bypass in executor

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
- Per-model transformers version pinning for reproducible compilation

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

[Unreleased]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a12...HEAD
[0.1.0-alpha.12]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a11...v0.1.0a12
[0.1.0-alpha.11]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a10...v0.1.0a11
[0.1.0-alpha.10]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a9...v0.1.0a10
[0.1.0-alpha.9]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a7...v0.1.0a9
[0.1.0-alpha.7]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a6...v0.1.0a7
[0.1.0-alpha.6]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a4...v0.1.0a6
[0.1.0-alpha.4]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a3...v0.1.0a4
[0.1.0-alpha.3]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a2...v0.1.0a3
[0.1.0-alpha.2]: https://github.com/Benkelaya/NeuroBrix/compare/v0.1.0a1...v0.1.0a2
[0.1.0-alpha.1]: https://github.com/Benkelaya/NeuroBrix/releases/tag/v0.1.0a1
