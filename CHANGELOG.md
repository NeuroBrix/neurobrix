# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
