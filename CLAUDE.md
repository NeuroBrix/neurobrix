# NeuroBrix — Project Guidelines

## Core Philosophy
- **ZERO HARDCODE:** All values derived from the NBX container. Nothing hardcoded.
- **ZERO FALLBACK:** System crashes explicitly if data is missing. No silent defaults.
- **UNIVERSAL:** One engine handles any model on any hardware. The engine DETECTS capabilities, never KNOWS models.
- **DATA-DRIVEN FLOWS:** Flow handlers orchestrate WHEN to run components. The graph handles HOW. No `executor._weights` access in flow handlers.

## Data-Driven Flow Rule (MANDATORY)
- **Tracer traces the FULL forward pass** per component: input → embedding → backbone → output (e.g. logits). Symbolic shapes handle dynamic dimensions (seq_len=23 at trace, any value at runtime).
- **Flow handlers NEVER access `executor._weights`** — all inference happens inside the graph via `executor.run()`.
- **variable_resolver + topology.json** manage all data flow between components — same pattern for ALL families (image, LLM, audio, video).
- **The diffusion flow (iterative_process.py) is the reference** — it never touches weights, only orchestrates scheduler steps + graph execution.
- **WRONG**: `embed = executor._weights["token_embed.weight"]; logits = hidden @ embed.T` (shortcut)
- **RIGHT**: `outputs = executor.run(inputs); logits = outputs["logits"]` (graph does everything)
- **Exception**: Input preprocessing (mel spectrogram, tokenization) happens before the graph — that's OK.

## NeuroTax Rule — RUNTIME MUST USE STANDARD NAMES (MANDATORY)
- **All weight keys in NBX safetensors are NeuroTax-normalized** — vendor names are gone after forge build.
- **Runtime code MUST match on NeuroTax standard names**, never on vendor names.
- **Standard names**: `block`, `attn`, `ffn`, `gate`, `up`, `down`, `q`, `k`, `v`, `out`, `norm`, `embed`, `head`, `enc`, `pred`, `joint`

## NBX Container Rule (MANDATORY)
- **NBX is a faithful transport/storage format** — reproduces vendor weights identically.
- **NEVER convert dtype in the builder** — vendor ships fp32 → NBX stores fp32. bf16 → bf16.
- **Runtime adapts** — DtypeEngine + Prism handle dtype conversion at execution time based on hardware.

## Architecture

```
neurobrix/
  cli/           Command-line interface (run, serve, chat, hub, import, ...)
  core/          Runtime engine (executor, prism solver, dtype, flows) — NATIVE mode (PyTorch)
  kernels/       Triton GPU kernels + NBXTensor + dispatch table + wrappers
  triton/        TRITON mode — zero-torch inference engine (see Triton Mode section)
  nbx/           .nbx container format
  serving/       Persistent model serving daemon
  config/        Hardware profiles (YAML), family configs
```

## CLI Commands

```bash
neurobrix serve --model <n>                              # Persistent serving (hardware auto-detected)
neurobrix chat [--temperature T]                         # Interactive chat
neurobrix stop                                           # Stop daemon
neurobrix run --model <n> --prompt <text>                 # Single-shot (native compiled mode)
neurobrix run --model <n> --prompt <text> --triton        # Single-shot (triton compiled mode)
neurobrix run --model <n> --prompt <text> --triton-sequential  # Single-shot (triton debug mode)
neurobrix hub / import / list / remove / clean            # Model management
neurobrix info / inspect / validate                      # Inspection
```

## Families
- `image` — Diffusion and VQ image generation (Sana, PixArt, FLUX, Flex)
- `llm` — Autoregressive language models (TinyLlama, Qwen3, DeepSeek)
- `audio` — Speech-to-text, TTS (Whisper, Kokoro, Chatterbox, Parakeet, Voxtral)
- `video` — Video generation

## Code Hygiene
- Delete unused code immediately (functions, imports, branches, files)
- No commented-out code (use git history)
- No "TODO: Remove" comments — remove it now
- Fix problems when detected, don't defer
- **No stubs or identity functions** (`lambda x: x` for missing ops = FORBIDDEN)
- **No spaghetti**: triton code goes in `triton/`, native code goes in `core/`. Never mix.

## ═══════════════════════════════════════════════════════════
## Triton Mode — Zero PyTorch Inference (MANDATORY RULES)
## ═══════════════════════════════════════════════════════════

### Zero-Torch Contract (STRICT)
The `--triton` and `--triton-sequential` modes execute with ZERO PyTorch dependency in the hot path.

**ALLOWED dependencies**: triton, ctypes (stdlib), numpy (weight loading boundary ONLY), Python stdlib.

**FORBIDDEN in any file under `src/neurobrix/triton/`**:
- `import torch` (except documented boundaries below)
- `torch.cuda.*`, `torch.Tensor`, `F.scaled_dot_product_attention`
- `nbx_to_torch()` in the hot path
- Any call that creates or manipulates a `torch.Tensor`

**Documented boundaries (the ONLY places torch may appear)**:
- `flow/audio.py`, `flow/rnnt.py` — audio preprocessing (torchaudio mel spectrogram). This is INPUT preprocessing, not the inference hot path.
- All other torch usage MUST be eliminated.

### Hardware Abstraction (MANDATORY)
NeuroBrix is UNIVERSAL. The DeviceAllocator MUST NOT hardcode CUDA.

**The `_RUNTIME_BACKENDS` mapping in `nbx_tensor.py`** dispatches to the correct GPU runtime:
- `nvidia` → `libcudart.so` (cudaMalloc, cudaMemcpy, cudaFree, cudaSetDevice)
- `amd` → `libamdhip64.so` (hipMalloc, hipMemcpy, hipFree, hipSetDevice)
- Backend auto-detected from hardware profile (`devices[0].brand`) or Triton (`triton.runtime.driver.active.get_current_target().backend`)

**FORBIDDEN**: hardcoded strings `"cuda"`, `"cuda:0"`, `libcudart.so` anywhere in `triton/`.
**CORRECT**: `f"{device_backend}:{device_idx}"` where `device_backend` comes from the hardware profile.

Triton kernels themselves need NO changes — Triton compiles for the correct backend automatically (PTX for NVIDIA, GCN for AMD). Only the memory allocator and device management are affected.

### Architectural Separation (MANDATORY)
ALL triton code lives in `src/neurobrix/triton/`. NOWHERE ELSE.

**FORBIDDEN**:
- Adding functions named `_*_triton_*()` in `graph_executor.py` or `core/`
- Putting triton logic (NBXTensor ops, kernel wrappers) in native flow handlers
- Creating "bridge" functions that duplicate code between native and triton paths

**What stays in `graph_executor.py`** (entry points only, < 10 lines each):
- `_run_triton()` — calls `TritonSequence.run()`
- `_ensure_triton_compiled()` — init
- `register_triton_interceptors()` — KV cache bridge

**Everything else goes in `triton/`**:
- MoE execution → `triton/moe.py`
- Weight conversion → `triton/weight_loader.py`
- Sequential dispatch → `triton/sequential.py`

### Device Management (MANDATORY)
The GPU device context is set ONCE at the execution entry point. Internal functions NEVER touch it.

**Entry points (the ONLY places that set device)**:
- `triton/sequence.py` `run()` — compiled mode
- `graph_executor.py` `_run_triton()` — dispatch
- `graph_executor.py` `_run_triton_sequential()` — dispatch

**FORBIDDEN to call `ensure_triton_device()` in**:
- `_strided_copy()`, `_strided_scatter()` — internal memory ops
- `NBXTensor.to()` — dtype casting
- Any wrapper in `wrappers.py` — use `_set_device(tensor)` which reads `tensor._device_idx`

**Device setter MUST use GPU driver API, NOT torch**:
- NVIDIA: `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` via `libcuda.so`
- AMD: `hipSetDevice` via `libamdhip64.so`
- OR: `triton.runtime.driver.active` if it exposes a device setter

### Contrat Tensoriel Universel (MANDATORY)
Guards (dtype, device, contiguous, broadcast) are in the BASE functions, never in individual wrappers.

**`_prepare_binary(a, b)`** in `wrappers.py`:
- Scalar detection → tensor ALWAYS in position `a`, scalar ALWAYS in `b` when `is_scalar=True`
- Device alignment → move `b` to `a`'s device if different
- Dtype alignment → cast to wider dtype
- Broadcasting → `expand()` + `contiguous()`
- All callers (mul, add, div, sub, comparisons) use `a` as tensor, `b` as scalar. No `hasattr(a, 'data_ptr')` checks.

**`_prepare_comparison(a, b)`**: Same contract, output is bool.

**`NBXTensor.cat(tensors, dim)`**: Dtype alignment + device alignment BEFORE kernel launch.

**`NBXTensor.__setitem__(key, value)`**: Dtype cast + device transfer + strided scatter.

### Triton Package Structure

```
src/neurobrix/triton/          # 24 files, zero torch in hot path
├── __init__.py
├── arena.py                   # O(1) slot storage for compiled execution
├── constants.py               # Base64 graph constants loader
├── dtype.py                   # TritonDtypeEngine (AMP rules, NBXDtype)
├── executor.py                # Full pipeline: load → compile → bind → run
├── generator.py               # Token generation loop
├── kv_cache.py                # KV cache + SDPA interceptor (GQA support)
├── moe.py                     # MoE fused execution (zero torch)
├── promotion.py               # Shared symbolic promotion (compiled + sequential)
├── samplers.py                # Sampling strategies (argmax, top-k, nucleus)
├── sequence.py                # Compiled hot loop (arena + closures)
├── sequential.py              # Op-by-op debug dispatcher
├── session.py                 # LM session (prefill/decode lifecycle)
├── symbols.py                 # Symbolic shape resolution
├── weight_loader.py           # safetensors → numpy → cudaMemcpy → NBXTensor
├── cfg/
│   └── engine.py              # CFG (classifier-free guidance) for diffusion
└── flow/                      # Flow handlers (zero torch)
    ├── autoregressive.py      # LLM text generation
    ├── iterative_process.py   # Diffusion (Sana, FLUX, PixArt)
    ├── audio.py               # Whisper STT
    ├── encoder_decoder.py     # Generic encoder-decoder
    ├── rnnt.py                # RNN-Transducer (Parakeet)
    ├── tts_llm.py             # TTS via LLM (Orpheus, Chatterbox)
    ├── dual_ar.py             # DualAR (OpenAudio)
    ├── forward_pass.py        # Sequential component execution
    └── static_graph.py        # Single forward pass
```

### Flow Support Status

| Flow | Family | Handler | Status |
|------|--------|---------|--------|
| autoregressive_generation | LLM | `flow/autoregressive.py` | ✅ Validated (TinyLlama) |
| iterative_process | Image | `flow/iterative_process.py` | Pipeline runs, image quality TBD |
| audio | Audio STT | `flow/audio.py` | Ported, not validated |
| encoder_decoder | Audio | `flow/encoder_decoder.py` | Ported, not validated |
| rnnt | Audio ASR | `flow/rnnt.py` | Ported, not validated |
| tts_llm | TTS | `flow/tts_llm.py` | Ported, not validated |
| dual_ar | Audio Gen | `flow/dual_ar.py` | Ported, not validated |
| forward_pass | Various | `flow/forward_pass.py` | Ported, no test model |
| static_graph | Various | `flow/static_graph.py` | Ported, no test model |

### Key Bugs Fixed (Reference for Future Debugging)
1. **`cat()` + `is_contiguous()`**: Must call `is_contiguous()` with parentheses — it's a method, not a property. `if tensor.is_contiguous` is always True (method object is truthy).
2. **Symbolic promotion ambiguity**: `_promote_seq_len_scalars` must handle s1/s3 collision when two symbols bind to same dimension.
3. **SDPA causal mask**: The graph stores `is_causal=False` with an explicit tril mask. The native mode overrides to `is_causal=True` + drops the mask. Triton mode MUST do the same — detect 2D causal mask → set `IS_CAUSAL=True` + drop mask. Double-masking produces numerical divergence.
4. **`__setitem__` strided scatter**: KV cache write via narrow view is non-contiguous. Flat `copy_kernel` writes to wrong positions. Must use `strided_scatter_kernel`.
5. **Device context corruption**: `ensure_triton_device` called in internal functions changes global device context. Must only be called at entry points.

### Anti-Patterns (NEVER DO THESE)

| ❌ WRONG | ✅ CORRECT |
|----------|-----------|
| Scan all flows, make a table of ❌, fix nothing | One flow at a time, depth-first, fixed to 100% |
| `"floor": lambda x: x` (stub identity) | Write the real kernel (10 lines) or crash explicitly |
| Add triton code in `graph_executor.py` | All triton code in `src/neurobrix/triton/` |
| `torch.cuda.set_device()` for device management | GPU driver API (`cuCtxSetCurrent` / `hipSetDevice`) |
| Call `ensure_triton_device` in internal functions | Device set ONCE at entry point |
| Patch dtype/device/contiguous in individual wrappers | Guards in `_prepare_binary`, `cat`, `__setitem__` |
| Test ops with random inputs, claim "cosine 1.0" | Test with REAL pipeline inputs |
| Say "validated" after testing only TinyLlama | Minimum: 2 LLMs + 1 diffusion model |
| `nbx_to_torch` in the hot path | Rewrite the module (scheduler etc.) in pure NBXTensor |
| Hardcode `"cuda"`, `"cuda:0"`, `libcudart.so` | Use hardware profile backend variable |

## Git Workflow (MANDATORY)

### Branch Strategy
- **main** = public branch (GitLab, PyPI). Clean, squashed commits only.
- Work directly on `main` locally. Commit frequently with clear messages.

### Commit Rules
1. **Group related changes** into a single commit
2. **Commit message format**: `type: description` (fix, feat, security, docs, refactor, perf, build)
3. **Push to origin/main** after each logical group of commits
4. Never push broken code. Run LLM regression test before pushing.

### Release Workflow
1. Accumulate changes under `## [Unreleased]` in CHANGELOG.md
2. When ready: bump version via `.claude/hooks/version-bump.sh [patch|minor|major]`
3. Stamp CHANGELOG, commit, tag, push.

### Forge Git (SEPARATE REPO — MANDATORY)
- `forge/` has its OWN `.git` — completely separate from NeuroBrix
- Remote: `https://gitlab.com/neurobrix/NeuroBrix_forge` (private)
- Auto-commit + auto-push hook on forge file edits

## Output Convention
Always generate test outputs in project root (`/home/mlops/NeuroBrix_System/`), not `/tmp/`.

## Modification Tracking
All code modifications MUST be logged in `MODIFICATIONS.md` at the project root.

## CHANGELOG Maintenance (MANDATORY)
Every commit that modifies source code MUST include a CHANGELOG.md update under `## [Unreleased]`.
Categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.
Style: Imperative mood, start with a verb.

## Regression Testing (MANDATORY)
After ANY modification to shared code (`kernels/`, `triton/sequence.py`, `triton/symbols.py`, `wrappers.py`, `nbx_tensor.py`, `dispatch.py`):
```bash
python -m neurobrix run --model TinyLlama-1.1B-Chat-v1.0 \
  --prompt "How much is 2+2? Give me just the number." \
  --triton --seed 42 2>&1 | tail -5
# MUST contain "4"
```
