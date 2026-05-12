# S3 — P-TRITON-CPU-INTEGRATION — readiness + plan (2026-05-12)

This document captures the result of the mandate-required web research on
upstream `triton-cpu` readiness and the resulting implementation plan
for S3. It is a handover for the next session.

## Upstream readiness (as of 2026-05-12)

- **Active repo**: `triton-lang/triton-cpu` (last push 2026-04-21).
  The `meta-pytorch/triton-cpu` fork is stale (last push 2025-08-18) —
  do not target the meta-pytorch fork.
- **Status**: officially marked "experimental / work in progress" in the
  upstream README; tracks Triton main; current LLVM pin
  `87717bf9f81f`.
- **Driver/runtime contract** (third_party/cpu/backend/driver.py):
  - `triton.runtime.driver.active` resolves to `CPUDriver`
    automatically when no GPU is visible (build with
    `TRITON_CPU_BACKEND=1`).
  - `getPointer()` accepts either a Python int or any object exposing
    `.data_ptr() → int64`. **NBXTensor already satisfies this contract
    verbatim** — no abstraction rewrite required.
  - Pointers are flat host addresses; no `cudaMalloc` equivalent.
    `posix_memalign` / `aligned_alloc` (64-byte alignment for AVX512)
    on host RAM is sufficient. Swap the `DeviceAllocator`
    implementation only.
  - `pStream` arg is accepted but **ignored** — execution is
    synchronous via OpenMP parallel-for over the grid. `device_sync`
    becomes a no-op on CPU.
  - fp16/bf16 scalar args are coerced to `float` at the launcher
    boundary; tensor payloads keep their half-precision bytes.

## Kernel coverage parity

| Op pattern | CPU backend status | NeuroBrix dependency |
|---|---|---|
| `tl.dot` matmul (fp32, bf16) | works; AVX512-BF16 path slow (#229 open) | mm/bmm/addmm |
| `tl.dot` fp16 Dot3D | **#147 open since 2024-09 — accuracy gap, not bit-similar** | mm/bmm in fp16 paths |
| `tl.sum`, `tl.max` reductions | works (softmax/layernorm tutorials ship) | rms_norm, softmax |
| Masked `tl.load`/`tl.store` | works on slow path | conv2d implicit-gemm halo |
| `make_block_ptr` fast GEMM with masks | **#222 open — does not support masks** | conv2d implicit-gemm fast path |
| FlashAttention pattern | works (06-fused-attention.py tutorial) | sdpa, flash_attention.py |
| Round-to-zero fp32 conversion | **#58 open — not supported** | low-impact for us |
| LLVM unrealized_conversion_cast on mixed precision | **#144 open — regression risk** | mixed-precision ops |
| torch 2.6+ build path | **#233 open — incompatibility** | install gate |

## Numerical risk summary

- **fp32 / bf16 paths** → feasible today. TinyLlama (fp32 registry
  default) is the natural first validation target. Sana 1024 BF16
  (already exists in the registry as `Sana_1600M_1024px_MultiLing`)
  is the natural second.
- **fp16 paths** → **upstream-blocked** by #147 (Dot3D accuracy) and
  #222 (masked GEMM fast path). Sana 4Kpx fp16 is in this category.
  This is a legitimate "épuisement technique" escalation candidate
  per the P-PRISM-NEVER-REFUSE v2 mandate — NeuroBrix cannot close
  the fp16 gap internally.

## Implementation plan (5 stages, ~250-350 lines)

1. **Stage 1 — install gate + driver switch (~40 lines).**
   - Add `triton-cpu` install detection in `cli/__init__.py`. Skip
     gracefully when missing (no hard crash on hosts without it).
   - When `--triton` / `--triton-sequential` is invoked on a CPU-only
     profile, set `os.environ["TRITON_CPU_BACKEND"] = "1"` BEFORE any
     triton import so the CPU driver registers as `active`.
   - Verify with `triton.runtime.driver.active.get_current_target()`
     resolving to a CPU target.

2. **Stage 2 — DeviceAllocator CPU backend (~80 lines).**
   - Extend `kernels/utils/device_allocator.py` with a CPU branch:
     `posix_memalign(64, nbytes)` → returns int64 host address.
     `free` via `ctypes.libc.free`.
   - NBXTensor methods `.data_ptr()`, `.numel()`, `.element_size()`,
     `.shape`, `.stride()` are dtype-string-driven and already
     device-agnostic — no edit needed.
   - `device_sync` and `device_empty_cache` become no-ops on CPU
     (already part of the `core/device_utils.py` indirection).

3. **Stage 3 — Triton wrapper CPU device routing (~60 lines).**
   - In `kernels/wrappers.py`, replace any explicit
     `triton.runtime.driver.active.set_current_device(idx)` with a
     CPU-aware call (no-op when target is CPU). Pattern is already
     factored — single audit pass.
   - Verify each wrapper accepts an NBXTensor whose `.device` is
     `"cpu"`. The kernel launches `_kernel[grid](...)` with `data_ptr`
     and the driver does the rest.

4. **Stage 4 — TritonSequence / TritonSequentialDispatcher
   device-awareness (~50 lines).**
   - Both currently assume `device_idx: int`. Add a CPU target path
     that skips `cudaMallocHost`-style operations and routes arena
     allocation through the CPU `DeviceAllocator` branch.
   - `kv_cache.py` similarly: pre-allocate host RAM buffers instead
     of CUDA buffers.

5. **Stage 5 — Validation matrix (R29 artefacts).**
   - TinyLlama --triton CPU pure → coherent text (fp32 path).
   - TinyLlama --triton-sequential CPU pure → coherent text.
   - Sana 1024 BF16 --triton CPU pure → coherent red apple
     (validates bf16 path, slow per #229 — accept).
   - Sana 1024 BF16 --triton-sequential CPU pure → coherent red apple.
   - Sana 4Kpx fp16 --triton CPU pure → **expected to fail or numerical
     drift**; document the upstream gap and escalate per mandate
     "épuisement technique" clause.

## Open questions for Hocine

1. Is the bf16 path acceptable as the S3 closure target, with
   fp16 Sana 4Kpx CPU-pure escalated as an upstream-block (vs
   waiting for #147 to close)?
2. Is install-on-demand (download triton-cpu wheel in the venv at
   first `--triton` CPU invocation) acceptable, or must the user
   install it manually?

## Pre-S3 anti-régression baseline (committed in `a3d2248`)

- TinyLlama compiled GPU PASS, sequential GPU PASS, sequential CPU
  PASS (S2), triton-sequential GPU PASS (R30 mirror).
- Sana 1600M 1024px sequential GPU PASS coherent red apple.
- PixArt-XL 1024 sequential GPU PASS coherent red apple.

## Resumption command

In the next session, start by:

```bash
git log --oneline -5    # confirm at a3d2248 or descendant
cat validation_outputs/p_prism_never_refuse_s2/S3_READINESS_AND_PLAN.md
```

Then begin Stage 1.
