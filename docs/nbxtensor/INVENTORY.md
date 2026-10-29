# nbxtensor — inventory of the scatter, 28/10/2026

Read by the supervisor on lot-2026-10-28 ced5ee7b (the branch point of
`nbxtensor`). The machine-readable form is `migration_list.tsv`, produced by
`nbxtensor/tools/scatter_scan.py` and kept honest by
`tests/unit/nbxtensor_boundary/test_the_scatter_only_shrinks.py`.

## The monolith

`src/neurobrix/kernels/nbx_tensor.py`, 4 681 lines, mixes six concerns:
malloc tracing (l. 85-170), the dtype enum and bf16 carriers (l. 174-306),
the torch boundary (l. 326, 693-800), backend capability tables and device
faults (l. 420-690), the allocator `DeviceAllocator` (l. 1040-2600, ~1 560
lines), backend detection (l. 2603-2820), strided copy / broadcast / slicing
helpers (l. 2829-3215), and the class `NBXTensor` (l. 3215-end).

## Facts that exist more than once

| Fact | Copies |
|---|---|
| bytes per element | `kernels/nbx_tensor.py:306 dtype_size` (`_DTYPE_SIZES`), `triton/itemsize.py:80`, `core/prism/memory_estimator.py:189 get_dtype_bytes_per_element` (plus `core/dtype/itemsize.py`, the PyTorch branch's mirror, legitimate) |
| bf16 <-> fp32 / fp16 conversion | `kernels/nbx_tensor.py:245,276`, `kernels/autotune_certify.py:269,302`, `triton/weight_loader.py:110,456` |
| allocator | `kernels/nbx_tensor.py:1040 DeviceAllocator`, `triton/replay.py:173 SlabAllocator` (+ `malloc`, `malloc_cuda`), `triton/arena.py`, `triton/memory_pool.py`, `kernels/metal_device.py` |
| the allocator's behaviour, re-derived | `core/prism/solver.py:1757 _arena_mb`, `:1762 _arena_over_allocation`, `:1788 _store_arena_over_allocation`, `:1846 _arena_door`; the coefficients `triton_arena_activation_factor` (1.3) and the 0.92 whole-component fraction |
| type promotion | `triton/dtype.py:1069 promote_func`, `kernels/wrappers.py:944 _promote_int_unary` |
| numel | `triton/expert_store.py:42`, `core/prism/work.py:56` (and NBXTensor's own) |
| the view rule | `kernels/dispatch.py:156 _view_strides` — outside the tensor that it describes |

## Size of what moves

`kernels/` (excluding the never-executed reference tree `triton_kernels_ref/`, deleted 2026-10-29):
22 modules and 174 kernel files under `ops/`; `kernels/wrappers.py` alone is
9 722 lines and will be split by op family. `triton/`: 9 modules move
(itemsize, arena, memory_pool, device_transfer, launcher_contract,
triton_ext_driver, metal_backend, cpu_backend, replay_graph); the rest is engine.

## What is already right

No torch import under `triton/` (60 files); torch under `kernels/` only in the
4 files the R33 guard lists; importing Prism, the DtypeEngine, the Triton
sequence, the flows, the weight loader or the CLI loads no torch (checked
28/10 20:00). The tensor computes right: gate outputs identical or judged.
