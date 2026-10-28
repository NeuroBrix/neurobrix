# nbxtensor — migration plan (before the stable)

Order decided by the supervisor on 28/10/2026 20:45 after Hocine's decision of
20:18. Each phase ends on proofs, never on a report. The representative set
for "identical outputs" (sha against the judged artifacts, Triton engine,
entry point, the machine's profile): TinyLlama, Qwen3-30B-A3B (4 cards),
whisper-large-v3-turbo, Kokoro-82M, PixArt-Sigma-XL-1024, Real-ESRGAN x4plus,
Janus-Pro-7B, Wan2.2-I2V-A14B (16g), mochi-1-preview (8 GB rung) — and on the
Mac the same families on Metal.

## M0 — boundary (supervisor, done 28/10)
Branch `nbxtensor`, package skeleton `nbxtensor/`, DESIGN / INVENTORY /
MIGRATION, `scatter_scan.py`, `migration_list.tsv`, the two boundary guards.

## M1 — move (Dell alone, one writer, no parallel edit of the moved files)
Freeze first: every in-flight fix lands on lot-2026-10-28 and is merged into
`nbxtensor` once; from then on new work goes to `nbxtensor` only.
Move module by module into `nbxtensor/src/nbxtensor/<area>/` with `git mv`
(history kept), rewrite the imports in the same commit, no re-export shim.
Order: dtype + itemsize (the one table), random, quant, backend, memory,
ops (directory as is; wrappers.py as is), then nbx_tensor.py itself —
WITHOUT its torch hand-off helpers (nbx_to_torch, nbx_dtype_to_torch,
set_torch_device, torch_device_str), which stay in the engine
(R-NBXT 6) until DLPack replaces them.
Per commit: unit suites + boundary guards green. Per area: no-card coverage
green + representative set identical. The scatter list shrinks commit by commit.
**Exit:** `kernels/` keeps only engine files (census, classification,
metadata_ops of the PyTorch branch); the engine imports nbxtensor.

## M2 — one fact once (Dell and Mac in parallel, by area)
- Mac: `dtype/` (one table, one conversion module replacing the three bf16
  copies, promotion in one place), `shape/` (view rule from dispatch.py,
  broadcast, strides, numel), Metal backend and allocator; card-less tests.
- Dell: split `nbx_tensor.py` (tensor / memory / backend), unify the
  allocators (DeviceAllocator + replay SlabAllocator + arena + pool behind one
  interface), split `wrappers.py` by op family into `ops/`.
**Exit:** migration_list.tsv holds only Prism's re-derivations (M3).

## M3 — cost declared, Prism asks (Dell, Mac reviews the Prism side)
`op.cost(...)` for every op, from launch_keys; `memory.plan(...)` returns the
peak of a plan. Prism's `_arena_*` functions, `get_dtype_bytes_per_element`,
the 1.3 factor and the 0.92 fraction are deleted. **Exit:** for every cell of
the representative set, Prism's priced peak and the measured peak agree
within the measured allocator envelope, printed per cell; migration_list.tsv
empty.

## M4 — benchmarks against torch (cards busy during M1-M3)
Mac writes `benchmarks/nbxtensor_vs_torch/` first (outside the library) (kernel level: the census shapes; model
level: the gate cells), then both machines run it: torch eager,
torch.compile, nbxtensor, per device. Runs ALONE on the machine (no other
process). **Exit:** a table per device, kernel and model, with the gaps
named; the losses become the optimisation queue (rule 14).

## M5 — final gate S7 on the extracted engine
One gate, scope = everything (the move touches every Triton cell), times =
benchmark, plus the R33 runtime audit (torch_provenance) per architecture,
plus the Metal pass. Then model-status, then the stable on Hocine's GO.

## M6 — publication (Hocine's GO only)
`NeuroBrix/nbxtensor` repository with history, README, API docs, the benchmark
table; PyPI later.
