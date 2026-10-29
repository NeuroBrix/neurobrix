# nbxtensor — design

Decided by Hocine on 28/10/2026 at 20:18: NBXTensor is NeuroBrix's master
piece and a library of its own, published as open source (Apache 2.0) under
the name **nbxtensor**. Its extraction happens **before** the stable release,
because the stable must not ship tensor semantics scattered across the engine.
Written by the supervisor; the agents implement it (MIGRATION.md).

## 1. Why

Banning torch from the Triton branch was meant to force one tensor library to
grow as the engine grew. It produced a working tensor (outputs judged
identical across the gate), but not a library: on 28/10 the tensor's semantics
were spread over the engine (INVENTORY.md). A planner that re-derives what the
allocator does with coefficients (0.92, 1.3) diverges from the allocator, and
each divergence cost days. The cure is a boundary: the facts live once, in
nbxtensor; the engine asks.

What nbxtensor competes on is NOT "Triton kernels" — `torch.compile` already
generates Triton. It is what an inference library can know that an eager
framework cannot: the whole graph and its shapes before the first run, so
memory can be planned, launches captured and replayed, precision and placement
decided globally. Precedents that won their niche this way: llama.cpp/ggml,
Apple MLX, tinygrad, Hugging Face candle. (vLLM did not: it is built on torch.)

## 2. Rules (R-NBXT, permanent, in addition to the engine's rules)

1. **One library, one boundary.** `nbxtensor` imports nothing from `neurobrix`
   (guard: tests/unit/nbxtensor_boundary/test_the_library_imports_no_engine.py).
   The Triton branch takes every tensor fact from nbxtensor's public API.
2. **Each fact exists once.** One dtype table and one itemsize reader, one
   conversion per dtype pair, one view rule, one broadcast rule, one promotion
   rule, one allocator family, one backend capability table. A second copy is
   a defect (guard: test_the_scatter_only_shrinks.py — the list only shrinks).
3. **Mechanism in the library, policy in the engine.** nbxtensor says what a
   dtype is, how a view strides, how memory is allocated and planned, what an
   op costs. Prism decides placement and strategy, the DtypeEngine decides
   widths, the flows orchestrate. The library never reads a NeuroBrix profile,
   family file or model name.
4. **Every op declares its cost.** `op.cost(shapes, dtypes, device)` returns
   outputs, workspace and scratch with lifetimes, bytes moved and flops — from
   the same launch decisions the kernel executes (launch_keys). Prism prices
   from these declarations. No coefficient in the engine stands in for a term
   the library did not declare.
5. **Memory is planned.** An allocator interface per device, and the static
   planner (greedy-by-size over known lifetimes) as a first-class API. Device
   facts the library needs (allocator envelope, runtime growth, bandwidths)
   are MEASURED by the library and refused by name when absent.
6. **Zero torch, everywhere in the library.** nbxtensor never imports or
   uses torch, not even as an optional dependency (Hocine, 28/10 20:33).
   Exchange with other tensor libraries goes through DLPack: nbxtensor
   implements `__dlpack__` / `__dlpack_device__` and `from_dlpack(any
   producer)`; the other side imports itself. The engine's current hand-off
   helpers (`nbx_to_torch`, `nbx_dtype_to_torch`, `set_torch_device` in
   kernels/nbx_tensor.py) NEVER move into nbxtensor: they stay in the engine
   until DLPack replaces them. The comparison against torch is a separate
   harness outside the library (`benchmarks/nbxtensor_vs_torch/`).
7. **Zero hardcode, zero fallback** (inherited from the engine).
8. **A move changes no behaviour.** A module moves with its tests; the move
   proves identical outputs (sha) on the representative set and a green
   no-card coverage. No kernel is rewritten during a move.
9. **Benchmarks go with the library, outside it.** Kernel level and model level,
   against torch eager and `torch.compile`, per device (V100 16/32 GB, M4 Pro).
   "Faster than torch" is claimed only cell by cell, measured.
10. **A public API, documented.** Exported names are the API; everything else
    is `_private`. A new export comes with its docstring and its test. The
    README is written for developers who will use nbxtensor, and every code
    example in it is executed by a test, so the README cannot lie.
11. **An API a Rust core can implement later** (Hocine, 28/10 20:53). Version 1
    is Python; version 2 will be a Rust core behind the SAME Python API
    (PyO3 + maturin wheels, as tokenizers, safetensors, polars and
    pydantic-core did), usable from Rust and C as well. So the v1 API uses
    plain data at its edges (dtypes, shapes, strides, device ids, byte
    counts, op descriptions as data), never Python-only behaviour that a Rust
    core could not reproduce: no monkey-patching, no duck-typed inputs, no
    dynamic attributes, no state hidden in module globals. Triton kernels stay
    Triton in v2; Rust would launch them precompiled.

## 3. Layout

```
nbxtensor/                     # own pyproject, own README/LICENSE; future own repository
  src/nbxtensor/
    dtype/      the dtype table (config/dtypes.yml moves here), NBXDtype, itemsize,
                conversions (bf16/fp16/fp32 carriers), promotion, range safety
    shape/      shapes, strides, contiguity, the view rule (ATen computeStride),
                broadcasting, slicing
    tensor.py   NBXTensor: the core class only (no allocator, no backend table inside)
    memory/     Allocator interface; cuda, metal, cpu allocators; arena and pool;
                slab allocator for replay; host pinned memory; transfers (D2D, peer, H2D);
                static planner; oom advice
    backend/    detection, capability table, launcher and launcher contract, Metal driver
                (triton-ext), CPU backend, device faults channel, graph capture/replay primitive
    ops/        the kernel files, the op table, launch keys, autotune (bucket, certified store,
                refusals, screen oracle), and the cost declaration of every op
    random/     seeded streams and torch-free draws
    quant/      QuantizedTensor
    io/         safetensors -> NBXTensor primitives (the engine keeps its loading policy)
    interop/    numpy and DLPack (the only exchange protocol; never torch)
  tools/        scatter_scan.py
benchmarks/nbxtensor_vs_torch/   OUTSIDE the library: imports both nbxtensor and torch
                            to measure one against the other
```

What stays in the engine: Prism, the DtypeEngine's policy (AMP op sets,
calibration), the flows, the execution modes (sequence, sequential, FrozenPlan
replay), census and certification of graphs, KV cache, expert store, the
weight loader's policy, symbolic shapes, the TilingEngine's decisions.

## 4. Public API (target of M2)

`nbxtensor.Tensor` (NBXTensor), `nbxtensor.dtype` (`float16` …, `itemsize`,
`convert`), constructors (`empty`, `zeros`, `from_numpy`, `to_numpy`),
shape ops (`view`, `reshape`, `permute`, slicing), `nbxtensor.device`,
`nbxtensor.memory` (`Allocator`, `plan`), `nbxtensor.ops.<name>` with
`.cost(...)`, `nbxtensor.capture` / `replay`, `nbxtensor.random`.
The exact names are fixed in M2 by the agents and reviewed by the supervisor.

## 5. Version 2 in Rust (after the stable)

Decided by Hocine on 28/10 20:53: v1 is Python and is finished first
(grouped, tested, published, the engine on it, the stable released). The Rust
rewrite of the core is the following project. Expected gain, to be measured by
the M4 benchmark before anything is ported: the host cost of each launch
(tensor metadata, allocator, argument packing — microseconds in Rust against
the current Python path), no GIL for loading and planning, memory safety in the
allocator and the view rule, and a library usable without Python. Not gained:
GPU time (kernels stay Triton). Candidates, in order the benchmark will
confirm: tensor metadata, allocators and the memory planner, the launcher.

## 6. Distribution

Developed in `nbxtensor/` of the NeuroBrix repository on branch `nbxtensor`,
installed by the engine as a path dependency. Extracted to its own repository
(`NeuroBrix/nbxtensor`, history kept with git filter-repo) and published on
Hocine's GO only. Forge never enters it.

## 7. Versions

Decided by Hocine on 29/10 15:52. The first publication of nbxtensor is **0.0.1**: no alpha, beta or release-candidate
label, ever. Every fix published after it increments the version (0.0.2, 0.0.3, …); the Rust rewrite of section 5 is
**2.0.0**. The one number lives in `nbxtensor/pyproject.toml` and `nbxtensor.__version__`, and a test holds them equal
(`nbxtensor/tests/test_version.py`).
