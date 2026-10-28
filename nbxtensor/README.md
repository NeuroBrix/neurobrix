# nbxtensor

An inference-first tensor library, torch-free.

PyTorch is a general framework built for research and training in eager mode:
dynamic graphs, autograd, a generic dispatcher, a caching allocator that
discovers memory as the program runs. Inference is a narrower problem, and a
better-informed one: the graph and the shapes are known before the first run.
nbxtensor is built for that case.

- **Memory is planned, not discovered.** Allocators per device behind one
  interface, and a static planner that assigns every buffer before the first run.
- **Every kernel declares its cost.** Outputs, workspace and scratch with their
  lifetimes, bytes moved, flops. A planner prices a model from these
  declarations instead of guessing what the allocator will do.
- **Capture and replay are first-class.** A repeated sequence of launches is
  recorded, verified byte-equal and replayed.
- **One table of facts.** One dtype table, one view rule, one conversion
  per pair of dtypes, one capability table per backend — declared, never guessed.
- **Backends:** NVIDIA CUDA, Apple Metal (through triton-ext), CPU.
- **No torch, ever.** nbxtensor never imports torch, not even optionally.
  Exchange with any tensor library goes through the DLPack protocol
  (`__dlpack__`, `from_dlpack`), so a torch user writes `torch.from_dlpack(t)`
  on their side. The benchmarks against torch eager and `torch.compile` live
  outside the library (`benchmarks/nbxtensor_vs_torch/` in the NeuroBrix repository).

nbxtensor is extracted from NeuroBrix, the universal inference engine whose
Triton engine is its first client. It lives in `nbxtensor/` of the NeuroBrix
repository until the extraction is complete, then in its own repository.

Status: **M0** — the boundary is drawn; the code moves in during M1.
Design, inventory and migration plan: `docs/nbxtensor/` in the NeuroBrix repository.

License: Apache 2.0.
