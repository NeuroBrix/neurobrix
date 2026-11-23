# The nbxtensor boundary — a library the engine imports, never the reverse

[nbxtensor](https://github.com/NeuroBrix/nbxtensor) is NeuroBrix's tensor library: the `NBXTensor`, its
devices and memory, and every compute kernel of the Triton branch. It replaces torch on that branch. It is
its own project, in its own repository, published on PyPI under Apache 2.0. The engine depends on one exact
release of it (`nbxtensor==0.1.11` in the engine's `pyproject.toml`) and never on a copy inside the engine.

## The red line

**All tensor logic lives in nbxtensor, never in the engine.** Tensors, devices, memory, arenas, copies,
kernels and replay belong to the library. The engine keeps policy: Prism, families, flows, the CLI and
serving. On the Triton branch, where the PyTorch branch calls torch, the engine calls nbxtensor in the same
place and the same way.

The boundary goes one way:

* the library imports nothing of the engine, not conditionally, not inside a function;
* the library's public API names nothing of NeuroBrix: any program can use it, the engine is one user;
* the Triton branch reaches devices only through nbxtensor;
* when something is missing on the Triton path, it is added to nbxtensor, never fetched from torch or worked
  around in the engine.

## The gates

The engine's suite holds the boundary:

- `tests/unit/nbxtensor_boundary/test_the_library_imports_no_engine.py`: the library imports no engine module.
- `tests/unit/nbxtensor_boundary/test_the_public_api_names_nothing_of_neurobrix.py`: the library's API is
  generic.
- `tests/unit/nbxtensor_boundary/test_the_triton_branch_reaches_devices_only_through_nbxtensor.py`: device
  access on the Triton branch goes through the library.
- `tests/unit/nbx_tensor/test_the_boundary_does_not_widen.py`: `nbxtensor/tensor.py` imports neither the
  engine nor torch.

The library has its own test suite in its own repository.
