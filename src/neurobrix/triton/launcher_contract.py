"""What a device driver must present to the engine's kernel launcher.

The launcher itself is not here. It is an engine component, vendor-agnostic,
and it belongs in the dispatch layer with CUDA as its first client. This
module is the **contract** that launcher will program against: the smallest
surface a driver has to expose for the launcher to compile a kernel once and
dispatch it many times, on any backend.

It exists because the Metal port had to build a driver anyway — upstream
Triton's own launcher imports torch inside its argument binder, on every
backend, and R33 does not have an Apple exception — and building it made the
surface concrete. `triton/metal_driver.py` is the first implementation and it
passes the checker below. A CUDA implementation has to pass the same checker,
unchanged, or the launcher would be programming against Metal's habits rather
than against an interface.

## The surface, in one paragraph

A driver **compiles** a `@triton.jit` function plus a signature and a set of
`tl.constexpr` values into a `CompiledKernel`. That object carries the
compiled artifact, the argument binding the launcher must fill, the
threadgroup/block width, and the shared-memory request. The launcher then
**launches** it with a grid and a flat argument list, where every device
buffer is passed as an **integer pointer** and every scalar as a typed Python
value. Ordering is expressed with **streams** and **events**, both opaque
integer handles.

Nothing in that paragraph is Metal-shaped, and that is the point.

## What the contract deliberately does NOT include

* **Tensors.** The launcher passes pointer integers, not container objects. A
  driver never learns what an NBXTensor is, and NBXTensor never learns what a
  backend is; the allocator already owns that seam
  (`kernels/nbx_tensor.py::DeviceAllocator`).
* **Autotuning, caching, heuristics.** Choosing `BLOCK_SIZE` is the engine's
  business and it is the same decision on every backend. A driver is asked to
  compile what it is given.
* **A device abstraction.** Allocation, copies and device queries are the
  allocator's, not the driver's. The driver receives addresses that already
  exist.
* **torch.** On any backend. A driver that needs torch to bind an argument
  has put the framework back in the execution path, which is the thing R33
  forbids and the reason this contract was written at all.

R33 preserved — no torch, at the boundary included.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

#: Scalar kinds a launcher may be asked to pack. A driver must accept every
#: one of these spellings, because they are Triton's own signature spellings
#: and the launcher passes them straight through.
SCALAR_DTYPES = ("i1", "i8", "i16", "i32", "i64",
                 "u8", "u16", "u32", "u64",
                 "fp16", "bf16", "fp32", "fp64")


@dataclass(frozen=True)
class ArgSlot:
    """One argument the launcher must supply, in binding order.

    `index` is the driver's own binding index, not the position in the Python
    call: a backend may reserve indices or reorder, and the launcher must not
    have to know which. It fills the slots in the order they are given.
    """

    index: int
    name: str
    is_pointer: bool
    dtype: str

    def __post_init__(self):
        if not self.is_pointer and self.dtype not in SCALAR_DTYPES:
            raise ValueError(
                f"{self.name}: {self.dtype!r} is not a scalar dtype the "
                f"launcher can pack; expected one of {SCALAR_DTYPES}")


@runtime_checkable
class CompiledKernel(Protocol):
    """The result of compiling once, launched many times."""

    #: The kernel's entry-point name, as the backend knows it.
    name: str

    #: The compiled artifact, opaque to the launcher, in whatever form this
    #: backend reloads without recompiling: a cubin, a metallib, an MSL
    #: source. The launcher only ever caches and hands it back.
    binary: bytes

    #: What `binary` is — "cubin", "metallib", "msl", "hsaco". Declared so a
    #: cache can refuse an artifact built for another backend instead of
    #: loading it and failing later.
    binary_kind: str

    #: Threads per block / threadgroup.
    block_size: int

    #: Bytes of shared / threadgroup memory the kernel requests.
    shared_memory: int

    #: The `tl.constexpr` values this kernel was compiled with. Part of the
    #: identity of the compilation, so a cache keyed on the source alone is
    #: wrong and this makes that visible.
    constexprs: Mapping[str, Any]

    #: The specialization markers this kernel was compiled with — the same
    #: mapping `compile()` was given. Also part of the compilation's
    #: identity: the same source and the same constexprs, compiled once
    #: claiming 16-byte alignment and once not, are two different kernels,
    #: and handing one back for the other is silently wrong numbers.
    specialization: Mapping[str, str]

    #: The arguments the launcher must supply, in binding order.
    binding: Sequence[ArgSlot]

    def launch(self, grid: Sequence[int], args: Sequence[Any],
               stream: int = 0) -> None:
        """Dispatch `grid` blocks.

        `args` is positional and matches `binding` one for one. A slot with
        `is_pointer` takes an **integer device address**; every other slot
        takes a Python int or float, which the driver packs to `dtype`.

        `stream` is 0 for the default stream, or a handle from
        `create_stream`. The call returns once the work is enqueued; ordering
        against other work is the caller's, expressed with events.
        """


@runtime_checkable
class LauncherDriver(Protocol):
    """What the launcher needs from a backend, and nothing more."""

    #: "cuda", "hip", "metal". The launcher uses it for diagnostics and for
    #: refusing a cached artifact from another backend, never to branch on.
    backend: str

    def compile(self, jit_fn, signature: Mapping[str, str],
                constexprs: Mapping[str, Any],
                num_warps: int = 4,
                specialization: Mapping[str, str] | None = None,
                num_stages: int | None = None,
                ) -> CompiledKernel:
        """Compile one `@triton.jit` function.

        `signature` maps parameter name to a Triton signature string —
        `"*fp32"` for a pointer, `"i32"` / `"fp32"` for a scalar, and
        `"constexpr"` for a compile-time parameter whose value is in
        `constexprs`. Raises rather than returning a partial result.

        `specialization` maps parameter name to Triton's per-argument marker:
        `"D"` when the argument is a multiple of 16 (a pointer's address, an
        integer's value), `""` when it is not, absent when the argument takes
        no marker at all. It is NOT decoration — the compiler turns "D" into
        a divisibility attribute and vectorizes loads and stores on it, so a
        driver that ignores this argument compiles a different kernel from
        the one the launcher asked for, at the same shape, with the same
        arguments. Measured on Metal 2026-09-05: dropping it changed
        rms_norm's fp16 output at two of four shapes while leaving fp32
        untouched.

        A driver is free to have no use for it, but must then say so
        explicitly rather than accept and discard it.

        `num_stages` is the software-pipelining depth, which an autotuner
        config carries alongside `num_warps`. Measured on Metal 2026-09-05:
        it changes nothing in the emitted MSL, because that lowerer does not
        pipeline — but it changes CUDA's generated code substantially, so the
        launcher passes it and each backend decides. Dropping it in the
        launcher would silently discard the autotuner's choice on the first
        backend that reads it.
        """

    # -- ordering ------------------------------------------------------------

    def create_stream(self) -> int:
        """A new stream. Never returns 0, which is the default stream."""

    def destroy_stream(self, stream: int) -> None:
        ...

    def synchronize_stream(self, stream: int) -> None:
        """Block until everything on `stream` has retired."""

    def create_event(self, timing: bool = False) -> int:
        """An event handle. `timing=False` is an ordering handle and must
        REFUSE to be read as a stopwatch — see `elapsed_ms`."""

    def destroy_event(self, event: int) -> None:
        ...

    def record_event(self, event: int, stream: int = 0) -> None:
        ...

    def synchronize_event(self, event: int) -> None:
        """Block until `event` has been reached."""

    def wait_event(self, stream: int, event: int) -> None:
        """Make `stream` wait for `event` WITHOUT blocking the host."""

    def elapsed_ms(self, start: int, end: int) -> float:
        """Milliseconds between two recorded timing events.

        Must raise for an event created with `timing=False`: an ordering
        handle is not a clock, and returning a plausible number instead of
        refusing is the failure this method exists to prevent.
        """


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

def verify_driver_contract(driver, jit_fn, signature, constexprs,
                           make_buffer, read_buffer, free_buffer,
                           grid, args_builder, expected):
    """Exercise a driver against the contract. Returns a list of failures.

    Written as a plain function rather than a test so a backend can run it
    from anywhere — a CI job, a `doctor` subcommand, a notebook on a machine
    none of us has. `tests/unit/triton/test_launcher_contract.py` calls it
    for Metal.

    It exercises the driver **exactly as `kernels/launcher.py` does**: Triton
    compiles for the target the driver names, the driver loads the artifact
    it declared, and the driver launches it with the launcher's own
    `(kind, value)` parameter list. A checker that used a private path of its
    own would certify a driver the engine cannot actually drive.

    The callables are the caller's, because allocation is the allocator's job
    and this module must not grow one:

    * `make_buffer(nbytes) -> int` — a device address.
    * `read_buffer(addr, nbytes) -> bytes`
    * `free_buffer(addr) -> None`
    * `args_builder(pointers) -> list` — the flat argument list, given the
      pointers it asked for by allocating them.
    """
    from neurobrix.kernels.launcher import _pack_param

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)
        return condition

    # -- the declared surface ------------------------------------------------
    check(isinstance(getattr(driver, "artifact_kind", None), str)
          and driver.artifact_kind,
          "a driver must declare `artifact_kind` — the key of its artifact "
          "in CompiledKernel.asm — so a cache cannot hand one backend "
          "another's binary")
    check(isinstance(getattr(driver, "wants_scratch_params", None), bool),
          "a driver must declare `wants_scratch_params`: whether its launch "
          "ABI carries Triton's two trailing scratch pointers")

    target = driver.target()
    check(isinstance(getattr(target, "backend", None), str) and target.backend,
          f"target() must name a backend, got {target!r}")
    check(int(getattr(target, "warp_size", 0)) > 0,
          f"target() must give a positive warp size, got {target!r}")

    # -- compile, load, launch: the launcher's own three steps ---------------
    import triton
    from triton.compiler.compiler import ASTSource

    compiled = triton.compile(
        ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs),
        target=target, options={"num_warps": 4})
    metadata = compiled.metadata

    artifact = compiled.asm.get(driver.artifact_kind)
    if not check(artifact is not None,
                 f"the compiler produced {sorted(compiled.asm)} and the "
                 f"driver takes {driver.artifact_kind!r}: nothing to load"):
        return failures

    block = driver.block_for(metadata)
    check(isinstance(block, tuple) and len(block) == 3
          and all(isinstance(b, int) and b > 0 for b in block),
          f"block_for() must return a 3-tuple of positive ints, got {block!r}")

    function = driver.load(artifact, metadata.name, metadata.shared)
    check(function is not None, "load() returned nothing")

    # -- the launch produces the right bytes, at a grid wider than one block -
    pointers = [make_buffer(n) for n in args_builder.buffer_sizes]
    try:
        values = args_builder(pointers)
        ordered = [(name, ty) for name, ty in signature.items()
                   if ty != "constexpr"]
        if not check(len(ordered) == len(values),
                     f"the signature has {len(ordered)} runtime parameters "
                     f"and args_builder gave {len(values)}"):
            return failures
        params = [_pack_param(ty, v) for (_n, ty), v in zip(ordered, values)]
        if driver.wants_scratch_params:
            params = params + [("ptr", 0), ("ptr", 0)]

        driver.launch(function, grid, block, metadata.shared, 0, params)
        got = read_buffer(pointers[args_builder.output_index],
                          args_builder.output_bytes)
        check(got == expected,
              "the launch produced the wrong bytes. The grid is deliberately "
              "wider than one block: a dropped program offset passes a "
              "single-block launch and fails here.")

        # A wrong-length argument list must be refused, not padded.
        try:
            driver.launch(function, grid, block, metadata.shared, 0,
                          params[:-1])
            failures.append("launch() accepted an argument list of the wrong "
                            "length instead of refusing it")
        except Exception:
            pass

        # A pointer the allocator never handed out must be refused. This is
        # the ownership rule: the engine owns device memory, and a launch
        # that reads an address from anywhere else is undefined behaviour
        # the driver is in the best position to catch.
        foreign = list(params)
        for i, (kind, _v) in enumerate(foreign):
            if kind == "ptr":
                foreign[i] = ("ptr", 0x1000)
                break
        try:
            driver.launch(function, grid, block, metadata.shared, 0, foreign)
            failures.append("launch() accepted a device address the "
                            "allocator never handed out")
        except Exception:
            pass
    finally:
        for address in pointers:
            free_buffer(address)

    # -- events --------------------------------------------------------------
    ordering = driver.create_event(timing=False)
    timing_a = driver.create_event(timing=True)
    timing_b = driver.create_event(timing=True)
    try:
        driver.record_event(ordering, 0)
        driver.synchronize_event(ordering)
        second = driver.create_stream()
        driver.wait_event(second, ordering)
        driver.synchronize_stream(second)
        driver.destroy_stream(second)

        driver.record_event(timing_a, 0)
        driver.record_event(timing_b, 0)
        driver.synchronize_event(timing_b)
        interval = driver.elapsed_ms(timing_a, timing_b)
        check(isinstance(interval, float) and interval >= 0.0,
              f"elapsed_ms must return a non-negative float, got {interval!r}")

        try:
            driver.elapsed_ms(ordering, ordering)
            failures.append("elapsed_ms returned a number for an ordering "
                            "event; an ordering handle is not a clock")
        except Exception:
            pass
    finally:
        for e in (ordering, timing_a, timing_b):
            driver.destroy_event(e)

    return failures
