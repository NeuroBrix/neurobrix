"""The NeuroBrix kernel launcher — one engine component, no vendor named.

Every `@triton.jit` kernel in the engine is launched as `kernel[grid](...)`.
That syntax belongs to Triton, and Triton's own launcher cannot be used: its
argument binder is C++ (`native_specialize_impl`) and it **imports torch to
decide whether an argument is a tensor**, on every backend, CUDA included.
R33 has no backend exception, so the launcher is ours.

## What it does

`kernel[grid](...)` still reads the same at all 64 call sites in
`wrappers.py`. This module replaces what that expression *resolves to*:

1. bind the call's arguments to the kernel's parameters;
2. compute the **specialization signature ourselves**, from the arguments, in
   pure Python — the types, the constexpr values, and Triton's own
   divisibility-by-16 and equal-to-1 markers;
3. hand signature, constexprs, grid, pointer integers and typed scalars to
   the **active driver**, through `triton/launcher_contract.py`;
4. the driver compiles once, caches, and dispatches.

## What it does not do

**It names no backend.** There is no `if metal`, no `if cuda`, anywhere in
this file. It asks a registry which driver is active. A driver registers
itself; the seam that already resolves backends
(`nbx_tensor._detect_gpu_backend`) is the one place a backend name appears,
and it appears there already.

**With no driver registered it does nothing at all.** `kernel[grid](...)`
falls through to Triton's own path, byte for byte the behaviour that shipped.
That is the CUDA path today, and it stays untouched until the machine that
owns CUDA activates a driver and re-measures the zoo behind it. Pinned by
`test_launcher_is_transparent_without_a_driver`.

R33 preserved — no torch, at the boundary included.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

# --- the driver registry ----------------------------------------------------
#
# One slot. A driver satisfying `triton.launcher_contract.LauncherDriver`
# registers itself here; the launcher asks, and never asks what it is.

_DRIVER = None
_LOCK = threading.RLock()


def register_driver(driver) -> None:
    """Install the driver the launcher will use. Idempotent per driver."""
    global _DRIVER
    with _LOCK:
        _DRIVER = driver


def unregister_driver() -> None:
    """Remove it. The launcher falls back to Triton's own path."""
    global _DRIVER
    with _LOCK:
        _DRIVER = None


def active_driver():
    """The registered driver, or None. The launcher's only question."""
    return _DRIVER


# --- specialization, recomputed here rather than asked of Triton ------------
#
# Triton computes this in C++ (`native_specialize_impl`) and imports torch on
# the way. The rules are small and total, so we compute them:
#
#   None                      -> ("constexpr", None)
#   bool                      -> ("u1", None)
#   int == 1 (specialized)    -> ("constexpr", 1)      # Triton folds 1s
#   int                       -> ("i32"|"i64", "D" if %16 == 0 else "")
#   float                     -> ("fp32", None)
#   anything with data_ptr()  -> ("*"+dtype, "D" if ptr %16 == 0 else "")
#
# `test_our_specialization_matches_triton` asserts every one of these against
# `native_specialize_impl` over a sample of real kernels and real arguments,
# because "equivalent" is not something to assert by reading.

_INT32_MIN, _INT32_MAX = -(2 ** 31), 2 ** 31 - 1

#: Triton's spelling for the element types NBXTensor can hold.
_DTYPE_NAMES = {
    "fp16": "fp16", "bf16": "bf16", "fp32": "fp32", "fp64": "fp64",
    "float16": "fp16", "bfloat16": "bf16", "float32": "fp32",
    "float64": "fp64", "int8": "i8", "int16": "i16", "int32": "i32",
    "int64": "i64", "uint8": "u8", "bool": "i1", "bool_": "i1",
    "i1": "i1", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
    "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
}


class SpecializationError(TypeError):
    """An argument the launcher cannot type. Refused, never guessed."""


def _element_type(value) -> str:
    """Triton's name for a tensor-like's element type."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise SpecializationError(
            f"{type(value).__name__} exposes data_ptr() but no dtype; the "
            f"launcher cannot name its element type")
    name = getattr(dtype, "name", None) or str(dtype)
    name = name.rsplit(".", 1)[-1]
    if name not in _DTYPE_NAMES:
        raise SpecializationError(f"unknown element type {name!r}")
    return _DTYPE_NAMES[name]


def specialize_argument(value, specialize: bool = True, align: bool = True):
    """`(type_string, specialization_key)` for one argument.

    Mirrors `native_specialize_impl(BaseBackend, value, False, specialize,
    align)`. Raises for anything it cannot type — Triton raises there too, and
    a launcher that guessed would compile a kernel for the wrong signature.
    """
    if value is None:
        return "constexpr", None
    if isinstance(value, bool):
        return "u1", None
    if isinstance(value, int):
        if specialize and value == 1:
            return "constexpr", 1
        type_name = ("i32" if _INT32_MIN <= value <= _INT32_MAX else "i64")
        if not specialize:
            return type_name, None
        return type_name, ("D" if align and value % 16 == 0 else "")
    if isinstance(value, float):
        return "fp32", None
    if hasattr(value, "data_ptr"):
        type_name = "*" + _element_type(value)
        if not specialize:
            return type_name, None
        return type_name, ("D" if align and value.data_ptr() % 16 == 0 else "")
    raise SpecializationError(
        f"failed to specialize argument of type: {type(value).__name__}")


# --- the decorator chain ----------------------------------------------------
#
# `kernel[grid]` is rarely a bare JITFunction. Two of Triton's decorators wrap
# it, and each one DECIDES something before the launch:
#
#   @triton.heuristics  computes meta-parameters from the arguments
#                       (BLOCK_SIZE_FEAT from feat_dim, EVEN_M from M...)
#   @triton.autotune    picks a Config — block sizes AND num_warps —
#                       by benchmarking, then caches the choice per key
#
# Peeling straight down to the JITFunction, which is what a launcher naively
# does, throws both decisions away: the heuristics kernels then have no value
# for their constexprs, and the autotuned ones silently run configs[0] on
# every shape. The first fails loudly, the second is a silent wrong — the
# kernel computes the right answer at the wrong speed and nothing says so.
#
# So the launcher walks the chain and honours each layer, in pure Python.
# Neither decorator needs torch: heuristics are plain callables, and the
# benchmark below times through the driver's events, which every backend
# presents because the contract requires them.

_LAUNCH_OPTIONS = ("num_warps", "num_stages", "num_ctas", "maxnreg")

#: Autotuner -> {key: Config}. Keyed by id of the Autotuner object, which is
#: process-lived, exactly like Triton's own `self.cache`.
_AUTOTUNE_CACHE: dict[int, dict] = {}

#: How the autotuner benchmark times a config. Small because these kernels
#: are the engine's hot ones and the choice is cached per shape thereafter.
_BENCH_WARMUP = 2
_BENCH_REPS = 5


def _autotune_key(tuner, named):
    """Triton's own cache key: the declared keys plus every argument dtype."""
    key = [named[name] for name in tuner.keys if name in named]
    for value in named.values():
        if hasattr(value, "dtype"):
            key.append(str(value.dtype))
    return tuple(key)


def _bench_config(driver, jit_fn, grid, args, kwargs) -> float:
    """Milliseconds for one config, timed through the driver's events.

    Torch-free by construction: `triton.testing.do_bench` is a torch
    function, and the contract already requires create_event / record_event /
    elapsed_ms of every backend precisely so the engine never needs it.
    """
    bound = _bind_arguments(jit_fn, args, kwargs)
    signature, constexprs, call_args, keys = _specialize(jit_fn, bound)
    resolved = _normalize_grid(grid, bound)
    compiled = driver.compile(jit_fn, signature, constexprs,
                              num_warps=int(kwargs.get("num_warps", 4)),
                              specialization=keys,
                              num_stages=kwargs.get("num_stages"))

    stream = driver.create_stream()
    start = driver.create_event(timing=True)
    end = driver.create_event(timing=True)
    try:
        for _ in range(_BENCH_WARMUP):
            compiled.launch(resolved, call_args, stream)

        # The launches and the events must share a stream: an event recorded
        # on one stream says nothing about work submitted on another, and a
        # benchmark that timed an empty queue would pick a config at random
        # while looking like it measured something.
        driver.record_event(start, stream)
        for _ in range(_BENCH_REPS):
            compiled.launch(resolved, call_args, stream)
        driver.record_event(end, stream)
        driver.synchronize_event(end)
        return driver.elapsed_ms(start, end) / _BENCH_REPS
    finally:
        driver.destroy_event(start)
        driver.destroy_event(end)
        driver.destroy_stream(stream)


def resolve_wrappers(driver, kernel, grid, args, kwargs):
    """Walk the decorator chain, returning `(jit_fn, kwargs)`.

    `kwargs` comes back carrying whatever the wrappers decided: the computed
    heuristics and the chosen config's meta-parameters and launch options.
    """
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    kwargs = dict(kwargs)
    for _ in range(8):                      # the chain is two deep in practice
        if isinstance(kernel, JITFunction):
            return kernel, kwargs

        if isinstance(kernel, Heuristics):
            named = {**dict(zip(kernel.arg_names, args)), **kwargs}
            for name, heuristic in kernel.values.items():
                kwargs[name] = heuristic(named)
                named[name] = kwargs[name]
            kernel = kernel.fn
            continue

        if isinstance(kernel, Autotuner):
            kwargs = _autotune(driver, kernel, grid, args, kwargs)
            kernel = kernel.fn
            continue

        if hasattr(kernel, "fn"):
            # An unknown wrapper. Refuse rather than peel: peeling is exactly
            # how the two known ones would have been silently dropped.
            raise SpecializationError(
                f"the launcher does not know the decorator "
                f"{type(kernel).__name__!r} wrapping "
                f"{getattr(kernel, '__name__', kernel)!r}. It decides "
                f"something before the launch and the launcher will not "
                f"guess what. Teach resolve_wrappers about it.")

        raise SpecializationError(f"not a Triton kernel: {kernel!r}")

    raise SpecializationError("decorator chain deeper than 8; refusing")


def _autotune(driver, tuner, grid, args, kwargs):
    """Triton's autotuning decision, made here and cached the same way."""
    if len(tuner.configs) == 1:
        config = tuner.configs[0]
    else:
        if tuner.early_config_prune or tuner.perf_model:
            raise SpecializationError(
                f"{tuner.base_fn.__name__}: autotune uses config pruning or a "
                f"perf model, which the launcher does not reproduce. Refused "
                f"rather than benchmarked against a different config set.")

        named = {**dict(zip(tuner.arg_names, args)), **kwargs}
        key = _autotune_key(tuner, named)
        cache = _AUTOTUNE_CACHE.setdefault(id(tuner), {})
        if key not in cache:
            jit_fn = tuner.fn
            while not hasattr(jit_fn, "params") and hasattr(jit_fn, "fn"):
                jit_fn = jit_fn.fn
            timings = {}
            for candidate in tuner.configs:
                trial = {**kwargs, **candidate.all_kwargs()}
                try:
                    timings[candidate] = _bench_config(
                        driver, jit_fn, grid, args, trial)
                except Exception as exc:
                    # A config the backend cannot run (a tile over the
                    # threadgroup-memory budget, say) is not a failure of the
                    # launch — it is one candidate out of the running. It
                    # becomes a failure only if none survives.
                    timings[candidate] = None
                    _record_refusal(tuner, candidate, exc)
            usable = {c: t for c, t in timings.items() if t is not None}
            if not usable:
                raise SpecializationError(
                    f"{tuner.base_fn.__name__}: every autotune config was "
                    f"refused by the backend at this shape. See "
                    f"launcher.refusals() for the reason of each.")
            cache[key] = min(usable, key=usable.get)
        config = cache[key]

    merged = {**kwargs, **config.all_kwargs()}
    return merged


#: Configs a backend refused during autotuning, kept so the report can say
#: which kernel lost which tile and why rather than only that it was slower.
_REFUSALS: list[dict] = []


def _record_refusal(tuner, config, exc) -> None:
    _REFUSALS.append({
        "kernel": getattr(tuner.base_fn, "__name__", str(tuner)),
        "config": str(config),
        "error": f"{type(exc).__name__}: {exc}",
    })


def refusals() -> list[dict]:
    """Every autotune config a backend refused, in order.

    The refused-kernel table of a run is read from here: a kernel whose tile
    the backend cannot take is a measured gap, not a footnote, and it is the
    first thing a port report should show.
    """
    return list(_REFUSALS)


def clear_refusals() -> None:
    """Start a fresh record. For a report that covers one run, not the process."""
    _REFUSALS.clear()


# --- the launch itself ------------------------------------------------------

class _BoundKernel:
    """What `kernel[grid]` becomes. Calling it launches."""

    __slots__ = ("_kernel", "_grid")

    def __init__(self, kernel, grid):
        self._kernel = kernel
        self._grid = grid

    def __call__(self, *args, **kwargs):
        driver = active_driver()
        if driver is None:
            # No driver: Triton's own path, unchanged. This is the CUDA
            # behaviour today and it must stay bit-identical until the machine
            # that owns CUDA activates a driver.
            return _triton_bound(self._kernel, self._grid)(*args, **kwargs)
        return launch(driver, self._kernel, self._grid, args, kwargs)


def _bind_arguments(jit_fn, args, kwargs) -> dict:
    """Every parameter of the kernel to its value, in declaration order.

    This is what Triton's generated binder produces, and what it hands to a
    grid callable — so `lambda meta: (cdiv(n, meta['BLOCK']),)` reads the
    same values here as it does there, constexprs included.
    """
    bound = {}
    positional = list(args)
    for index, param in enumerate(jit_fn.params):
        if param.name in kwargs:
            bound[param.name] = kwargs[param.name]
        elif index < len(positional):
            bound[param.name] = positional[index]
        elif getattr(param, "has_default", False):
            bound[param.name] = param.default
        else:
            raise SpecializationError(
                f"{jit_fn.__name__}: no value for parameter {param.name!r}")
    return bound


def _specialize(jit_fn, bound):
    """`(signature, constexprs, call_args, keys)` for one bound argument set.

    `keys` is the half of the specialization that is NOT the type: Triton's
    per-argument marker, "D" when a pointer or an integer is a multiple of
    16. It is not decoration. Triton turns it into a `tt.divisibility`
    attribute on the IR, and the middle end vectorizes loads and stores on
    the strength of it — so a launcher that computes the marker and then
    drops it compiles a DIFFERENT kernel from the one Triton compiles, at
    the same shape, with the same arguments.

    That is not a theory. Dropping it here changed rms_norm's fp16 result at
    two of four milestone shapes (2x4096 and 8x1536), deterministically, in
    the first light A/B on 2026-09-05: same inputs, same driver, different
    vectorization, different summation order. fp32 was unaffected, which is
    exactly why this would have been easy to miss.
    """
    signature, constexprs, call_args, keys = {}, {}, [], {}
    for param in jit_fn.params:
        value = bound[param.name]
        if param.is_constexpr:
            signature[param.name] = "constexpr"
            constexprs[param.name] = value
            keys[param.name] = ""       # Triton lists it, with no attribute
            continue
        type_name, key = specialize_argument(value)
        if type_name == "constexpr":
            # Triton folds an argument equal to 1 (and None) into a constant.
            # Doing the same is what keeps our compiled kernel the one it
            # would have compiled.
            signature[param.name] = "constexpr"
            constexprs[param.name] = value
            keys[param.name] = ""
            continue
        signature[param.name] = type_name
        if isinstance(key, str):
            keys[param.name] = key
        call_args.append(
            value.data_ptr() if hasattr(value, "data_ptr") else value)
    return signature, constexprs, call_args, keys


def launch(driver, kernel, grid, args, kwargs) -> None:
    """Honour the decorators, bind, specialize, compile, dispatch."""
    jit_fn, kwargs = resolve_wrappers(driver, kernel, grid, args, kwargs)

    num_warps = int(kwargs.get("num_warps", 4))
    num_stages = kwargs.get("num_stages")
    for option in _LAUNCH_OPTIONS:
        kwargs.pop(option, None)

    bound = _bind_arguments(jit_fn, args, kwargs)
    signature, constexprs, call_args, keys = _specialize(jit_fn, bound)

    compiled = driver.compile(jit_fn, signature, constexprs,
                              num_warps=num_warps, specialization=keys,
                              num_stages=num_stages)
    compiled.launch(_normalize_grid(grid, bound), call_args)


def _normalize_grid(grid, meta=None):
    """A grid callable is given the bound arguments, as Triton gives them."""
    if callable(grid):
        grid = grid(meta if meta is not None else {})
    if isinstance(grid, int):
        return (grid,)
    return tuple(grid)


def _unwrap_jit(kernel):
    """Peel down to the JITFunction, DISCARDING what the decorators decide.

    For inspection only — a compile census, a signature dump, the R33 proof.
    The launch path uses `resolve_wrappers`, which honours them instead.
    """
    from triton.runtime.jit import JITFunction

    seen = 0
    while not isinstance(kernel, JITFunction) and hasattr(kernel, "fn") \
            and seen < 8:
        kernel = kernel.fn
        seen += 1
    return kernel


# --- installation -----------------------------------------------------------

_ORIGINAL_GETITEM = None


def _triton_bound(kernel, grid):
    """Triton's own `kernel[grid]`, whatever we did to `__getitem__`."""
    if _ORIGINAL_GETITEM is None:                       # pragma: no cover
        return kernel[grid]
    return _ORIGINAL_GETITEM(kernel, grid)


def install() -> None:
    """Route every `kernel[grid](...)` in the process through this launcher.

    One seam rather than 64 edited call sites: the call sites keep reading
    `kernel[grid](...)`, which is what they mean, and what that expression
    resolves to is the engine's decision. It also means a launch site added
    tomorrow is covered without anyone remembering to route it.
    """
    global _ORIGINAL_GETITEM
    from triton.runtime.jit import KernelInterface

    if _ORIGINAL_GETITEM is not None:
        return
    _ORIGINAL_GETITEM = KernelInterface.__getitem__

    def __getitem__(self, grid):
        return _BoundKernel(self, grid)

    KernelInterface.__getitem__ = __getitem__


def uninstall() -> None:
    """Restore Triton's `__getitem__`. Tests use it; the engine does not."""
    global _ORIGINAL_GETITEM
    from triton.runtime.jit import KernelInterface

    if _ORIGINAL_GETITEM is None:
        return
    KernelInterface.__getitem__ = _ORIGINAL_GETITEM
    _ORIGINAL_GETITEM = None


def is_installed() -> bool:
    return _ORIGINAL_GETITEM is not None
