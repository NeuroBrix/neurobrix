"""What the backend in force can do, as declared or measured: float64, the flash and reduction tiles, pointers loaded
from memory, device-side asserts. Cut out of tensor.py (M2, 2026-10-29), unchanged."""

from __future__ import annotations

from typing import Dict, Optional

from nbxtensor.backend import runtime as _rt




# Whether the backend HONOURS a device-side assert the author explicitly kept
# with `@triton.jit(debug=True)`. This is not a question about debug builds:
# `tl.device_assert` reaches the IR at all ONLY when `options.debug` is true
# (triton/language/semantic.py: `if not self.builder.options.debug: return`),
# so a `tt.assert` in the IR is by construction one the author demanded.
#
# CUDA and ROCm halt the thread at the assert and the next sync raises. The
# Metal backend COMPUTES the predicate and then discards it — measured
# 2026-09-10: with debug=True the generated MSL carries `mask_5 = idx < 2`
# and never reads it. The guard is paid for in registers and not delivered,
# so a kernel whose only bounds check is that assert performs the very access
# the assert was there to prevent (`index_put` wrote 8 floats past the end of
# a 24-float tensor; `embedding` read past the end of the weight).
#
# This is a capability, not a vendor test — adding a backend is adding a row.
_BACKEND_TRAPS_ON_DEVICE_ASSERT = {"cuda": True, "hip": True, "metal": False}

# Whether a kernel can load a device ADDRESS out of a tensor and then read
# through it — the [E] int64 pointer table that lets one launch reach every
# expert's weights instead of one launch per expert.
#
# Measured 2026-09-17 on triton-ext (M4 Pro, Toolchain 27) with a control that
# differs by the one variable — whether the address arrives AS a pointer or as
# an integer to be converted — and in BOTH Triton spellings, because they are
# different constructs and cannot be assumed to share a fate:
#
#     control  (pointer as a pointer)       : nonzero 256/256   correct
#     subject  (.to(pointer_type))          : nonzero   0/256   ALL ZEROS
#     subject  (tl.cast bitcast=True)       : nonzero   0/256   ALL ZEROS
#
# The second subject is the spelling our own kernels use, which is what makes
# the engine's zeros attributable to this and not to us. Nothing is raised: the
# kernel compiles, launches and writes zeros. The reproducer contains none of
# our code (`repro_ext_pointer_from_table.py`) and is drafted for upstream.
#
# It is recorded HERE because the engine cannot catch what is never raised: a
# band that cannot address its experts has to refuse by name, and a refusal
# needs a fact to stand on. This is a capability, not a vendor test — adding a
# backend is adding a row.
_BACKEND_LOADS_POINTERS_FROM_MEMORY = {"cuda": True, "hip": True, "metal": False}
# Whether the backend's kernels can read and write a float64 (or complex128) element. Metal
# has no double: its lowering NARROWS f64 to float in silence (`triton/metal_backend.py`,
# `TRITON_METAL_FP64_UNAVAILABLE`), so a kernel over a float64 buffer reads eight-byte
# elements as four-byte ones. Measured 2026-09-28: `from_numpy(float64).to(bf16)` on the
# M4 Pro gave 1 139 of 143 360 bits right and 3 255 non-finite values (the Sana 4K sincos
# positional embedding, every off-trace size). A census says "not this time"; the door in
# `NBXTensor.to` says "never": a cast from or to such a dtype is refused BY NAME there.
#
# The answer is the CALLER's to give (`set_backend_has_fp64`), not a table here: it is a fact
# about the hardware the caller declares once (NeuroBrix: the vendor profile's
# `precision.kernels_carry_fp64.triton`, bound by `kernels.wrappers.set_hardware_profile`).
# Unset, a question that needs it is refused — a float64 element is never assumed readable.
_BACKEND_HAS_FP64 = None


def set_backend_has_fp64(value: bool) -> None:
    """Declare whether this process's backend kernels compute float64 / complex128 elements."""
    global _BACKEND_HAS_FP64
    if not isinstance(value, bool):
        raise TypeError(f"set_backend_has_fp64 takes a bool, got {value!r}")
    _BACKEND_HAS_FP64 = value


def backend_has_fp64() -> bool:
    """Whether the backend's kernels handle float64 elements, as declared by
    `set_backend_has_fp64`; refused while undeclared."""
    if _BACKEND_HAS_FP64 is None:
        raise RuntimeError(
            f"ZERO FALLBACK: whether the '{_rt._detect_gpu_backend()}' backend's kernels compute "
            f"float64 was never declared (`set_backend_has_fp64`); it is the hardware's fact, "
            f"never assumed")
    return _BACKEND_HAS_FP64


# The smallest `tl.dot` tile dimension a backend's attention lowering handles
# CORRECTLY. On cuda/hip the floor is the TensorCore minimum, 16, and a decode
# step at seqlen_q=1 uses it. Metal's generic attention lowering silently
# mis-computes below 32 -- rows past the first become garbage, for any
# head_dim -- and refuses rather than emit it; its tiled template does take a
# smaller tile, but only for a kernel its detector resolves completely at
# head_dim 64, which is not every kernel.
#
# Sixteen wastes half a Q tile at seqlen_q=1. A refusal wastes the model.
#
# This is a capability, not a vendor test — adding a backend is adding a row.
_BACKEND_FA_MIN_TILE = {"cuda": 16, "hip": 16, "metal": 32}


def fa_min_tile() -> int:
    """The smallest attention tile dimension this backend computes correctly."""
    return _backend_capability(
        _BACKEND_FA_MIN_TILE, "_BACKEND_FA_MIN_TILE",
        "the smallest attention tile dimension it computes correctly")


# The largest 2-D reduction tile (product of block dims) a backend's lowering
# stages correctly. The generic Metal lowering stages one element per thread,
# so a tile above the 1024-thread threadgroup has lanes with nothing to hold
# their part -- it refuses rather than reduce part of the tile. cuda/hip lower
# a 2-D reduce differently and take the tile the reduction kernels were tuned
# for (BLOCK_M*BLOCK_N = 8192); their entry is that tuned size so min() is the
# identity and nothing moves there.
#
# A capability, not a vendor test -- adding a backend is adding a row.
_BACKEND_REDUCE_TILE_MAX = {"cuda": 1 << 20, "hip": 1 << 20, "metal": 1024}


def reduce_tile_max() -> int:
    """The largest 2-D reduction tile this backend stages correctly."""
    return _backend_capability(
        _BACKEND_REDUCE_TILE_MAX, "_BACKEND_REDUCE_TILE_MAX",
        "the largest 2-D reduction tile it stages correctly")


# Whether a backend's lowering stages a 2-D reduce ONE ELEMENT PER THREAD along the reduced extent: then that extent of
# a tile cannot exceed the threads of the program that reduces it (its warps x the backend's threads per warp). Metal,
# measured 2026-09-07: the GEMV's 8 x 256 at 4 warps and the mv's 64 x 256 at 4 warps refused, naming the numbers
# (tests/unit/config/test_apple_profiles_agree_with_the_device.py held those tiles in the profile until the certifier
# chose them; the rule is the kernels' prune now, `ops/_configs.prune_gemv_vec` / `prune_addmv`). cuda/hip reduce many
# elements per thread in registers.
_BACKEND_REDUCE_PER_THREAD = {"cuda": False, "hip": False, "metal": True}
_BACKEND_THREADS_PER_WARP = {"cuda": 32, "hip": 64, "metal": 32}


def reduced_extent_max(num_warps: int) -> Optional[int]:
    """The largest reduced extent a program of `num_warps` stages on this backend, or None when its lowering bounds
    none (it reduces many elements per thread)."""
    if not _backend_capability(_BACKEND_REDUCE_PER_THREAD, "_BACKEND_REDUCE_PER_THREAD",
                               "whether it stages a 2-D reduce one element per thread"):
        return None
    return int(num_warps) * int(_backend_capability(_BACKEND_THREADS_PER_WARP, "_BACKEND_THREADS_PER_WARP",
                                                    "its threads per warp"))


def _backend_capability(table, name: str, what: str) -> bool:
    """One row of one capability table, or a refusal naming both."""
    backend = _rt._detect_gpu_backend()
    value = table.get(backend)
    if value is None:
        raise RuntimeError(
            f"ZERO FALLBACK: backend {backend!r} does not say {what} "
            f"({name}, rows: {sorted(table)}). Adding a backend is adding a "
            f"row, not guessing.")
    return value


#: What the ENGINE declared about a backend, above the tables: a capability that is a
#: property of the launcher it selected, not of the silicon, is the engine's to state.
#: nbx_tensor is a library with a boundary (R33, the ratchet cell
#: test_the_boundary_does_not_widen) and asks the engine nothing; the engine tells it,
#: through `declare_backend_capability`, before a kernel is launched.
_DECLARED_CAPABILITIES: Dict[str, Dict[str, bool]] = {}


def declare_backend_capability(backend: str, **capabilities: bool) -> None:
    """The engine states a capability of the launcher it selected for `backend`
    (`loads_pointers_from_memory=True` for Metal's triton_ext, proven through its pinned
    scope). Read by the capability questions below, above the static tables."""
    for name, value in capabilities.items():
        if not isinstance(value, bool):
            raise TypeError(f"a backend capability is a bool, not {value!r} ({backend}.{name})")
    _DECLARED_CAPABILITIES.setdefault(backend, {}).update(capabilities)


def backend_loads_pointers_from_memory() -> bool:
    """True where a kernel may dereference an address it LOADED from a tensor.

    On Metal the answer is a property of the LAUNCHER, not the silicon: a
    loaded address reads correctly exactly when the driver keeps the captured
    address alive across launches, which `triton_ext` does through
    `pinned_addresses` (proven against an fp64 oracle,
    test_the_moe_table_reads_through_a_pinned_scope, 2026-09-20; the archived
    fork had no such scope and the same load returned zeros with nothing
    raised — the measured defect this gate refuses). So the metal row is a
    SELECTION read, not a constant — and the selection is the engine's, so the
    engine DECLARES it here on every Metal run's entry
    (`triton/metal_backend.ensure_triton_metal_or_raise`); undeclared, the table's
    False stands until a launcher proves its own lifetime contract. Until
    2026-09-26 this function imported the engine's metal_backend to ask, which
    widened the library's boundary (the ratchet cell went red on main).
    """
    backend = _rt._detect_gpu_backend()
    declared = _DECLARED_CAPABILITIES.get(backend, {}).get("loads_pointers_from_memory")
    if declared is not None:
        return declared
    return _backend_capability(
        _BACKEND_LOADS_POINTERS_FROM_MEMORY, "_BACKEND_LOADS_POINTERS_FROM_MEMORY",
        "whether a kernel can read through a pointer loaded from memory")


def backend_traps_on_device_assert() -> bool:
    """True where `tl.device_assert` actually stops the kernel."""
    return _backend_capability(
        _BACKEND_TRAPS_ON_DEVICE_ASSERT, "_BACKEND_TRAPS_ON_DEVICE_ASSERT",
        "whether it honours a device-side assert")
