"""NBXTensor — Tensor type for --triton mode.

Wraps raw CUDA device pointers with shape/strides/dtype metadata.
All metadata ops are pure Python CPU computation.
Triton kernels receive NBXTensor via .data_ptr() and .dtype.

Dependencies: ctypes (stdlib), math (stdlib), triton (lazy, for dtype mapping).

Weight loading: safetensors → numpy → cudaMemcpy H2D → NBXTensor
Allocation: raw cudaMalloc via ctypes
Metadata ops: pure Python stride/shape computation
"""

from __future__ import annotations

import bisect          # module level, NOT inside the range helpers: they run on the
                      # finalizer path, and at interpreter shutdown `sys.meta_path` is
                      # None, so a function-level import raises ImportError AFTER the
                      # cudaFree has already happened — aborting the bookkeeping that
                      # follows it and printing a traceback that can mask a real
                      # [NBX-CUDA-ERROR] from the same free (observed 2026-09-18).
import ctypes
import functools
import math
import os
import struct
import sys
import threading
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .shape import (MAX_NDIM, broadcast_strides, contiguous_strides, is_contiguous, max_linear_offset,
                    slice_bounds)
from nbxtensor.dtype.nbx_dtype import (NBXDtype, _COMPLEX_DTYPES, _DTYPE_FROM_STR, _DTYPE_SIZES,  # noqa: F401
                                      _DTYPE_TYPESTR, _FLOATING_DTYPES, _NUMPY_DTYPE_MAP, dtype_size, parse_dtype)
from nbxtensor.memory.allocator import DeviceAllocator, DeviceOOMError, ImpossibleExtentError  # noqa: F401
from nbxtensor.memory import allocator as _alloc
from nbxtensor.backend import runtime as _rt, capabilities as _caps, faults as _faults  # noqa: F401


# NBX_TRACE_DEL_BIG_MB=N (default off): NBXTensor.__del__ logs every
# free of an owns_data tensor >= N MB. Read ONCE at import
# (P-KERNEL-LAUNCH-HYGIENE C4) — __del__ fires per tensor, and an
# environ.get there is pure overhead on the free path. Consequence:
# changing the env var mid-process no longer applies; acceptable for a
# diagnostic that is set before launch. None == trace disabled.
try:
    _TRACE_DEL_BIG_ENV = os.environ.get("NBX_TRACE_DEL_BIG_MB", "0")
    _TRACE_DEL_BIG_THRESH: Optional[int] = (
        int(_TRACE_DEL_BIG_ENV) * 1024 * 1024
        if _TRACE_DEL_BIG_ENV != "0" else None)
except Exception:
    _TRACE_DEL_BIG_THRESH = None


# ============================================================================
# TRITON DTYPE MAPPING (lazy import)
# ============================================================================

_TL_DTYPE_MAP = None


def _get_tl_dtype(nbx_dtype: NBXDtype):
    """Map NBXDtype → triton.language dtype. Lazy import of triton."""
    global _TL_DTYPE_MAP
    if _TL_DTYPE_MAP is None:
        import triton.language as tl
        _TL_DTYPE_MAP = {
            NBXDtype.float16: tl.float16,
            NBXDtype.bfloat16: tl.bfloat16,
            NBXDtype.float32: tl.float32,
            NBXDtype.float64: tl.float64,
            NBXDtype.int8: tl.int8,
            NBXDtype.int16: tl.int16,
            NBXDtype.int32: tl.int32,
            NBXDtype.int64: tl.int64,
            NBXDtype.uint8: tl.uint8,
            NBXDtype.bool_: tl.uint8,  # bool stored as uint8 (same as PyTorch)
        }
    result = _TL_DTYPE_MAP.get(nbx_dtype)
    if result is None:
        import triton.language as tl
        return tl.float32
    return result


# ============================================================================
# STRIDED COPY
# ============================================================================

# Shape/stride metadata buffers, keyed by (values, device). Never
# evicted — a frozen replay plan records raw device pointers inside its
# recorded kernel launches, so a freed buffer would leave the plan
# pointing at reclaimed memory. See _upload_int64_array.
_INT64_ARRAY_CACHE: dict = {}


def _saturating_cast(src_dtype, dst_dtype) -> bool:
    """The protected float16 downcast (clamp finite values to ±65504) —
    the triton mirror of the compiled DtypeEngine's conversion contract,
    decided once here for every cast path (`.to`, `copy_`, `__setitem__`)."""
    return (dst_dtype == NBXDtype.float16
            and src_dtype in (NBXDtype.float32, NBXDtype.float64, NBXDtype.bfloat16))


def _copy_nd(src: 'NBXTensor', dst: 'NBXTensor'):
    """Copy `src` into `dst` — any strides on either side, `src` broadcast
    to `dst`'s shape, the dtype converted on the store — in ONE launch.

    Real dtypes only: a complex source keeps the contiguous-then-cast
    path (the kernel sees a float pointer, and a complex cast is a pair
    of floats per element that `copy_kernel` already handles).
    """
    import triton
    from nbxtensor.ops.strided_copy import strided_copy_nd_kernel

    if tuple(src._shape) != tuple(dst._shape):
        src = src.expand(*dst._shape)
    ndim = dst.ndim
    if ndim > MAX_NDIM:
        raise RuntimeError(
            f"NBXTensor._copy_nd: ndim={ndim} exceeds the PyTorch hard limit "
            f"of {MAX_NDIM}. Please open an issue with the model name and trace site.")
    n = dst._numel
    if n == 0:
        return
    if ndim == 0:
        # a scalar: one element, both offsets 0 — a rank-1 walk of extent 1
        shape, sst, dst_st = (1,), (0,), (0,)
        ndim = 1
    else:
        shape, sst, dst_st = tuple(dst._shape), tuple(src._strides), tuple(dst._strides)
    shape_buf = _upload_int64_array(shape, dst._device_idx)
    src_stride_buf = _upload_int64_array(sst, dst._device_idx)
    dst_stride_buf = _upload_int64_array(dst_st, dst._device_idx)
    _m = _ew_meta(n)                 # the certified tile of the count's class (ops/elementwise_launch.py)
    BLOCK = _m['BLOCK_SIZE']
    _rt._set_device(dst)
    strided_copy_nd_kernel[(triton.cdiv(n, BLOCK),)](
        src, dst, n, shape_buf, src_stride_buf, dst_stride_buf,
        BLOCK_SIZE=BLOCK, num_warps=_m['num_warps'], NDIM=ndim,
        SATURATE_F16=_saturating_cast(src._dtype, dst._dtype),
        WIDE=_index_wide(n, max_linear_offset(shape, sst), max_linear_offset(shape, dst_st)))


def _ew_meta(n: int) -> dict:
    """An element-wise launch's certified tile and warps (`ops/elementwise_launch.ew_meta`), imported at use: that
    module resolves through an autotuner that allocates its scratch here."""
    from nbxtensor.ops.elementwise_launch import ew_meta
    return ew_meta(n)


def _upload_int64_array(values, device_idx: int) -> 'NBXTensor':
    """Upload a small tuple/list of ints to the GPU as a contiguous
    int64 NBXTensor. Used to hand shape and stride vectors to the N-D
    strided_copy / strided_scatter kernels.

    Zero-torch: goes through `NBXTensor.from_numpy`, which internally
    uses `DeviceAllocator.malloc_cuda` + `DeviceAllocator.memcpy` (via
    `_GPU_BACKENDS`).

    MEMOISED by value. The original form allocated, uploaded and freed a
    fresh buffer per call, on the reasoning that "200 bytes per upload is
    amortised across a whole forward pass (< 1 ms total)". Measured on
    2026-08-20 that reasoning does not hold: `strided_copy` runs ~757
    times per decode token on the canonical 30B row, so these metadata
    uploads were **616 host-to-device transfers per token** — costing
    more than every matmul in the model combined. The assumption was
    true when the strided path was rare and false once it became hot.

    The values are shapes and strides: a model touches a few hundred
    distinct tuples, so the cache is bounded in practice at tens of
    kilobytes.

    It is deliberately NEVER evicted, and that is a correctness
    requirement rather than a simplification: a frozen replay plan
    records raw device POINTERS inside its kernel launches, so evicting
    a buffer between recording and replay would leave the plan pointing
    at freed memory.

    Safe to share because every consumer is read-only — the four call
    sites are `_strided_copy` and `_strided_scatter`, and the kernels
    only `tl.load` these pointers, never `tl.store` to them.
    """
    key = (tuple(int(v) for v in values), int(device_idx))
    hit = _INT64_ARRAY_CACHE.get(key)
    if hit is not None:
        return hit

    import numpy as _np
    arr = _np.asarray(values, dtype=_np.int64)
    DeviceAllocator.set_device(device_idx)
    buf = NBXTensor.from_numpy(arr)
    _INT64_ARRAY_CACHE[key] = buf
    return buf


def _upload_index_array(values, device_idx: int, max_offset: int) -> 'NBXTensor':
    """The same, in the NARROWEST integer that holds every index the kernel
    will compute from it.

    The element type of this buffer decides the width of the whole index
    computation downstream: `tl.load(shape_ptr + dim)` takes its dtype from
    the pointer, and Triton then promotes the flat offset, every `%`, every
    `//` and every stride multiply to match. At int64 that is a 64-bit
    integer division and modulo per dimension per element, on hardware —
    Apple's — that has no integer-division instruction at any width and
    emulates both.

    Measured 2026-09-08 on an M4 Pro, `strided_copy_kernel` over a
    (2048, 5632) bf16 transpose, the same kernel and the same launch, only
    the metadata's width changed:

        int64 metadata   3.794 ms    12.2 GB/s
        int32 metadata   1.411 ms    32.7 GB/s
        a contiguous copy of the same bytes    0.384 ms   120.3 GB/s

    `max_offset` is the largest linear offset the kernel can form from these
    values — `sum((shape[i] - 1) * stride[i])` at the call site. int32 is
    used only when that bound and every value fit in it, so the narrower
    buffer computes the SAME offsets rather than nearly the same ones; a
    tensor large enough to overflow keeps int64 and keeps today's behaviour.
    Numerically inert on CUDA by the same argument, where it is also the
    cheaper arithmetic.
    """
    key = (tuple(int(v) for v in values), int(device_idx), "narrow")
    hit = _INT64_ARRAY_CACHE.get(key)
    if hit is not None:
        return hit

    import numpy as _np
    _I32_MAX = 2 ** 31 - 1
    fits = (abs(int(max_offset)) <= _I32_MAX
            and all(abs(int(v)) <= _I32_MAX for v in values))
    arr = _np.asarray(values, dtype=_np.int32 if fits else _np.int64)
    DeviceAllocator.set_device(device_idx)
    buf = NBXTensor.from_numpy(arr)
    _INT64_ARRAY_CACHE[key] = buf
    return buf


def _index_wide(n_elements: int, *max_offsets: int) -> bool:
    """64-bit index arithmetic in the strided kernels (`WIDE`) when the element count or an operand's largest offset
    is within a factor of two of 2^31 (a masked lane forms an offset past the extent before its mask drops it; the
    factor covers it). Below it the 32-bit path computes the same offsets. The kernels' flat offset was made 64-bit
    from the program id for tensors past 2^31 elements, and that promoted every `%` and `//` of every copy back to
    64-bit — emulated on Apple's GPU: the (1,128,9,242,362) bf16 permute copy of a VAE conv3d ran 84 ms (4.8 GB/s)
    against 8.3 ms (48 GB/s) at 32 bits, same bytes (Mac, 2026-10-21); the narrow metadata of `_upload_index_array`
    had stopped mattering."""
    return max([int(n_elements)] + [abs(int(b)) for b in max_offsets]) >= (1 << 30)




def _strided_copy(src: 'NBXTensor', dst: 'NBXTensor'):
    """Copy non-contiguous src into contiguous dst via the N-D scalable
    Triton kernel.

    The kernel is parameterised by `NDIM: tl.constexpr`; Triton's
    kernel cache stores one specialised compilation per distinct ndim
    encountered at runtime (first call compiles, subsequent calls hit
    the cache at near-zero overhead). Shape and strides travel via
    small int64 scratch buffers on-device, loaded inside the kernel
    body — this keeps the kernel text identical regardless of ndim.

    History: the prior 5-D-hardcoded kernel silently dropped dims
    beyond 5 (via `[1] * (5 - ndim)` returning `[]` when `ndim > 5`),
    producing OOB reads. 6-D PixArt patchify clones corrupted
    (Alpha) or crashed (Sigma) on this path; 8-D SANA-Video VAE
    clones never reached production. N-D path covers all of them
    without a hardcoded ceiling.
    """
    import triton
    from nbxtensor.ops.strided_copy import strided_copy_kernel

    # Complex tensors are stored interleaved [real, imag], but Triton sees the
    # data_ptr as a plain float (no complex dtype). The strided kernel would copy
    # ONE float per element with the per-(complex)-element strides applied as
    # float offsets — corrupting the imaginary half (chatterbox s3gen stft does
    # fft_r2c -> transpose -> contiguous; the transposed-complex contiguous
    # mangled the imag spectrum). Reinterpret both as a float view with a
    # trailing [2] dim (strides * 2 in elements, +1 for the re/imag axis) and
    # recurse — this copies BOTH floats per complex element with correct strides.
    # Built manually (NOT view_as_real, which calls .contiguous() on strided
    # input → would recurse back into this same broken path).
    if src.is_complex():
        _fdt = (NBXDtype.float64 if src._dtype == NBXDtype.complex128
                else NBXDtype.float32)
        _fsrc = NBXTensor(
            src._data_ptr, tuple(src._shape) + (2,),
            tuple(s * 2 for s in src._strides) + (1,),
            _fdt, src._device, src._offset * 2,
            device_idx=src._device_idx,
            base=src._base if src._base is not None else src, pinned=src._pinned)
        _fdst_shape = tuple(dst._shape) + (2,)
        _fdst = NBXTensor(
            dst._data_ptr, _fdst_shape, contiguous_strides(_fdst_shape),
            _fdt, dst._device, dst._offset * 2,
            device_idx=dst._device_idx,
            base=dst._base if dst._base is not None else dst, pinned=dst._pinned)
        _strided_copy(_fsrc, _fdst)
        return

    ndim = src.ndim
    if ndim > MAX_NDIM:
        raise RuntimeError(
            f"NBXTensor._strided_copy: ndim={ndim} exceeds the PyTorch "
            f"hard limit of {MAX_NDIM}. Please open an issue with the "
            f"model name and trace site."
        )
    n = src._numel

    _bound = max_linear_offset(src._shape, src._strides)
    shape_buf = _upload_index_array(src._shape, src._device_idx, _bound)
    stride_buf = _upload_index_array(src._strides, src._device_idx, _bound)

    _m = _ew_meta(n)                 # the certified tile of the count's class (ops/elementwise_launch.py)
    BLOCK = _m['BLOCK_SIZE']
    grid = (triton.cdiv(n, BLOCK),)
    _rt._set_device(src)
    strided_copy_kernel[grid](
        src, dst, n,
        shape_buf, stride_buf,
        BLOCK_SIZE=BLOCK, num_warps=_m['num_warps'],
        NDIM=ndim,
        WIDE=_index_wide(n, _bound),
    )


def _strided_scatter(src: 'NBXTensor', dst: 'NBXTensor'):
    """Scatter contiguous src into non-contiguous dst via the N-D
    scalable Triton kernel. Inverse of `_strided_copy`.

    Used by `NBXTensor.__setitem__` for KV-cache indexed writes where
    the destination is a narrow view with non-contiguous strides. KV
    caches are typically 4-D today so this path rarely exercises the
    N-D machinery, but it stays in sync with `_strided_copy` for
    consistency and to future-proof any emerging cache layouts.
    """
    import triton
    from nbxtensor.ops.strided_copy import strided_scatter_kernel

    ndim = dst.ndim
    if ndim > MAX_NDIM:
        raise RuntimeError(
            f"NBXTensor._strided_scatter: ndim={ndim} exceeds the "
            f"PyTorch hard limit of {MAX_NDIM}. Please open an issue."
        )
    n = src._numel

    _bound = max_linear_offset(dst._shape, dst._strides)
    shape_buf = _upload_index_array(dst._shape, dst._device_idx, _bound)
    stride_buf = _upload_index_array(dst._strides, dst._device_idx, _bound)

    _m = _ew_meta(n)                 # the certified tile of the count's class (ops/elementwise_launch.py)
    BLOCK = _m['BLOCK_SIZE']
    grid = (triton.cdiv(n, BLOCK),)
    _rt._set_device(src)
    strided_scatter_kernel[grid](
        src, dst, n,
        shape_buf, stride_buf,
        BLOCK_SIZE=BLOCK, num_warps=_m['num_warps'],
        NDIM=ndim,
        WIDE=_index_wide(n, _bound),
    )


def _fill_constant(t: 'NBXTensor', value) -> 'NBXTensor':
    """Fill a contiguous NBXTensor's storage with a scalar constant via the
    pure-Triton `full_kernel`. Backs `NBXTensor.ones` and the non-zero branch
    of `fill_` — the zero case stays on the cheaper byte-`memset` path. Flat
    1-D indexing assumes contiguous storage (the only callers are freshly
    allocated tensors and the existing full-storage fill_ semantics).
    """
    if t._nbytes == 0:
        return t
    import triton
    from nbxtensor.ops.fill_op import fill_kernel

    n = t._numel
    _m = _ew_meta(n)                 # the certified tile of the count's class (ops/elementwise_launch.py)
    BLOCK = _m['BLOCK_SIZE']
    if not t.is_contiguous():
        # Flat 1-D indexing assumes contiguous storage; on a strided view it
        # would write the wrong addresses. Fill a contiguous temp, then scatter
        # into the view's strided positions.
        tmp = NBXTensor.empty(t._shape, t._dtype, f"cuda:{t._device_idx}")
        _rt._set_device(tmp)
        fill_kernel[(triton.cdiv(n, BLOCK),)](tmp, value, n, BLOCK_SIZE=BLOCK, num_warps=_m['num_warps'])
        _strided_scatter(tmp, t)
        return t
    grid = (triton.cdiv(n, BLOCK),)
    _rt._set_device(t)
    fill_kernel[grid](t, value, n, BLOCK_SIZE=BLOCK, num_warps=_m['num_warps'])
    return t


# ============================================================================
# STRIDE COMPUTATION
# ============================================================================

# ============================================================================
# NBXTensor
# ============================================================================

class NBXTensor:
    """Tensor descriptor for --triton mode.

    Properties for Triton kernel compatibility:
      .data_ptr() → raw CUDA pointer (int) — Triton reads this
      .dtype      → triton.language dtype (tl.float16 etc.) — Triton JIT uses this
      .nbx_dtype  → NBXDtype enum — internal, zero dependency
      .shape      → tuple of ints
      .stride()   → tuple of ints (element strides)

    All metadata ops (view, reshape, permute...) are pure Python.
    Allocation via raw cudaMalloc.
    """

    __slots__ = ('_data_ptr', '_shape', '_strides', '_dtype', '_device',
                 '_nbytes', '_numel', '_offset', '_owns_data', '_device_idx',
                 '_base',     # reference to parent tensor (prevents GC of underlying memory);
                              # also holds the numpy buffer for unpinned CPU tensors
                 '_pinned',   # True for pinned host memory (cudaMallocHost); used by __del__
                              # to dispatch to free_host_pinned instead of free_cuda
                 '_elem_size',  # C6b: dtype_size(_dtype) cached at construction
                 '_contig',     # C6a: contiguity flag cached at construction
                 )

    def __init__(self, data_ptr: int, shape: Tuple[int, ...],
                 strides: Tuple[int, ...], dtype: NBXDtype,
                 device: str = 'cuda', offset: int = 0, owns_data: bool = False,
                 device_idx: int = 0, base=None, pinned: bool = False):
        self._data_ptr = data_ptr
        self._shape = tuple(shape)
        self._strides = tuple(strides)
        self._dtype = dtype
        self._device = device
        self._offset = offset
        self._owns_data = owns_data
        self._device_idx = device_idx
        self._base = base  # keep parent alive to prevent use-after-free on views
        self._pinned = pinned
        self._numel = math.prod(shape) if shape else 1
        # C6b: element size cached once — shape/strides/dtype are fixed at
        # construction (every view/reshape builds a NEW NBXTensor; __init__
        # is the single construction path). `_elem_size` and `_nbytes` are
        # derived from `dtype` HERE and never again, so assigning `_dtype`
        # after construction leaves a tensor whose declared length belongs
        # to the old dtype — a buffer overrun when the new dtype is smaller.
        # There is no such retag left in the tree: the bf16-bits site
        # (graph_executor._load_constant_triton) declares the dtype to
        # from_numpy instead.
        self._elem_size = dtype_size(dtype)
        self._nbytes = self._numel * self._elem_size
        # C6a: contiguity is a pure function of the (immutable) shape and
        # strides — compute the flag once instead of per is_contiguous()
        # call in the hot loops.
        self._contig = is_contiguous(self._shape, self._strides)

    def __del__(self):
        # Targeted lifecycle trace: NBX_TRACE_DEL_BIG_MB=N (default off)
        # logs every __del__ on owns=True NBXTensors >= N MB. Lets us
        # tell whether a leaked NBXTensor is actually freed and when.
        # Gate is the module-level constant read once at import (C4) —
        # zero env lookup on the per-tensor free path. The try/except
        # keeps the free below unconditional (same as before), including
        # at interpreter shutdown when module globals may be gone.
        try:
            if (_TRACE_DEL_BIG_THRESH is not None and self._owns_data
                    and self._data_ptr
                    and self._nbytes >= _TRACE_DEL_BIG_THRESH):
                print(f"[NBX_DEL data_ptr={self._data_ptr} "
                      f"shape={self._shape} nbytes={self._nbytes/1024/1024:.0f}MB]",
                      flush=True)
        except Exception:
            pass

        # A tensor built by `NBXTensor.__new__` without `__init__` has NO
        # slot set at all — `copy`, `pickle`, the routing-contract test in
        # test_prefill_determinism, and any `__init__` that raises part-way
        # all leave one. Reading `_owns_data` there raises AttributeError, and
        # Python does not propagate an exception out of `__del__`: it prints an
        # unraisable exception and ABANDONS THE REST OF THIS METHOD — which is
        # the free below. So the one case the old line could not survive is the
        # one where failing silently would leak a device buffer.
        #
        # Returning early is correct and not a papered-over free: `__init__` is
        # pure assignment and allocates nothing, so a tensor that did not finish
        # it cannot own device memory. Ownership is recorded FROM the
        # `owns_data` argument, never established inside the tensor.
        try:
            owns, ptr = self._owns_data, self._data_ptr
        except AttributeError:
            return

        if owns and ptr:
            # Interpreter teardown clears the allocator's module before this one (it was imported while this module
            # was importing, and modules are cleared in reverse order of import): its globals are None, a free would
            # raise inside it. The process is exiting and hands every device buffer back to the driver as it does.
            if getattr(_alloc, "DeviceAllocator", None) is None:
                return
            if self._device == 'cuda':
                DeviceAllocator.free_cuda(ptr)
            elif self._device == 'cpu' and self._pinned:
                # Pinned host memory allocated via cudaMallocHost.
                DeviceAllocator.free_host_pinned(ptr)
            # Unpinned CPU: backed by self._base (numpy array) — Python
            # GC drops it automatically when self goes out of scope.

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    def data_ptr(self) -> int:
        """Raw CUDA pointer. Triton kernels read this."""
        return self._data_ptr + self._offset * self._elem_size

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self):
        """Returns triton.language dtype for Triton JIT compatibility."""
        return _get_tl_dtype(self._dtype)

    @property
    def nbx_dtype(self) -> NBXDtype:
        """Internal dtype — zero dependency."""
        return self._dtype

    @property
    def device(self):
        """Device descriptor. Returns self (NBXTensor is its own device context)."""
        return self

    @property
    def type(self) -> str:
        """Device type string for device.type compatibility."""
        return self._device

    @property
    def index(self) -> int:
        """Device index for device.index compatibility."""
        return 0  # default GPU 0

    @property
    def raw_device(self) -> str:
        return self._device

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        """Element count. Method (not property) for ATen API compat."""
        return self._numel

    def nbytes(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return dtype_size(self._dtype)

    def is_contiguous(self) -> bool:
        """Method for ATen API compat. Returns the flag computed at
        construction (C6a): _shape/_strides are never mutated after
        __init__ — every view/reshape/slice constructs a new NBXTensor."""
        return self._contig

    def is_expanded(self) -> bool:
        """True when the view replays elements along a broadcast axis.

        Expand views have one or more (shape > 1, stride == 0) axes, which
        means `numel * element_size > actual_backing_bytes`. Any caller that
        intends to `memcpy(nbytes)` across a device boundary MUST first
        materialise via `contiguous()` — otherwise the H2D / D2D read runs
        past the real allocation and copies garbage.
        """
        return any(st == 0 and sh > 1
                   for sh, st in zip(self._shape, self._strides))

    def spans_densely(self) -> bool:
        """True when the view's elements tile exactly one dense span of
        `nbytes` starting at data_ptr() — contiguous, or a permutation of a
        contiguous layout (a transpose). False for an expand view (stride 0
        on a broadcast axis) and for a narrow on an inner axis (gaps between
        rows): a single memcpy of nbytes copies the wrong bytes from those.
        The one definition: `triton.device_transfer.transfer_tensor` asks it too."""
        if self._numel == 0:
            return True
        expected = 1
        for sh, st in sorted(((sh, st) for sh, st in zip(self._shape, self._strides)
                              if sh != 1), key=lambda p: p[1]):
            if st != expected:
                return False
            expected *= sh
        return True

    @property
    def is_cuda(self) -> bool:
        return self._device == 'cuda'

    @property
    def is_cpu(self) -> bool:
        """True for NBXTensor allocated on host memory (zero3 offload,
        pinned or unpinned). Every NBXTensor is either CUDA or CPU
        today; there is no third state."""
        return self._device == 'cpu'

    @property
    def is_pinned(self) -> bool:
        """True for CPU-backed tensors allocated via cudaMallocHost
        (fast non_blocking H2D DMA). False otherwise, including all
        CUDA tensors."""
        return self._pinned

    # ========================================================================
    # TYPE CHECKS
    # ========================================================================

    def is_floating_point(self) -> bool:
        return self._dtype in _FLOATING_DTYPES

    def is_complex(self) -> bool:
        return self._dtype in _COMPLEX_DTYPES

    # ========================================================================
    # COMPLEX SUPPORT — complex64/128 stored as interleaved [real, imag] pairs
    # (numpy '<c8'/'<c16', torch convention). Reinterprets are zero-copy when
    # contiguous: the complex storage IS the float [..,N,2] storage. _offset is
    # in elements of _dtype, so the complex↔float offset converts ×2 / ÷2.
    # ========================================================================

    def view_as_real(self) -> 'NBXTensor':
        """complex [..,N] → float [..,N,2] (interleaved real/imag)."""
        if not self.is_complex():
            return self
        x = self if self.is_contiguous() else self.contiguous()
        fdt = NBXDtype.float64 if x._dtype == NBXDtype.complex128 else NBXDtype.float32
        new_shape = x._shape + (2,)
        return NBXTensor(x._data_ptr, new_shape, contiguous_strides(new_shape),
                         fdt, x._device, x._offset * 2,
                         device_idx=x._device_idx,
                         base=x._base if x._base is not None else x, pinned=x._pinned)

    def view_as_complex(self) -> 'NBXTensor':
        """float [..,N,2] → complex [..,N] (interleaved real/imag).

        complex64 requires fp32 real/imag pairs and complex128 requires fp64
        pairs — there is no complex32. A half-precision input (fp16/bf16, e.g.
        a rotary-embedding q/k narrowed to the V100 compute dtype) therefore
        CANNOT be reinterpreted as complex64 in place: the storage is half the
        size the complex64 tag implies, so the downstream complex kernels read
        past the data and produce garbage/Inf. Promote fp16/bf16 to fp32 first
        (a real copy), then tag complex64. fp32→complex64 and fp64→complex128
        stay zero-copy views.
        """
        x = self if self.is_contiguous() else self.contiguous()
        if x._dtype in (NBXDtype.float16, NBXDtype.bfloat16):
            x = x.to(NBXDtype.float32)
        assert x._shape and x._shape[-1] == 2, \
            f"view_as_complex expects last dim == 2, got {x._shape}"
        cdt = NBXDtype.complex128 if x._dtype == NBXDtype.float64 else NBXDtype.complex64
        new_shape = x._shape[:-1]
        return NBXTensor(x._data_ptr, new_shape, contiguous_strides(new_shape),
                         cdt, x._device, x._offset // 2,
                         device_idx=x._device_idx,
                         base=x._base if x._base is not None else x, pinned=x._pinned)

    @property
    def real(self) -> 'NBXTensor':
        """Real part (strided view, stride 2) of a complex tensor; self if real."""
        if not self.is_complex():
            return self
        vr = self.view_as_real()
        return vr.select(vr.ndim - 1, 0)

    @property
    def imag(self) -> 'NBXTensor':
        """Imaginary part (strided view, stride 2) of a complex tensor."""
        if not self.is_complex():
            return NBXTensor.zeros(self._shape, dtype=self._dtype, device=self._device)
        vr = self.view_as_real()
        return vr.select(vr.ndim - 1, 1)

    # ========================================================================
    # SHAPE QUERY METHODS
    # ========================================================================

    def size(self, dim=None):
        if dim is not None:
            return self._shape[dim]
        return self._shape

    def dim(self) -> int:
        return len(self._shape)

    def stride(self, dim=None):
        if dim is not None:
            return self._strides[dim]
        return self._strides

    # ========================================================================
    # __cuda_array_interface__ — standard CUDA interop
    # ========================================================================

    @property
    def __cuda_array_interface__(self):
        return {
            'shape': self._shape,
            'typestr': _DTYPE_TYPESTR[self._dtype],
            'data': (self.data_ptr(), False),
            'version': 3,
            'strides': tuple(s * dtype_size(self._dtype) for s in self._strides),
        }

    # ========================================================================
    # FACTORY METHODS — raw cudaMalloc
    # ========================================================================

    @staticmethod
    def empty(shape, dtype=None, device='cuda') -> 'NBXTensor':
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        nbx_dt = dtype if isinstance(dtype, NBXDtype) else parse_dtype(str(dtype)) if dtype else NBXDtype.float32
        dev_str = 'cuda'
        # C3: resolve an explicit device index WITHOUT touching the
        # driver; cudaGetDevice fires at most once per allocation, and
        # not at all on the hot path (explicit device matching the
        # cached current device).
        dev_idx = None
        if isinstance(device, str):
            if ':' in device:
                dev_idx = int(device.split(':')[1])
        elif hasattr(device, '_device_idx'):
            dev_idx = device._device_idx
        elif hasattr(device, 'index') and device.index is not None:
            dev_idx = device.index

        numel = math.prod(shape) if shape else 1
        nbytes = numel * dtype_size(nbx_dt)
        if nbytes == 0:
            if dev_idx is None:
                dev_idx = DeviceAllocator.get_device()
            return NBXTensor(0, shape, contiguous_strides(shape), nbx_dt, dev_str, device_idx=dev_idx)

        if dev_idx is None:
            # No explicit device: allocate on the current one. Single
            # query, no switch, and the resolved idx is threaded into
            # malloc_cuda so it does not re-query.
            dev_idx = DeviceAllocator.get_device()
            ptr = DeviceAllocator.malloc_cuda(nbytes, dev_idx=dev_idx)
        elif dev_idx == _rt._cached_device_idx():
            # Explicit device already fully current (C2 cache invariant
            # guarantees the runtime device equals the cached idx): no
            # driver query, no switch.
            ptr = DeviceAllocator.malloc_cuda(nbytes, dev_idx=dev_idx)
        else:
            # Multi-GPU: allocate on the requested device, restore the
            # previous one after. The restore target comes from a real
            # driver query (never from the cache) so a cold/invalidated
            # cache can never misdirect the restore.
            cur = DeviceAllocator.get_device()
            if cur != dev_idx:
                DeviceAllocator.set_device(dev_idx)
            ptr = DeviceAllocator.malloc_cuda(nbytes, dev_idx=dev_idx)
            if cur != dev_idx:
                DeviceAllocator.set_device(cur)

        return NBXTensor(ptr, shape, contiguous_strides(shape), nbx_dt, dev_str,
                         owns_data=True, device_idx=dev_idx)

    @staticmethod
    def zeros(shape, dtype=None, device='cuda') -> 'NBXTensor':
        t = NBXTensor.empty(shape, dtype, device)
        if t._nbytes > 0:
            DeviceAllocator.memset_cuda(t._data_ptr, 0, t._nbytes)
        return t

    @staticmethod
    def ones(shape, dtype=None, device='cuda') -> 'NBXTensor':
        return _fill_constant(NBXTensor.empty(shape, dtype, device), 1.0)

    @staticmethod
    def empty_like(other: 'NBXTensor', dtype=None, device=None) -> 'NBXTensor':
        dev = device if device else f"cuda:{other._device_idx}"
        return NBXTensor.empty(other._shape,
                               dtype if dtype is not None else other._dtype,
                               dev)

    @staticmethod
    def zeros_like(other: 'NBXTensor', dtype=None, device=None) -> 'NBXTensor':
        return NBXTensor.zeros(other._shape,
                               dtype if dtype is not None else other._dtype,
                               device if device else other._device)

    @staticmethod
    def ones_like(other: 'NBXTensor', dtype=None, device=None) -> 'NBXTensor':
        return NBXTensor.ones(other._shape,
                              dtype if dtype is not None else other._dtype,
                              device if device else other._device)

    @staticmethod
    def from_numpy(arr, dtype: 'Optional[NBXDtype]' = None) -> 'NBXTensor':
        """Load numpy array to GPU. For weight loading from safetensors.

        `dtype` names the NBX dtype the BITS in `arr` already are, for the
        types numpy cannot express — bf16 travels in a uint16 container. It
        is authoritative and must agree with the container's element size:
        the buffer allocated here is `arr.nbytes`, and a tensor whose dtype
        implies a different element size declares a length its own
        allocation does not have. That mismatch used to be created by
        retagging `_dtype` after construction (the tensor kept the
        `_elem_size` / `_nbytes` of the dtype it was BUILT with): TinyLlama's
        22 rotary tables, uint16 bf16 bits mistagged float32 by the fallback
        below, each declared 524288 bytes over a 262144-byte buffer, and
        every consumer that trusts `_nbytes` then ran off the end into
        whichever allocation followed it — which is why the answer depended
        on the sequence length. Measured 2026-09-07 on Apple.

        An unmapped numpy dtype is refused rather than silently called
        float32: guessing an element size is what produced the above.
        """
        import numpy as np
        dtype_name = str(arr.dtype)
        if dtype is not None:
            nbx_dt = dtype if isinstance(dtype, NBXDtype) else parse_dtype(dtype)
            if dtype_size(nbx_dt) != arr.dtype.itemsize:
                raise ValueError(
                    f"from_numpy: dtype {nbx_dt.name} is "
                    f"{dtype_size(nbx_dt)} bytes per element but the numpy "
                    f"container '{dtype_name}' is {arr.dtype.itemsize}; the "
                    f"buffer is sized from the container, so the two must "
                    f"agree.")
        elif dtype_name in _NUMPY_DTYPE_MAP:
            nbx_dt = _NUMPY_DTYPE_MAP[dtype_name]
        else:
            raise TypeError(
                f"from_numpy: numpy dtype '{dtype_name}' has no NBX "
                f"equivalent. Pass dtype=NBXDtype.<t> to declare what the "
                f"bits are (bf16 travels as uint16); do not let the element "
                f"size be guessed.")
        arr_c = np.ascontiguousarray(arr)
        nbytes = arr_c.nbytes
        shape = arr_c.shape
        dev_idx = DeviceAllocator.get_device()
        ptr = DeviceAllocator.malloc_cuda(nbytes)
        # H2D copy
        DeviceAllocator.memcpy(ptr, arr_c.ctypes.data, nbytes, kind=1)
        return NBXTensor(ptr, shape, contiguous_strides(shape), nbx_dt, 'cuda',
                         owns_data=True, device_idx=dev_idx)

    @staticmethod
    def from_raw(ptr: int, shape: Tuple[int, ...], dtype: NBXDtype,
                 device: str = 'cuda', owns_data: bool = False,
                 device_idx: int = 0, base=None) -> 'NBXTensor':
        """Wrap an existing raw pointer. For interop at boundaries.

        Args:
            base: Reference to the object owning the memory (prevents GC).
                  For torch.Tensor → NBXTensor conversion, pass the torch.Tensor
                  so it stays alive as long as the NBXTensor exists.
        """
        return NBXTensor(ptr, shape, contiguous_strides(shape), dtype, device,
                         owns_data=owns_data, device_idx=device_idx, base=base)

    # ------------------------------------------------------------------
    # CPU form — zero3 weight offload and similar scenarios where
    # weights live on host memory between promotions to GPU. Uses numpy
    # for unpinned backing (default) or cudaMallocHost for pinned
    # backing (faster H2D DMA). Zero torch dependency.
    # ------------------------------------------------------------------

    @staticmethod
    def empty_cpu(shape, dtype: 'Optional[NBXDtype]' = None,
                  pinned: bool = False) -> 'NBXTensor':
        """Allocate a CPU-backed NBXTensor.

        Unpinned (pinned=False, default): backed by a numpy array. Cheap
        to allocate, usable by numpy-based prep code, but incurs a
        pageable→pinned staging copy on H2D.

        Pinned (pinned=True): backed by cudaMallocHost-allocated memory.
        Enables non_blocking DMA at ~2× unpinned throughput. Used by
        zero3 for weights that will be transferred on the hot path.
        """
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        nbx_dt = dtype if isinstance(dtype, NBXDtype) else (
            parse_dtype(str(dtype)) if dtype else NBXDtype.float32)
        numel = math.prod(shape) if shape else 1
        elem = dtype_size(nbx_dt)
        nbytes = numel * elem

        if nbytes == 0:
            return NBXTensor(0, shape, contiguous_strides(shape), nbx_dt,
                             'cpu', device_idx=0)

        if pinned:
            ptr = DeviceAllocator.malloc_host_pinned(nbytes)
            return NBXTensor(ptr, shape, contiguous_strides(shape), nbx_dt,
                             'cpu', owns_data=True, device_idx=0,
                             pinned=True)
        else:
            # Numpy backing — keep the array alive via _base.
            import numpy as np
            np_dtype_name = {
                NBXDtype.float32: np.float32,
                NBXDtype.float16: np.float16,
                NBXDtype.bfloat16: np.uint16,   # bf16 stored as uint16 bits
                NBXDtype.float64: np.float64,
                NBXDtype.int64: np.int64,
                NBXDtype.int32: np.int32,
                NBXDtype.int16: np.int16,
                NBXDtype.int8: np.int8,
                NBXDtype.uint8: np.uint8,
                NBXDtype.bool_: np.uint8,
                NBXDtype.complex64: np.complex64,
                NBXDtype.complex128: np.complex128,
            }
            # TOTAL, and it REFUSES what it does not know. `.get(nbx_dt,
            # np.float32)` stood here with no complex entry, so the host buffer
            # was allocated at four bytes an element while `to_cpu` copied
            # `self._nbytes`, eight bytes an element, into it. Double the buffer,
            # every complex readback, at every size: `malloc(): corrupted top
            # size`. And not at the copy -- the first small probe printed a
            # correct-looking array and the NEXT allocation died, which is how it
            # survived. complex128 was the same bug at four times over.
            #
            # A default makes "a dtype I do not know" indistinguishable from
            # "float32": a wrong-sized allocation waiting for that dtype to
            # arrive. Every NBXDtype is named above; a new one that is not gets a
            # refusal naming itself, not a corrupted heap.
            if nbx_dt not in np_dtype_name:
                raise RuntimeError(
                    f"empty_cpu has no numpy backing for {nbx_dt}: add it to the "
                    f"map above. Defaulting would size the host buffer for "
                    f"another dtype and the copy that follows would write past it.")
            arr = np.empty(shape, dtype=np_dtype_name[nbx_dt])
            return NBXTensor(arr.ctypes.data, shape,
                             contiguous_strides(shape), nbx_dt, 'cpu',
                             owns_data=True, device_idx=0,
                             base=arr,  # keep numpy alive
                             pinned=False)

    def to_cuda(self, device_idx: int = 0) -> 'NBXTensor':
        """Copy this tensor to a CUDA device. Returns a new GPU NBXTensor.

        Handles CPU→GPU (kind=1 H2D) and GPU→GPU (kind=3 D2D). If the
        tensor is already on the requested CUDA device, returns self.
        A view whose elements do not tile one dense span (an expand view's
        stride-0 axis, or a narrow on an inner axis — a stacked expert
        slab's gate half) is materialised first: the copy is ONE memcpy of
        nbytes from data_ptr, which is the view's bytes only when they are
        dense.
        """
        if self._device == 'cuda' and self._device_idx == device_idx:
            return self
        src = self if self.spans_densely() else self.contiguous()
        DeviceAllocator.set_device(device_idx)
        ptr = DeviceAllocator.malloc_cuda(src._nbytes)
        if src._nbytes > 0:
            # kind: 1 = H2D if source is CPU, 3 = D2D if source is GPU.
            kind = 1 if src._device == 'cpu' else 3
            DeviceAllocator.memcpy(ptr, src.data_ptr(), src._nbytes, kind=kind)
        return NBXTensor(ptr, src._shape, src._strides, src._dtype,
                         'cuda', owns_data=True, device_idx=device_idx)

    def to_cuda_async(self, device_idx: int = 0, stream: int = 0) -> 'NBXTensor':
        """Non-blocking variant of to_cuda for zero3 pipelining.

        Enqueues the memcpy on `stream` instead of running synchronously.
        Caller is responsible for sequencing — either via stream_wait_event
        before reading the returned tensor, or via stream_synchronize.

        For H2D to actually overlap with compute on the default stream, the
        source NBXTensor MUST be pinned (self._pinned == True). Otherwise
        the driver falls back to a synchronous bounce-buffer copy and no
        overlap happens.

        Returns a GPU NBXTensor backed by a freshly-malloc'd pointer. The
        allocation itself is synchronous (cudaMalloc); only the fill is
        asynchronous.
        """
        if self._device == 'cuda' and self._device_idx == device_idx:
            return self
        src = self if self.spans_densely() else self.contiguous()
        DeviceAllocator.set_device(device_idx)
        ptr = DeviceAllocator.malloc_cuda(src._nbytes)
        if src._nbytes > 0:
            kind = 1 if src._device == 'cpu' else 3
            DeviceAllocator.memcpy_async(
                ptr, src.data_ptr(), src._nbytes,
                kind=kind, stream=stream)
        return NBXTensor(ptr, src._shape, src._strides, src._dtype,
                         'cuda', owns_data=True, device_idx=device_idx)

    def to_cpu(self, pinned: bool = False) -> 'NBXTensor':
        """Copy this tensor to host memory. Returns a new CPU NBXTensor.

        pinned=True allocates via cudaMallocHost for fast future H2D
        promotion; pinned=False uses numpy backing.
        """
        if self._device == 'cpu' and self._pinned == pinned:
            return self
        dst = NBXTensor.empty_cpu(self._shape, self._dtype, pinned=pinned)
        # The destination is row-major: a source that is not (a transpose, a
        # narrow on an inner axis, an expand) is laid out row-major on its own
        # side first — one memcpy of nbytes from data_ptr is its elements only
        # when it already is.
        src = self if self.is_contiguous() else self.contiguous()
        if src._nbytes > 0:
            # kind: 2 = D2H from GPU, 0 = H2H if we're re-packing CPU.
            kind = 2 if src._device == 'cuda' else 0
            DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(),
                                   src._nbytes, kind=kind)
            # A device result crossing to the host is an observation, and a
            # D2H copy has already flushed. Checking here rather than only at
            # `sync_device` matters because a decode never calls that: it
            # reads its logits, and this is that read.
            if kind == 2:
                _faults.check_device_faults()
        return dst

    def numpy(self):
        """Host numpy array (boundary conversion — copies to CPU). Used at the
        triton output boundary (e.g. saving a waveform). float/int dtypes only;
        bf16 has no numpy dtype (cast to float32 first)."""
        import numpy as np
        import ctypes
        f = self.contiguous()
        if f._device != 'cpu':
            f = f.to_cpu()
        typestr = _DTYPE_TYPESTR[f._dtype]          # every member has its type string in the dtype table
        nbytes = f.numel() * dtype_size(f._dtype)
        # A view over the host buffer, then one copy. `ctypes.string_at(ptr, n)` hands `n` to
        # PyBytes_FromStringAndSize as a C int: a tensor of 2 GiB or more read back as
        # "Negative size passed to PyBytes_FromStringAndSize" (the certifier on 2026-09-07:
        # seven census shapes whose fp32 output exceeds 2 GiB — a 4K convolution, a
        # 1,221,120-row matmul, a 16384² baddbmm — "no config could run").
        if nbytes == 0:
            return np.empty(tuple(self._shape), dtype=np.dtype(typestr))
        buf = (ctypes.c_uint8 * nbytes).from_address(f.data_ptr())
        view = np.ctypeslib.as_array(buf).view(np.dtype(typestr)).reshape(tuple(self._shape))
        out = view.copy()
        del view, buf                                   # nothing outlives the host buffer
        return out

    def pin_host(self) -> 'NBXTensor':
        """Promote an unpinned CPU tensor to pinned host memory.

        No-op if the tensor is already pinned. Used by Zero3Strategy to
        accelerate subsequent H2D transfers during block-wise execution.
        Raises on non-CPU tensors — pinning a GPU tensor is nonsensical.
        """
        if self._device != 'cpu':
            raise RuntimeError(
                f"pin_host() called on device={self._device}; pinning only "
                f"applies to CPU-backed NBXTensors")
        if self._pinned:
            return self
        return self.to_cpu(pinned=True)

    # ========================================================================
    # METADATA OPS — pure Python, zero data movement
    # ========================================================================

    def view(self, *shape) -> 'NBXTensor':
        # Match PyTorch's aten::view tolerance: if the source tensor is
        # expanded (stride-0 broadcast) or otherwise non-contiguous, the
        # old path silently re-strode the view as contiguous while the
        # underlying memory still held the unmaterialized broadcast —
        # later kernel launches walked past the end of the backing
        # allocation and got "Pointer argument cannot be accessed from
        # Triton" on the second batch iteration (Janus RoPE bmm). Matches
        # reshape's fallback: contiguous() first, then re-stride.
        if not self.is_contiguous():
            return self.contiguous().view(*shape)
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        neg_idx = None
        known = 1
        for i, s in enumerate(shape):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx is not None:
            shape = list(shape)
            shape[neg_idx] = self._numel // known
            shape = tuple(shape)
        return NBXTensor(self._data_ptr, shape, contiguous_strides(shape),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def reshape(self, *shape) -> 'NBXTensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        if self.is_contiguous():
            return self.view(*shape)
        return self.contiguous().view(*shape)

    def unflatten(self, dim: int, sizes) -> 'NBXTensor':
        """Split ONE dim into `sizes` (their product its extent), as torch's `unflatten` does: a pure
        view on any strides — the split dim walks memory at its own stride, so its parts walk it at
        that stride times the sizes after them. `reshape` makes a non-contiguous tensor contiguous
        first: a sliced stretch's reading of a strided input copied the WHOLE input for every slice
        (HAT-S at 2048 px, q/k/v views of the 6 912 MB qkv, 3 x 2 304 MB per pass, 2026-10-22)."""
        dim = dim % self.ndim
        sizes = [int(v) for v in sizes]
        prod = 1
        for v in sizes:
            prod *= v
        if prod != int(self._shape[dim]):
            raise ValueError(f"unflatten: sizes {sizes} do not make extent {self._shape[dim]} of dim {dim}")
        st, inner = [], int(self._strides[dim])
        for v in reversed(sizes):
            st.append(inner)
            inner *= v
        shape = tuple(self._shape[:dim]) + tuple(sizes) + tuple(self._shape[dim + 1:])
        strides = tuple(self._strides[:dim]) + tuple(reversed(st)) + tuple(self._strides[dim + 1:])
        return self.as_strided(shape, strides)

    def merge_dims(self, d0: int, d1: int) -> 'NBXTensor':
        """Merge two ADJACENT dims into one — as a pure view when the
        strides allow it, else via reshape (which materialises).

        The merge is a legal view iff stride(d0) == shape(d1)*stride(d1)
        — the two dims already walk memory as one. This holds on tensors
        whose non-contiguity lives in ANOTHER dim: the bucketed KV-cache
        view `buffer[:, :, :padded, :]` is non-contiguous in the T dim,
        so plain `reshape(BH, T, D)` copied the whole padded K/V per
        layer per decode step (the copies-at-source census, 2026-08-23),
        while its (B, H) pair still merges for free. Callers that feed
        the merged tensor to a kernel must handle its strides
        (`bmm(..., allow_strided_b=True)`); the fallback keeps the
        semantics identical for every stride pattern.
        """
        d0 = d0 % self.ndim
        d1 = d1 % self.ndim
        if d1 != d0 + 1:
            raise ValueError(f"merge_dims needs adjacent dims, got {d0},{d1}")
        if self._strides[d0] != self._shape[d1] * self._strides[d1]:
            new_shape = (self._shape[:d0]
                         + (self._shape[d0] * self._shape[d1],)
                         + self._shape[d1 + 1:])
            return self.reshape(new_shape)
        new_shape = (self._shape[:d0]
                     + (self._shape[d0] * self._shape[d1],)
                     + self._shape[d1 + 1:])
        new_strides = (self._strides[:d0] + (self._strides[d1],)
                       + self._strides[d1 + 1:])
        return NBXTensor(self._data_ptr, new_shape, new_strides,
                         self._dtype, self._device, self._offset,
                         device_idx=self._device_idx,
                         base=self._base if self._base is not None else self,
                         pinned=self._pinned)

    def unsqueeze(self, dim: int) -> 'NBXTensor':
        dim = dim % (self.ndim + 1) if dim >= 0 else dim + self.ndim + 1
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        stride_val = new_strides[dim] * self._shape[dim] if dim < len(self._strides) else 1
        new_shape.insert(dim, 1)
        new_strides.insert(dim, stride_val)
        return NBXTensor(self._data_ptr, tuple(new_shape), tuple(new_strides),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def squeeze(self, dim=None) -> 'NBXTensor':
        if dim is not None:
            dim = dim % self.ndim
            if self._shape[dim] != 1:
                return self
            s = list(self._shape)
            st = list(self._strides)
            s.pop(dim)
            st.pop(dim)
        else:
            s, st = [], []
            for sz, sr in zip(self._shape, self._strides):
                if sz != 1:
                    s.append(sz)
                    st.append(sr)
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def permute(self, *dims) -> 'NBXTensor':
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return NBXTensor(self._data_ptr,
                         tuple(self._shape[d] for d in dims),
                         tuple(self._strides[d] for d in dims),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def transpose(self, dim0: int, dim1: int) -> 'NBXTensor':
        dim0, dim1 = dim0 % self.ndim, dim1 % self.ndim
        s = list(self._shape)
        st = list(self._strides)
        s[dim0], s[dim1] = s[dim1], s[dim0]
        st[dim0], st[dim1] = st[dim1], st[dim0]
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def t(self) -> 'NBXTensor':
        if self.ndim < 2:
            return self
        return self.transpose(0, 1)

    @property
    def T(self) -> 'NBXTensor':
        return self.transpose(-2, -1) if self.ndim >= 2 else self

    def expand(self, *shape) -> 'NBXTensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        new_s, new_st = broadcast_strides(self._shape, self._strides, shape)
        return NBXTensor(self._data_ptr, new_s, new_st,
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def expand_as(self, other) -> 'NBXTensor':
        return self.expand(*other._shape if isinstance(other, NBXTensor) else other.shape)

    def repeat(self, *sizes) -> 'NBXTensor':
        """aten::repeat — TILE the tensor `sizes[i]` times along each dim
        (numpy.tile / torch.Tensor.repeat), NOT element-wise repeat_interleave.
        When len(sizes) > ndim, leading singleton dims are prepended. R33-pure:
        view → expand (stride-0 broadcast) → contiguous → reshape; the
        interleaved `[1,d0,1,d1,…]` view collapses to `[r0*d0, r1*d1, …]`.
        """
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = tuple(sizes[0])
        sizes = tuple(int(s) for s in sizes)
        m = len(sizes)
        assert m >= self.ndim, f"repeat sizes {sizes} fewer than ndim {self.ndim}"
        x = self.contiguous()
        base = (1,) * (m - x.ndim) + tuple(x._shape)
        view_shape, expand_shape, out_shape = [], [], []
        for i in range(m):
            view_shape += [1, base[i]]
            expand_shape += [sizes[i], base[i]]
            out_shape.append(sizes[i] * base[i])
        return (x.view(tuple(view_shape))
                 .expand(tuple(expand_shape))
                 .contiguous()
                 .reshape(tuple(out_shape)))

    def narrow(self, dim: int, start: int, length: int) -> 'NBXTensor':
        dim = dim % self.ndim
        s = list(self._shape)
        s[dim] = length
        new_off = self._offset + start * self._strides[dim]
        return NBXTensor(self._data_ptr, tuple(s), self._strides,
                         self._dtype, self._device, new_off, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def select(self, dim: int, index: int) -> 'NBXTensor':
        dim = dim % self.ndim
        # PyTorch semantics: negative index counts from the end.
        # Without this, `select(0, -1)` on a (300,) contiguous tensor
        # produced `new_off = -1 * 1 = -1`, and `data_ptr() = base - elem`
        # pointed 4 bytes BEFORE the cudaMalloc'd block. Triton's
        # cuPointerGetAttribute then flagged the pointer as CPU memory
        # and raised "Pointer argument (at 0) cannot be accessed from
        # Triton (cpu tensor?)" (PixArt T5 encoder aten.select+aten.add
        # chain at position_ids build).
        if index < 0:
            index = self._shape[dim] + index
        new_off = self._offset + index * self._strides[dim]
        s = list(self._shape)
        st = list(self._strides)
        s.pop(dim)
        st.pop(dim)
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, new_off, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'NBXTensor':
        start_dim = start_dim % self.ndim
        end_dim = end_dim % self.ndim
        if start_dim == end_dim:
            return self
        flat = 1
        for i in range(start_dim, end_dim + 1):
            flat *= self._shape[i]
        new_s = list(self._shape[:start_dim]) + [flat] + list(self._shape[end_dim + 1:])
        return self.view(*new_s)

    def contiguous(self) -> 'NBXTensor':
        if self.is_contiguous():
            return self
        if self._device == 'cpu':
            return self._contiguous_cpu()
        new = NBXTensor.empty(self._shape, self._dtype, f"cuda:{self._device_idx}")
        n = self._numel
        if n > 0:
            _strided_copy(self, new)
        return new

    def _contiguous_cpu(self) -> 'NBXTensor':
        """CPU-side materialization of a non-contiguous view.

        The shared _strided_copy path launches a Triton kernel over GPU
        pointers, so it cannot reach host memory. This helper performs
        the same strided → contiguous copy in numpy (C-implemented
        `ascontiguousarray` over an `as_strided` view) and preserves the
        source's pinned / unpinned backing. Expand views (stride == 0)
        materialize correctly because numpy replays the same source
        element along zero-stride axes.
        """
        import numpy as np
        dst = NBXTensor.empty_cpu(self._shape, self._dtype, pinned=self._pinned)
        if self._numel == 0:
            return dst
        esz = dtype_size(self._dtype)
        # Byte window touched by the strided view starting at self.data_ptr():
        # last-reachable element offset + one element for esz bytes.
        max_elem_offset = 1 + sum(
            (dim - 1) * st for dim, st in zip(self._shape, self._strides)
            if dim > 0)
        n_bytes = max_elem_offset * esz
        src_bytes = np.ctypeslib.as_array(
            ctypes.cast(self.data_ptr(), ctypes.POINTER(ctypes.c_ubyte)),
            shape=(n_bytes,),
        )
        byte_strides = tuple(s * esz for s in self._strides)
        strided_view = np.lib.stride_tricks.as_strided(
            src_bytes,
            shape=tuple(self._shape) + (esz,),
            strides=byte_strides + (1,),
        )
        contig = np.ascontiguousarray(strided_view)
        ctypes.memmove(dst.data_ptr(), contig.ctypes.data, dst._nbytes)
        return dst

    def unfold(self, dimension: int, size: int, step: int) -> 'NBXTensor':
        dimension = dimension % self.ndim
        n = (self._shape[dimension] - size) // step + 1
        s = list(self._shape)
        st = list(self._strides)
        s[dimension] = n
        s.append(size)
        st[dimension] = self._strides[dimension] * step
        st.append(self._strides[dimension])
        return NBXTensor(self._data_ptr, tuple(s), tuple(st),
                         self._dtype, self._device, self._offset, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def as_strided(self, size, stride, storage_offset=None) -> 'NBXTensor':
        off = storage_offset if storage_offset is not None else self._offset
        return NBXTensor(self._data_ptr, tuple(size), tuple(stride),
                         self._dtype, self._device, off, device_idx=self._device_idx, base=self._base if self._base is not None else self, pinned=self._pinned)

    def view_as(self, other) -> 'NBXTensor':
        s = other._shape if isinstance(other, NBXTensor) else other.shape
        return self.view(*s)

    def reshape_as(self, other) -> 'NBXTensor':
        s = other._shape if isinstance(other, NBXTensor) else other.shape
        return self.reshape(*s)

    def movedim(self, src: int, dst: int) -> 'NBXTensor':
        src, dst = src % self.ndim, dst % self.ndim
        dims = list(range(self.ndim))
        dims.pop(src)
        dims.insert(dst, src)
        return self.permute(*dims)

    def __getitem__(self, key):
        """Basic tensor indexing: t[0], t[:, :, 0], t[:, 2:5], etc.

        Fancy indexing by Python list of integers (e.g.
        t[:, [0, -3, -2, -1]]) is NOT supported. NBXTensor enforces
        explicit slicing via narrow, cat, or index_select_wrapper to
        guarantee numerical and structural parity with torch. Prior to
        this guard, list keys were silently no-op'd, returning the
        original tensor unchanged — a bomb that masked a real bug in
        the Sana CHI handler (see commit log). The crash is intentional
        and forces the explicit pattern at every call site.
        """
        # Reject fancy indexing by list — silently no-op'd before, now hard-fails.
        if isinstance(key, list):
            raise RuntimeError(
                "ZERO FALLBACK: NBXTensor does not support fancy indexing by "
                "Python list (e.g. tensor[[0, 1, 2]]). Use narrow + cat for "
                "static patterns or index_select_wrapper for dynamic indices. "
                "See src/neurobrix/core/components/handlers/text_encoder_handler.py "
                "for an example of the narrow + cat pattern."
            )
        if isinstance(key, tuple):
            for k in key:
                if isinstance(k, list):
                    raise RuntimeError(
                        "ZERO FALLBACK: NBXTensor does not support fancy indexing "
                        "by Python list within a tuple (e.g. tensor[:, [0, 1, 2]]). "
                        "Use narrow + cat for static patterns or "
                        "index_select_wrapper for dynamic indices. See "
                        "src/neurobrix/core/components/handlers/text_encoder_handler.py "
                        "for an example of the narrow + cat pattern."
                    )
        if isinstance(key, int):
            return self.select(0, key)
        # P-TRITON-CHAIN-CPU-POINTER C3d 2026-05-15: handle negative slice
        # start/stop per torch / numpy contract. Previous code did
        # `k.start or 0` which silently converted None → 0 but left
        # negative values untouched. For `t[-N:]` (slice(-N, None)),
        # `start` stayed at `-N` and `narrow(dim, -N, length)` computed
        # `new_off = offset - N*stride` → data_ptr pointed BEFORE the
        # cudaMalloc'd block → Triton's cuPointerGetAttribute rejected
        # the pointer with "Pointer argument (at 0) cannot be accessed
        # from Triton (cpu tensor?)". The Sana 4Kpx VAE chain wrapper's
        # `halo_carry[:, :, -top_size:, :]` triggered this at the third
        # iteration of the 4096² chains (chains 7-9 of run).
        if isinstance(key, slice):
            start, stop = slice_bounds(key.start, key.stop, self._shape[0], key.step)
            return self.narrow(0, start, stop - start)
        if isinstance(key, tuple):
            result = self
            dim_offset = 0
            for dim, k in enumerate(key):
                actual_dim = dim - dim_offset
                if isinstance(k, int):
                    result = result.select(actual_dim, k)
                    dim_offset += 1  # select reduces ndim
                elif isinstance(k, slice):
                    start, stop = slice_bounds(k.start, k.stop, result._shape[actual_dim], k.step)
                    result = result.narrow(
                        actual_dim, start, stop - start)
                # slice(None) = : → no-op on this dim
        return result

    def __setitem__(self, key, value):
        """Indexed write: buffer[..., start:end, :] = source.

        Handles the KV cache pattern: contiguous slice assignment.
        Uses strided_scatter_kernel when the destination view is non-contiguous.
        Guards: dtype cast and device transfer before copy.
        """
        dst = self[key]
        if isinstance(value, NBXTensor):
            # Dtype guard: a complex source is cast first (the nd copy kernel
            # is real-only); a real source is converted on the copy's store.
            if value._dtype != dst._dtype and (value.is_complex() or dst.is_complex()):
                value = value.to(dst._dtype)
            # Device guard: transfer source to destination device
            if hasattr(value, '_device_idx') and hasattr(dst, '_device_idx') and value._device_idx != dst._device_idx:
                src_contig = value.contiguous()
                tmp = NBXTensor.empty(src_contig._shape, src_contig._dtype, f"cuda:{dst._device_idx}")
                DeviceAllocator.memcpy(tmp.data_ptr(), src_contig.data_ptr(), src_contig._nbytes, kind=3)
                value = tmp
            if (value.is_contiguous() and dst.is_contiguous()
                    and tuple(value._shape) == tuple(dst._shape)):
                DeviceAllocator.memcpy(dst.data_ptr(), value.data_ptr(), value._nbytes)
            elif dst._numel > 0:
                # One launch reads the source by its strides and writes the
                # view by its own — no contiguous copy of the source first
                # (the KV cache's V arrives as a head-strided view: its write
                # was a strided copy, a cast and a scatter per layer per token).
                _copy_nd(value, dst)

    def unbind(self, dim: int = 0) -> tuple:
        dim = dim % self.ndim
        return tuple(self.select(dim, i) for i in range(self._shape[dim]))

    # ========================================================================
    # DATA OPS — cudaMemcpy
    # ========================================================================

    def clone(self) -> 'NBXTensor':
        new = NBXTensor.empty(self._shape, self._dtype, f"cuda:{self._device_idx}")
        n = self._numel
        if n > 0:
            if self.is_contiguous():
                DeviceAllocator.memcpy(new._data_ptr, self.data_ptr(), self._nbytes)
            else:
                _strided_copy(self, new)
        return new

    def detach(self) -> 'NBXTensor':
        return self  # no autograd in triton mode

    def fill_(self, value) -> 'NBXTensor':
        if self._nbytes == 0:
            return self
        # aten::fill carries its value as a 0-dim tensor (e.g. via
        # aten::lift_fresh); collapse it to a host scalar before the kernel.
        if isinstance(value, NBXTensor):
            value = value.item()
        if value == 0:
            DeviceAllocator.memset_cuda(self.data_ptr(), 0, self._nbytes)
            return self
        # Non-zero fill via the pure-Triton fill_kernel.
        return _fill_constant(self, value)

    def copy_(self, src, non_blocking: bool = False) -> 'NBXTensor':
        if isinstance(src, NBXTensor):
            # PyTorch Tensor.copy_ converts src to self's dtype. A raw byte
            # memcpy across a dtype boundary reinterprets the bits — e.g. an
            # fp16 source written into an fp32 destination decodes the fp16
            # bit-pairs as fp32 (≈0), and the byte counts differ (the chatterbox
            # s3gen prompt-feat copy: fp16 src 50240 B into fp32 dst 100480 B).
            # Cast to self's dtype first, then contiguous-pack the source so the
            # flat copy of self._nbytes bytes is well-defined.
            if src.dtype != self.dtype and (src.is_complex() or self.is_complex()):
                src = src.to(self.dtype)
            if not (src.is_contiguous() and self.is_contiguous()
                    and src.dtype == self.dtype
                    and tuple(src._shape) == tuple(self._shape)) and not self.is_complex():
                # Strides on either side, a dtype to convert, or a source to
                # broadcast: one launch, the source read in place.
                _copy_nd(src, self)
                return self
            if not src.is_contiguous():
                src = src.contiguous()
            if not self.is_contiguous():
                # Destination is a STRIDED view, e.g. an in-place slice-assignment
                # `buf[..., :k] = src` that the tracer lowers to slice + copy_.
                # A flat memcpy of self._nbytes would write the source CONTIGUOUSLY
                # at self.data_ptr(), ignoring the view's strides — scattering the
                # data to the wrong addresses. (Wan RoPE builds the rotated Q/K via
                # an empty_like buffer + slice-copy; the flat memcpy left the buffer
                # garbage → zero self-attn Q/K → washed-out VACE.) Scatter the
                # contiguous src into self's strided layout via the same dedicated
                # kernel `__setitem__` and `_fill_constant` use. R33-pure.
                #
                # The scatter maps flat src index i -> dst's strided position of
                # multi-index i, so it requires equal element counts. A broadcast
                # copy_ (src smaller than dst) into a strided view is unsupported
                # — surface it loudly rather than silently under-writing. The old
                # flat memcpy was already broken for that case (it read past src),
                # so this raises no previously-working path.
                if src._numel != self._numel:
                    raise RuntimeError(
                        "NBXTensor.copy_ into a strided (non-contiguous) "
                        "destination requires matching element counts; "
                        "broadcast copy_ into a strided view is unsupported "
                        f"(src._numel={src._numel} dst._numel={self._numel}).")
                if self.is_complex():
                    # The scatter kernel sees a plain float ptr; reinterpret both
                    # as a float view with a trailing [2] (re/imag) so each complex
                    # element's two floats land at the correct strided positions.
                    # Mirrors _strided_copy's complex handling.
                    _fdt = (NBXDtype.float64 if self._dtype == NBXDtype.complex128
                            else NBXDtype.float32)
                    _src_shape2 = tuple(src._shape) + (2,)
                    _fsrc = NBXTensor(
                        src._data_ptr, _src_shape2, contiguous_strides(_src_shape2),
                        _fdt, src._device, src._offset * 2,
                        device_idx=src._device_idx,
                        base=src._base if src._base is not None else src,
                        pinned=src._pinned)
                    _fdst = NBXTensor(
                        self._data_ptr, tuple(self._shape) + (2,),
                        tuple(s * 2 for s in self._strides) + (1,),
                        _fdt, self._device, self._offset * 2,
                        device_idx=self._device_idx,
                        base=self._base if self._base is not None else self,
                        pinned=self._pinned)
                    _strided_scatter(_fsrc, _fdst)
                else:
                    _strided_scatter(src, self)
                return self
            DeviceAllocator.memcpy(self.data_ptr(), src.data_ptr(), self._nbytes)
        return self

    def item(self) -> float:
        """Read single scalar from GPU via cudaMemcpy D2H."""
        assert self._numel == 1, f"item() requires 1 element, got {self._numel}"
        esz = dtype_size(self._dtype)
        buf = (ctypes.c_char * esz)()
        DeviceAllocator.memcpy(ctypes.addressof(buf), self.data_ptr(), esz, kind=2)

        if self._dtype == NBXDtype.float32:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))[0]
        elif self._dtype == NBXDtype.float64:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_double))[0]
        elif self._dtype == NBXDtype.int32:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int32))[0]
        elif self._dtype == NBXDtype.int64:
            return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int64))[0]
        elif self._dtype == NBXDtype.float16:
            raw = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint16))[0]
            return struct.unpack('e', struct.pack('H', raw))[0]
        elif self._dtype == NBXDtype.bool_:
            return bool(ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))[0])
        return 0.0

    # ========================================================================
    # DTYPE CONVERSION
    # ========================================================================

    def to(self, *args, **kwargs) -> 'NBXTensor':
        target = None
        if args:
            a = args[0]
            if isinstance(a, NBXDtype):
                target = a
            elif isinstance(a, str):
                # Distinguish device string ("cuda:0") from dtype string ("float32")
                if 'cuda' in a or 'cpu' in a:
                    return self  # .to(device) — already on device
                target = parse_dtype(a)
            elif isinstance(a, NBXTensor):
                return self
            else:
                # Check if it's a device-like object (has .type attribute = "cuda"/"cpu")
                s = str(a)
                if 'cuda' in s or 'cpu' in s or 'device' in s:
                    return self  # .to(device) — already on device
                target = parse_dtype(s)
        if 'dtype' in kwargs:
            d = kwargs['dtype']
            target = d if isinstance(d, NBXDtype) else parse_dtype(str(d))
        if target is None or target == self._dtype:
            return self
        # Complex tensors must never be downcast to a REAL dtype via .to():
        # PyTorch itself raises on complex->real .to() (real conversions go via
        # .real / view_as_real). When it appears here it is always a spurious
        # compute-dtype downcast of a complex tensor (e.g. the RoPE frequency
        # table, stored complex128) by the AMP/Prism dtype pipeline — which would
        # reinterpret the interleaved real/imag bytes as a single real array and
        # poison the value with NaN. Preserve the complex tensor. complex<->complex
        # casts (complex128<->complex64) still proceed below.
        if self.is_complex() and target not in _COMPLEX_DTYPES:
            return self
        # A float64 element on a backend whose kernels have none is not cast, it is misread:
        # refused by name (`backend_has_fp64`, asked only when fp64 is in play). The caller places the array at a dtype the
        # backend has (a float64 formula result is cast on the host first).
        if (self._dtype in (NBXDtype.float64, NBXDtype.complex128)
                or target in (NBXDtype.float64, NBXDtype.complex128)) and not _caps.backend_has_fp64():
            raise RuntimeError(
                f"ZERO FALLBACK: a cast from {self._dtype.name} to {target.name} on the "
                f"'{_rt._detect_gpu_backend()}' backend, whose kernels have no float64: its lowering "
                f"narrows double in silence and the result would be garbage, not a cast "
                f"(2026-09-28: 1 139 of 143 360 bits right on the Sana 4K positional embedding). "
                f"Place the array at a dtype this backend has; a float64 formula result is cast "
                f"on the host first.")
        # Cast via Triton copy kernel (tl.store auto-converts dtype).
        # Downcasts from wider floats to float16 use the PROTECTED
        # conversion (clamp finite values to ±65504) — the triton mirror
        # of the compiled DtypeEngine contract; raw tl.store overflows
        # bf16/fp32-range sentinels (finfo.min mask fills) to ±inf.
        _sat = _saturating_cast(self._dtype, target)
        new = NBXTensor.empty(self._shape, target, f"cuda:{self._device_idx}")
        n = self._numel
        if n > 0 and not self.is_contiguous() and not self.is_complex():
            # a strided source is read in place and cast on the store —
            # one launch instead of a contiguous copy and a cast of it
            _copy_nd(self, new)
            return new
        if n > 0:
            from nbxtensor.ops.copy_op import copy_kernel
            import triton
            src = self.contiguous()
            _rt._set_device(src)
            _m = _ew_meta(n)                     # the certified tile of the count's class (ops/elementwise_launch.py)
            copy_kernel[(triton.cdiv(n, _m['BLOCK_SIZE']),)](src, new, n,
                                                 BLOCK_SIZE=_m['BLOCK_SIZE'], num_warps=_m['num_warps'],
                                                 SATURATE_F16=_sat)
        return new

    def float(self) -> 'NBXTensor':
        return self.to(NBXDtype.float32)

    def half(self) -> 'NBXTensor':
        return self.to(NBXDtype.float16)

    def bfloat16(self) -> 'NBXTensor':
        return self.to(NBXDtype.bfloat16)

    def int(self) -> 'NBXTensor':
        return self.to(NBXDtype.int32)

    def long(self) -> 'NBXTensor':
        return self.to(NBXDtype.int64)

    def type_as(self, other) -> 'NBXTensor':
        target = other._dtype if isinstance(other, NBXTensor) else parse_dtype(str(other.dtype))
        return self.to(target)

    # ========================================================================
    # CREATION HELPERS
    # ========================================================================

    def new_zeros(self, size, **kwargs) -> 'NBXTensor':
        return NBXTensor.zeros(tuple(size) if isinstance(size, list) else size,
                               kwargs.get('dtype', self._dtype),
                               kwargs.get('device', self._device))

    def new_ones(self, size, **kwargs) -> 'NBXTensor':
        # Must actually FILL ones — the former `NBXTensor.empty` returned
        # uninitialized memory (a leftover TODO). The Allegro/T5 extended
        # attention mask opens with `aten::new_ones([]) & (positions >= 0)`;
        # the uninitialized 0-dim byte read as 0 -> False -> the AND zeroed the
        # whole [1,1,S,1] q-side -> extended mask = -inf on ALL keys -> every
        # T5 self-attention row went UNIFORM (softmax of a constant row), so
        # positive-prompt embeddings washed toward the pad-token mean (cond
        # branch corr 0.9288 vs the sequential oracle while the empty-prompt
        # uncond branch stayed clean).
        return NBXTensor.ones(tuple(size) if isinstance(size, list) else size,
                              kwargs.get('dtype', self._dtype),
                              kwargs.get('device', self._device))

    def new_empty(self, size, **kwargs) -> 'NBXTensor':
        return NBXTensor.empty(tuple(size) if isinstance(size, list) else size,
                               kwargs.get('dtype', self._dtype),
                               kwargs.get('device', self._device))

    # ========================================================================
    # STATIC ASSEMBLY — cat, stack (require data copy)
    # ========================================================================

    @staticmethod
    def cat(tensors: list, dim: int = 0) -> 'NBXTensor':
        # Filter out 0-dim scalars and empty tensors (Gemma2 attention pattern)
        valid = [t for t in tensors if t.ndim > 0 and t._numel > 0]
        if not valid:
            return tensors[0] if tensors else NBXTensor.empty((0,), NBXDtype.float32, 'cuda')
        if len(valid) == 1:
            return valid[0]
        tensors = valid
        # Complex cat: the copy kernel concatenates the interleaved [re,im] float
        # storage as if it were flat floats. For dim==0 that is correct (whole
        # element-blocks stay contiguous), but for any inner dim it splits the
        # boundary mid-pair, pairing re of one operand with im of the next
        # (the Wan rotary freqs concat per-axis along the last dim). Route via
        # the real view: cat on the same element dim with the [.,2] pair axis
        # appended (untouched), then reinterpret as complex.
        if valid[0].is_complex():
            ndc = valid[0].ndim
            dd = dim % ndc
            real_parts = [t.view_as_real() for t in valid]
            return NBXTensor.cat(real_parts, dd).view_as_complex()
        dim = dim % tensors[0].ndim
        # Contiguous-guard (manifesto pattern): the cat copy kernels
        # flat-index their sources; a strided view (e.g. the MLA rope
        # de-interleave slices x[..., 0::2]) fed directly reads wrong
        # addresses — layout-dependent illegal access at runtime.
        # Contiguous inputs short-circuit at zero cost.
        tensors = [t.contiguous() for t in tensors]
        # Align dtypes — cat_copy_kernel requires all inputs same type
        target_dtype = tensors[0]._dtype
        tensors = [t.to(target_dtype) if t._dtype != target_dtype else t for t in tensors]
        # Align devices — all tensors must be on same GPU
        target_dev = tensors[0]._device_idx
        aligned = []
        for t in tensors:
            if hasattr(t, '_device_idx') and t._device_idx != target_dev:
                src = t.contiguous()
                dst = NBXTensor.empty(src._shape, src._dtype, f"cuda:{target_dev}")
                DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), src._nbytes, kind=3)
                aligned.append(dst)
            else:
                aligned.append(t)
        tensors = aligned
        out_shape = list(tensors[0]._shape)
        out_shape[dim] = sum(t._shape[dim] for t in tensors)
        # Allocate on same device as first input tensor
        DeviceAllocator.set_device(tensors[0]._device_idx)
        out = NBXTensor.empty(tuple(out_shape), target_dtype, tensors[0]._device)

        if dim == 0:
            # Simple case: concat on first dim → contiguous slices
            offset = 0
            for t in tensors:
                tc = t.contiguous()
                if tc._nbytes > 0:
                    DeviceAllocator.memcpy(
                        out._data_ptr + offset * _DTYPE_SIZES[out._dtype],
                        tc.data_ptr(), tc._nbytes)
                offset += tc._numel
        else:
            # General case: use cat_copy_kernel which handles dim offsets
            import triton
            from nbxtensor.ops.cat_op import cat_copy_kernel_4

            dim_size_out = out_shape[dim]
            # Product of dims after cat dim
            dim_prod_post = 1
            for d in range(dim + 1, len(out_shape)):
                dim_prod_post *= out_shape[d]

            # Process tensors in groups of 4 (kernel handles up to 4)
            dim_offset = 0
            for batch_start in range(0, len(tensors), 4):
                batch = tensors[batch_start:batch_start + 4]
                # Pad to 4 with dummy (zero-element) tensors
                contig = [t.contiguous() for t in batch]
                while len(contig) < 4:
                    contig.append(out)  # dummy, total_elements=0

                sizes = [c._shape[dim] if i < len(batch) else 0 for i, c in enumerate(contig)]
                totals = [c._numel if i < len(batch) else 0 for i, c in enumerate(contig)]
                offsets = []
                off = dim_offset
                for s in sizes:
                    offsets.append(off)
                    off += s

                max_total = max(totals)
                if max_total == 0:
                    continue
                _m = _ew_meta(max_total)                 # the certified tile of the count's class (ops/elementwise_launch.py)
                BLOCK = _m['BLOCK_SIZE']
                grid = (triton.cdiv(max_total, BLOCK), len(batch))
                _rt._set_device(out)
                cat_copy_kernel_4[grid](
                    out,
                    contig[0], contig[1], contig[2], contig[3],
                    sizes[0], sizes[1], sizes[2], sizes[3],
                    dim_size_out, dim_prod_post,
                    offsets[0], offsets[1], offsets[2], offsets[3],
                    totals[0], totals[1], totals[2], totals[3],
                    BLOCK_X=BLOCK, num_warps=_m['num_warps'],
                )
                dim_offset = off

        return out

    @staticmethod
    def stack(tensors: list, dim: int = 0) -> 'NBXTensor':
        return NBXTensor.cat([t.unsqueeze(dim) for t in tensors], dim=dim)

    # ========================================================================
    # REPR
    # ========================================================================

    # ========================================================================
    # PYTHON OPERATORS — delegate to kernel wrappers
    # ========================================================================

    def __add__(self, other):
        from nbxtensor.ops.elementwise_wrappers import add
        return add(self, other)

    def __radd__(self, other):
        from nbxtensor.ops.elementwise_wrappers import add
        return add(other, self)

    def __sub__(self, other):
        from nbxtensor.ops.elementwise_wrappers import sub
        return sub(self, other)

    def __rsub__(self, other):
        from nbxtensor.ops.elementwise_wrappers import rsub
        return rsub(self, other)

    def __mul__(self, other):
        from nbxtensor.ops.elementwise_wrappers import mul
        return mul(self, other)

    def __rmul__(self, other):
        from nbxtensor.ops.elementwise_wrappers import mul
        return mul(other, self)

    def __truediv__(self, other):
        from nbxtensor.ops.elementwise_wrappers import div
        return div(self, other)

    def __neg__(self):
        from nbxtensor.ops.elementwise_wrappers import neg
        return neg(self)

    def __repr__(self) -> str:
        return (f"NBXTensor(shape={self._shape}, dtype={self._dtype.name}, "
                f"device={self._device}, ptr=0x{self._data_ptr:x})")
