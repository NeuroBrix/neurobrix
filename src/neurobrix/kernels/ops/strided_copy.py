"""Strided ↔ contiguous copy kernels — N-D scalable, Triton-JIT.

Two operations:

* `strided_copy_kernel`    — read from a non-contiguous (strided) source,
                             write to a contiguous destination. Used by
                             `NBXTensor.clone` / `NBXTensor.contiguous`
                             whenever the source is a view (permute,
                             transpose, narrow, expand, ...).

* `strided_scatter_kernel` — read from a contiguous source, write to a
                             non-contiguous destination. Used by
                             `NBXTensor.__setitem__` (KV cache indexed
                             writes into a narrow view).

* `strided_copy_nd_kernel`  — read from a strided source, write to a
                             strided destination, converting the dtype on
                             the store with `copy_kernel`'s exact protected
                             fp16 rounding. One launch where `__setitem__`,
                             `copy_` and `.to()` used to take a contiguous
                             copy of the source, a cast of that copy and a
                             scatter of the cast (three launches and two
                             transients per KV-cache write, 2026-09-07).

Both kernels are parameterised by `NDIM: tl.constexpr`. Triton
specialises and caches one compiled kernel per distinct ndim
encountered at runtime: the per-dimension index decomposition loop is
unrolled at compile time via `tl.static_range(NDIM)`.

The shape and stride vectors are passed as GPU pointers to small int64
arrays (NDIM elements each). The kernel loads them inside the JIT body,
so callers can handle any tensor up to PyTorch's 25-D hard limit
without recompiling the code. A 6-D PixArt patchify clone, an 8-D
SANA-Video VAE reshape, and a hypothetical 10-12-D multi-view 4D-scene
permute all route through the same kernel text.

Why this matters: the previous version was hardcoded to 5 dims
(`d0..d4`, `s0..s4`). The wrapper silently dropped higher dims via
`[1] * (5 - ndim)` returning `[]` for ndim > 5. 6-D PixArt clones
produced out-of-bounds reads past the source allocation — silently
corrupted data (Alpha) or `cudaErrorIllegalAddress` (Sigma), and
8-D SANA-Video VAE clones had been unreachable entirely.
"""

import triton
import triton.language as tl


@triton.jit
def strided_copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    shape_ptr,     # GPU pointer → NDIM int64 shape values (source)
    stride_ptr,    # GPU pointer → NDIM int64 stride values (source)
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    """Copy strided src → contiguous dst. Dimensions are unrolled at
    compile time via the NDIM constexpr; shape/strides are loaded from
    GPU scratch buffers so the kernel text is identical for every
    ndim Triton specialises.

    Args:
        src_ptr:     Source tensor data pointer (may be non-contiguous).
        dst_ptr:     Destination tensor data pointer (contiguous).
        n_elements:  Total elements to copy (= src._numel).
        shape_ptr:   Pointer to NDIM int64 shape values (source shape).
        stride_ptr:  Pointer to NDIM int64 stride values (source strides).
        BLOCK_SIZE:  Elements per thread block (constexpr).
        NDIM:        Rank of the tensors (constexpr; triggers
                     specialisation + kernel cache entry).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose the flat offset into multi-dim indices, walking dims
    # from the innermost (NDIM-1) outward. At each step, `remaining %
    # d` gives the current-dim index and `remaining // d` feeds the
    # next outer dim. The src_offsets accumulator multiplies each
    # per-dim index by the source's stride for that dim — this is
    # what lets the kernel read from a non-contiguous layout while
    # writing contiguously.
    remaining = offsets
    src_offsets = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        s = tl.load(stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        src_offsets = src_offsets + idx * s

    vals = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def strided_scatter_kernel(
    src_ptr, dst_ptr,
    n_elements,
    shape_ptr,     # GPU pointer → NDIM int64 shape values (destination)
    stride_ptr,    # GPU pointer → NDIM int64 stride values (destination)
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    """Scatter contiguous src → strided dst. Inverse of
    `strided_copy_kernel`. Same N-D specialisation strategy.

    Args:
        src_ptr:     Source tensor data pointer (contiguous).
        dst_ptr:     Destination tensor data pointer (may be non-
                     contiguous — e.g. a narrow view into a KV cache).
        n_elements:  Total elements to scatter (= dst._numel).
        shape_ptr:   Pointer to NDIM int64 shape values (dst shape).
        stride_ptr:  Pointer to NDIM int64 stride values (dst strides).
        BLOCK_SIZE:  Elements per thread block.
        NDIM:        Rank of the tensors (constexpr).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    remaining = offsets
    dst_offsets = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        s = tl.load(stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        dst_offsets = dst_offsets + idx * s

    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


@triton.jit
def strided_copy_nd_kernel(
    src_ptr, dst_ptr,
    n_elements,
    shape_ptr,        # GPU pointer → NDIM int64 extents (shared by src and dst)
    src_stride_ptr,   # GPU pointer → NDIM int64 source strides (0 on a broadcast dim)
    dst_stride_ptr,   # GPU pointer → NDIM int64 destination strides
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
    SATURATE_F16: tl.constexpr = False,
):
    """Copy strided src → strided dst, casting on the store.

    The flat element index is decomposed once into the multi-index and
    each side's offset accumulates its own strides, so any pair of
    layouts of the same extents copies in one pass — a broadcast source
    (stride 0) included. SATURATE_F16 is the protected float16 downcast
    of `copy_kernel`, the same expression, so a cast through this kernel
    is bit-identical to a cast through `copy_kernel` after `contiguous()`.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    remaining = offsets
    src_offsets = tl.zeros_like(offsets)
    dst_offsets = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        ss = tl.load(src_stride_ptr + dim)
        ds = tl.load(dst_stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        src_offsets = src_offsets + idx * ss
        dst_offsets = dst_offsets + idx * ds

    vals = tl.load(src_ptr + src_offsets, mask=mask)
    if SATURATE_F16:
        f = vals.to(tl.float32)
        is_finite = (f == f) & (f != float("inf")) & (f != float("-inf"))
        clamped = tl.minimum(tl.maximum(f, -65504.0), 65504.0)
        vals = tl.where(is_finite, clamped, f)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)
