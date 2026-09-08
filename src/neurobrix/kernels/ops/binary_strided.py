"""Element-wise add, sub, mul and div over two operands read by their own strides —
pure @triton.jit, N-D scalable like `strided_copy`.

`_prepare_binary` used to broadcast a smaller operand by `expand(...).
contiguous()` — a full-size transient per call: a conv bias (32,) over a
448 × 448 image was 6.4 M elements written and read back per layer
(351 strided copies per Real-ESRGAN upscale, the copy census of
2026-09-07); an attention mask (S, S) broadcast over the heads the same.
These kernels read each operand through its strides (0 on a broadcast
dim), so the broadcast costs nothing and a transposed operand is read
in place. The arithmetic is the flat kernels' expression, verbatim
(`x + alpha * y`, `x * y`), so the bytes are those of the copying path.
"""

import triton
import triton.language as tl


@triton.jit
def add_strided_nd_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    alpha,
    shape_ptr,      # GPU pointer → NDIM int64 extents of the output
    x_stride_ptr,   # GPU pointer → NDIM int64 strides of x (0 on a broadcast dim)
    y_stride_ptr,   # GPU pointer → NDIM int64 strides of y
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    """out = x + alpha * y, x and y read by their strides, out contiguous."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets
    x_off = tl.zeros_like(offsets)
    y_off = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        xs = tl.load(x_stride_ptr + dim)
        ys = tl.load(y_stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        x_off = x_off + idx * xs
        y_off = y_off + idx * ys
    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)
    tl.store(output_ptr + offsets, x + alpha * y, mask=mask)


@triton.jit
def mul_strided_nd_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    shape_ptr,
    x_stride_ptr,
    y_stride_ptr,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    """out = x * y, x and y read by their strides, out contiguous."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets
    x_off = tl.zeros_like(offsets)
    y_off = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        xs = tl.load(x_stride_ptr + dim)
        ys = tl.load(y_stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        x_off = x_off + idx * xs
        y_off = y_off + idx * ys
    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)
    tl.store(output_ptr + offsets, x * y, mask=mask)


@triton.jit
def sub_strided_nd_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    alpha,
    shape_ptr,
    x_stride_ptr,
    y_stride_ptr,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    """out = x - alpha * y, x and y read by their strides, out contiguous."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets
    x_off = tl.zeros_like(offsets)
    y_off = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        xs = tl.load(x_stride_ptr + dim)
        ys = tl.load(y_stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        x_off = x_off + idx * xs
        y_off = y_off + idx * ys
    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)
    tl.store(output_ptr + offsets, x - alpha * y, mask=mask)


@triton.jit
def div_strided_nd_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    shape_ptr,
    x_stride_ptr,
    y_stride_ptr,
    BLOCK_SIZE: tl.constexpr,
    NDIM: tl.constexpr,
):
    """out = x / y, x and y read by their strides, out contiguous."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    remaining = offsets
    x_off = tl.zeros_like(offsets)
    y_off = tl.zeros_like(offsets)
    for i in tl.static_range(NDIM):
        dim = NDIM - 1 - i
        d = tl.load(shape_ptr + dim)
        xs = tl.load(x_stride_ptr + dim)
        ys = tl.load(y_stride_ptr + dim)
        idx = remaining % d
        remaining = remaining // d
        x_off = x_off + idx * xs
        y_off = y_off + idx * ys
    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)
    tl.store(output_ptr + offsets, x / y, mask=mask)
