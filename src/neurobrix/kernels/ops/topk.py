"""Top-K — pure @triton.jit kernels. Extracted from FlagGems.

Two-stage approach:
  topk_stage1_kernel: per-chunk top-k selection via iterative max/min
  topk_stage2_kernel: merge chunks via bitonic sort

Supporting JIT functions:
  _get_finfo_val: dtype min/max constants for masking
  _compare_and_swap: bitonic compare-and-swap with index tracking
  _bitonic_merge: single merge stage of bitonic sort
  argsort: full bitonic argsort
"""

import triton
import triton.language as tl
import triton.language.core as core

try:
    from triton.language.standard import _log2, zeros_like
except ImportError:
    pass


# --- Dtype limit constants (inline, no torch) ---

_MIN_FLOAT32_VAL = tl.constexpr(-3.4028235e+38)
_MAX_FLOAT32_VAL = tl.constexpr(3.4028235e+38)
_MIN_FLOAT16_VAL = tl.constexpr(-65504.0)
_MAX_FLOAT16_VAL = tl.constexpr(65504.0)
_MIN_BFLOAT16_VAL = tl.constexpr(-3.3895314e+38)
_MAX_BFLOAT16_VAL = tl.constexpr(3.3895314e+38)
_MIN_INT32_VAL = tl.constexpr(-2147483648)
_MAX_INT32_VAL = tl.constexpr(2147483647)


@triton.jit
def _get_finfo_val(
    dtype,
    return_max,
):
    """Saturating limit for `dtype`, as a Python float literal.

    The limit MUST match the dtype of the buffer being loaded, not the
    dtype the kernel computes in. `tl.load(..., other=v)` materialises
    `v` in the POINTER's element type, so an fp32 limit handed to an
    fp16 buffer overflows and lands as -inf. That is not a benign
    sentinel here: `_compare_and_swap` permutes with a masked sum, and
    `0 * -inf` is NaN, which loses every comparison and collapses the
    sort network (measured: pad slots surfacing into the top-k with
    their INT32_MIN sentinel index, whenever chunk_num*k was not a
    power of two).

    Literals rather than `tl.constexpr(torch.finfo(...).max)` — the
    upstream FlagGems form — because R33 forbids torch here and the
    constexpr wrapper mixes fp64 into the comparison. Dropping the
    dtype branching along with the torch call was the regression.
    """
    if dtype is tl.float16:
        if return_max:
            return 65504.0
        else:
            return -65504.0
    elif dtype is tl.bfloat16:
        if return_max:
            return 3.3895314e+38
        else:
            return -3.3895314e+38
    else:
        if return_max:
            return 3.4028235e+38
        else:
            return -3.4028235e+38


# --- Stage 1: per-chunk top-k ---


@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_chunk_idx = tl.program_id(1)
    chunk_num = tl.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
            chunk_select_idx = tl.argmax(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val)
            chunk_select_idx = tl.argmin(x_val, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        if DESCENDING:
            mask_v = tl.full(x_val.shape, float('-inf'), dtype=tl.float32)
        else:
            mask_v = tl.full(x_val.shape, float('inf'), dtype=tl.float32)
        x_val = tl.where(cols == chunk_select_idx, mask_v, x_val)


# --- Bitonic sort helpers ---


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # Slice left/right with stride 2**(n_dims - i - 1).
    #
    # SELECT the unwanted lane away rather than multiplying it by zero.
    # The upstream form is `tl.sum(y * (1 - mask), 1)`, which is exact
    # for finite values but yields NaN on any +-inf, because 0 * inf is
    # NaN. Selecting keeps the identity (-inf + 0 == -inf), so a
    # non-finite entry sorts to its extreme instead of poisoning the
    # comparison network. Callers do put -inf in the data: the sampler
    # masks rejected logits with -inf and then takes a top-k over the
    # result.
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(
        tl.sum(core.where(mask == 0, y, zeros_like(y)), 1)[:, None, :], shape
    ).to(x.dtype)
    right = core.broadcast_to(
        tl.sum(core.where(mask == 1, y, zeros_like(y)), 1)[:, None, :], shape
    ).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(
        tl.sum(core.where(mask == 0, y_idx, zeros_like(y_idx)), 1)[:, None, :], shape
    ).to(ids.dtype)
    right_idx = core.broadcast_to(
        tl.sum(core.where(mask == 1, y_idx, zeros_like(y_idx)), 1)[:, None, :], shape
    ).to(ids.dtype)
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # Compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """Bitonic merge stage.

    order: 0=ascending, 1=descending, 2=alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)

    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


# --- Stage 2: merge chunks via bitonic sort ---


@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    sort_dim: tl.constexpr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    mask_val = _get_finfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=mask_val).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)
