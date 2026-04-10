"""Triton Dispatch — ONE table for ALL ops.

Every op goes through this table: compute (GPU kernel) + metadata (CPU stride math).
If an op is not here → CRASH. No fallback, no second path.

Compute wrappers: wrappers.py. Metadata: NBXTensor methods.
Dependencies: wrappers.py, NBXTensor. Used exclusively in --triton mode.
"""

from typing import Optional, Callable, Dict
import triton

_OP_MAP: Optional[Dict[str, Callable]] = None


_EW_BLOCK = 1024


def _1d_grid_fixed(n):
    """Grid for element-wise kernels. Fixed BLOCK_SIZE=1024."""
    return (triton.cdiv(n, _EW_BLOCK),)


def _tensor_to_int(val):
    """Convert tensor-like to Python int if possible."""
    if isinstance(val, (int, float)):
        return int(val)
    if hasattr(val, 'item'):
        if hasattr(val, 'ndim') and val.ndim == 0:
            return int(val.item())
        if hasattr(val, 'numel') and val.numel() == 1:
            return int(val.item())
    return val


def _tensor_to_int_or_none(val):
    if val is None:
        return None
    return _tensor_to_int(val)


# ============================================================================
# METADATA OPS — pure Python, NBXTensor methods, CPU stride math
# ============================================================================

def _meta_view(x, shape):
    if isinstance(shape, (list, tuple)):
        return x.view(*shape)
    return x.view(shape)

def _meta_reshape(x, shape):
    if isinstance(shape, (list, tuple)):
        return x.reshape(*shape)
    return x.reshape(shape)

def _meta_flatten(x, start_dim=0, end_dim=-1):
    return x.flatten(start_dim, end_dim)

def _meta_squeeze(x, dim=None):
    return x.squeeze(dim) if dim is not None else x.squeeze()

def _meta_unsqueeze(x, dim):
    return x.unsqueeze(_tensor_to_int(dim))

def _meta_permute(x, dims):
    return x.permute(*dims) if isinstance(dims, (list, tuple)) else x.permute(dims)

def _meta_transpose(x, dim0, dim1):
    return x.transpose(_tensor_to_int(dim0), _tensor_to_int(dim1))

def _meta_t(x):
    return x.t()

def _meta_contiguous(x, *args, **kwargs):
    return x.contiguous()

def _meta_expand(x, size):
    return x.expand(*size) if isinstance(size, (list, tuple)) else x.expand(size)

def _meta_expand_as(x, other):
    return x.expand_as(other)

def _meta_narrow(x, dim, start, length):
    return x.narrow(_tensor_to_int(dim), _tensor_to_int(start), _tensor_to_int(length))

def _meta_slice(x, dim, start=None, end=None, step=1):
    dim = int(_force_int(dim))
    start = int(_force_int(start)) if start is not None else 0
    dim_size = x.size(dim) if callable(getattr(x, 'size', None)) else x.shape[dim]
    end = int(_force_int(end)) if end is not None else dim_size
    if end > dim_size:
        end = dim_size
    if start < 0:
        start = max(0, dim_size + start)
    if end < 0:
        end = max(0, dim_size + end)
    length = max(0, end - start)
    return x.narrow(dim, start, length)


def _force_int(val):
    """Force ANY value to Python int. Handles NBXTensor, scalars, etc."""
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if hasattr(val, 'item'):
        return int(val.item())
    return int(val)

def _meta_select(x, dim, index):
    return x.select(_tensor_to_int(dim), _tensor_to_int(index))

def _meta_clone(x, *args, **kwargs):
    return x.clone()

def _meta_detach(x):
    return x.detach()

def _meta_alias(x):
    return x

def _meta_fill(x, value):
    return x.fill_(value)

def _meta_as_strided(x, size, stride, storage_offset=None):
    return x.as_strided(size, stride, storage_offset)

def _meta_view_as(x, other):
    return x.view_as(other)

def _meta_reshape_as(x, other):
    return x.reshape_as(other)

def _meta_unfold(x, dimension, size, step):
    return x.unfold(dimension, size, step)

def _meta_unbind(x, dim=0):
    return x.unbind(dim)

def _meta_select_item(x):
    return x.item()

def _meta_copy(x, src, *args, **kwargs):
    return x.copy_(src)

def _meta_equal(x, y):
    # Scalar comparison — returns bool
    return x.numel() == y.numel()  # simplified

def _meta_to(x, *args, **kwargs):
    return x.to(*args, **kwargs)

def _meta_type_as(x, other):
    return x.type_as(other)

def _meta_repeat(x, *repeats):
    if len(repeats) == 1 and isinstance(repeats[0], (list, tuple)):
        return x.repeat(*repeats[0])
    return x.repeat(*repeats)

def _meta_movedim(x, src, dst):
    return x.movedim(src, dst)


def _meta_index(x, indices):
    """aten::index — advanced indexing with list of optional index tensors.

    For single index on dim 0: x[idx] where idx can be multi-dimensional.
    Output shape = idx.shape + x.shape[1:]
    Uses index_select internally, then reshapes to match expected output.
    """
    if isinstance(indices, (list, tuple)):
        for dim, idx in enumerate(indices):
            if idx is not None:
                from neurobrix.kernels import wrappers as w
                # Flatten idx, index_select, then reshape to idx.shape + remaining
                orig_idx_shape = idx.shape
                flat_idx = idx.reshape(-1) if idx.ndim > 1 else idx
                selected = w.index_select_wrapper(x, dim, flat_idx)
                # Reshape: flatten selected along indexed dim to match idx shape
                if idx.ndim > 1:
                    out_shape = list(orig_idx_shape) + list(x.shape[dim+1:])
                    selected = selected.reshape(*out_shape)
                return selected
        return x
    from neurobrix.kernels import wrappers as w
    return w.index_select_wrapper(x, 0, indices)


# ============================================================================
# CREATION OPS — ATen signatures, Triton fill kernel, NBXTensor allocation
#
# Reference: FlagGems ops/{ones,zeros,full,arange}.py
# Signatures match aten::ones, aten::zeros, etc.
# ============================================================================

def _extract_creation_args(args, kwargs):
    """Extract (size, dtype, device) from ATen creation op args.

    ATen passes: (size_list, *, dtype=..., layout=..., device=..., pin_memory=...)
    Compiled sequence may pass size as positional or keyword.
    """
    size = args[0] if args else kwargs.get('size', ())
    if isinstance(size, (int, float)):
        size = (int(size),)
    dtype = kwargs.get('dtype')
    device = kwargs.get('device', 'cuda')
    return size, dtype, device


def _create_ones(*args, **kwargs):
    """aten::ones — fill with 1.0 via Triton fill kernel."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.fill_op import fill_kernel
    size, dtype, device = _extract_creation_args(args, kwargs)
    out = NBXTensor.empty(size, dtype=dtype, device=device or 'cuda')
    n = out.numel()
    if n > 0:
        fill_kernel[_1d_grid_fixed(n)](out, 1.0, n, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_zeros(*args, **kwargs):
    """aten::zeros — cudaMemset zero."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    size, dtype, device = _extract_creation_args(args, kwargs)
    return NBXTensor.zeros(size, dtype=dtype, device=device or 'cuda')


def _create_empty(*args, **kwargs):
    """aten::empty — uninitialized allocation."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    size, dtype, device = _extract_creation_args(args, kwargs)
    return NBXTensor.empty(size, dtype=dtype, device=device or 'cuda')


def _create_full(*args, **kwargs):
    """aten::full — fill with arbitrary value via Triton fill kernel."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.fill_op import fill_kernel
    size = args[0] if args else kwargs.get('size', ())
    fill_value = args[1] if len(args) > 1 else kwargs.get('fill_value', 0)
    dtype = kwargs.get('dtype')
    device = kwargs.get('device', 'cuda')
    out = NBXTensor.empty(size, dtype=dtype, device=device or 'cuda')
    n = out.numel()
    if n > 0:
        fill_kernel[_1d_grid_fixed(n)](out, float(fill_value), n, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_scalar_tensor(value=0, **kwargs):
    """aten::scalar_tensor — single-element tensor."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.fill_op import fill_kernel
    dtype = kwargs.get('dtype')
    device = kwargs.get('device', 'cuda')
    out = NBXTensor.empty((1,), dtype=dtype, device=device or 'cuda')
    fill_kernel[(1,)](out, float(value) if value is not None else 0.0, 1, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_arange(start_or_end, end=None, step=1, *, dtype=None,
                   layout=None, device=None, pin_memory=None, **kwargs):
    """aten::arange — sequential values via Triton kernel.

    Reference: FlagGems ops/arange.py — uses triton arange_func kernel.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.arange_op import arange_kernel
    import math
    if end is None:
        start, end = 0, start_or_end
    else:
        start = start_or_end
    start = float(start)
    end = float(end)
    step = float(step)
    if step == 0:
        raise ValueError("step must be non-zero")
    size = max(0, math.ceil((end - start) / step))
    if dtype is None:
        from neurobrix.kernels.nbx_tensor import NBXDtype
        dtype = NBXDtype.int64 if isinstance(start_or_end, int) and (end is None or isinstance(end, int)) else NBXDtype.float32
    out = NBXTensor.empty((size,), dtype=dtype, device=device or 'cuda')
    if size > 0:
        BLOCK_SIZE = 128
        grid = (triton.cdiv(size, BLOCK_SIZE),)
        arange_kernel[grid](out, start, step, size, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _create_zeros_like(x, *, dtype=None, layout=None, device=None,
                       pin_memory=None, **kwargs):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.zeros_like(x, dtype=dtype, device=device)


def _create_ones_like(x, *, dtype=None, layout=None, device=None,
                      pin_memory=None, **kwargs):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.fill_op import fill_kernel
    out = NBXTensor.empty_like(x, dtype=dtype, device=device)
    n = out.numel()
    if n > 0:
        fill_kernel[(triton.cdiv(n, _EW_BLOCK),)](out, 1.0, n, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_empty_like(x, *, dtype=None, layout=None, device=None,
                       pin_memory=None, **kwargs):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    return NBXTensor.empty_like(x, dtype=dtype, device=device)


def _create_full_like(x, fill_value, *, dtype=None, layout=None, device=None,
                      pin_memory=None, **kwargs):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.fill_op import fill_kernel
    out = NBXTensor.empty_like(x, dtype=dtype, device=device)
    n = out.numel()
    if n > 0:
        fill_kernel[(triton.cdiv(n, _EW_BLOCK),)](out, float(fill_value), n, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_linspace(start, end, steps, *, dtype=None, layout=None,
                     device=None, pin_memory=None, **kwargs):
    """aten::linspace — evenly spaced values."""
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from neurobrix.kernels.ops.fill_op import fill_kernel
    steps = int(steps)
    out = NBXTensor.empty((steps,), dtype=dtype or NBXDtype.float32, device=device or 'cuda')
    # TODO: proper Triton linspace kernel
    return out


def _build_creation_ops():
    """Creation ops with proper ATen signatures."""
    from neurobrix.kernels.nbx_tensor import NBXTensor

    return {
        "zeros": _create_zeros,
        "ones": _create_ones,
        "empty": _create_empty,
        "full": _create_full,
        "scalar_tensor": _create_scalar_tensor,
        "zeros_like": _create_zeros_like,
        "ones_like": _create_ones_like,
        "empty_like": _create_empty_like,
        "full_like": _create_full_like,
        "new_zeros": lambda x, size, **kw: x.new_zeros(size, **kw),
        "new_ones": lambda x, size, **kw: x.new_ones(size, **kw),
        "new_empty": lambda x, size, **kw: x.new_empty(size, **kw),
        "arange": _create_arange,
        "linspace": _create_linspace,
        # Cat/Stack
        "cat": lambda tensors, dim=0: NBXTensor.cat(tensors, dim),
        "stack": lambda tensors, dim=0: NBXTensor.stack(tensors, dim),
        "concat": lambda tensors, dim=0: NBXTensor.cat(tensors, dim),
    }


# ============================================================================
# BUILD COMPLETE OP MAP
# ============================================================================

def _build_op_map() -> Dict[str, Callable]:
    """Build the COMPLETE aten_op → function mapping.

    ONE table. ALL ops. Compute + Metadata + Creation.
    No second path. No fallback. Missing = crash.
    """
    from neurobrix.kernels import wrappers as w

    ops = {
        # =================================================================
        # COMPUTE OPS — Triton GPU kernels (wrappers.py)
        # =================================================================

        # --- Activations ---
        "relu": w.relu,
        "silu": w.silu,
        "gelu": w.gelu,
        "sigmoid": w.sigmoid_wrapper,
        "tanh": w.tanh_wrapper,
        "hardsigmoid": w.hardsigmoid,
        "hardswish": w.hardswish,
        "leaky_relu": w.leaky_relu,
        "elu": w.elu,
        "mish": w.mish,
        "selu": w.selu_wrapper,

        # --- Unary element-wise ---
        "neg": w.neg,
        "exp": w.exp,
        "sin": w.sin,
        "cos": w.cos,
        "rsqrt": w.rsqrt,
        "sqrt": w.sqrt_wrapper,
        "abs": w.abs_wrapper,
        "log": w.log_wrapper,
        "reciprocal": w.reciprocal,
        "pow": w.pow_wrapper,
        "clamp": w.clamp,
        "erf": w.erf,

        # --- Binary element-wise ---
        "add": w.add,
        "mul": w.mul,
        "div": w.div,
        "sub": w.sub,
        "rsub": w.rsub,
        "where": w.where_wrapper,
        "maximum": w.maximum_wrapper,
        "minimum": w.minimum_wrapper,
        "masked_fill": w.masked_fill,

        # --- Comparisons ---
        "gt": w.gt,
        "ge": w.ge,
        "lt": w.lt,
        "eq": w.eq,
        "ne": w.ne,
        "le": w.le,

        # --- Normalization ---
        "native_layer_norm": w.native_layer_norm,
        "rms_norm": w.rms_norm,
        "native_group_norm": w.group_norm_wrapper,
        "group_norm": w.group_norm_wrapper,
        "cudnn_batch_norm": w.batch_norm_wrapper,
        "native_batch_norm": w.batch_norm_wrapper,
        "batch_norm": w.batch_norm_wrapper,

        # --- Softmax ---
        "softmax": w.softmax,
        "_softmax": w.softmax,
        "log_softmax": w.log_softmax,
        "_log_softmax": w.log_softmax,

        # --- Matmul ---
        "mm": w.mm,
        "bmm": w.bmm,
        "addmm": w.addmm,
        "matmul": w.matmul_wrapper,

        # --- Embedding ---
        "embedding": w.embedding,

        # --- Reductions ---
        "mean": w.mean_wrapper,
        "sum": w.sum_wrapper,
        "amax": w.amax_wrapper,
        "argmax": w.argmax_wrapper,
        "argmin": w.argmin_wrapper,
        "prod": w.prod_wrapper,
        "min": w.min_wrapper,
        "max": w.max_wrapper,
        "std": w.std_wrapper,
        "var": w.var_wrapper,

        # --- GLU ---
        "glu": w.glu_wrapper,

        # --- Spatial ---
        "upsample_nearest1d": w.upsample_nearest1d_wrapper,
        "upsample_nearest2d": w.upsample_nearest2d_wrapper,
        "upsample_nearest3d": w.upsample_nearest3d_wrapper,
        "upsample_bilinear2d": w.upsample_bilinear2d_wrapper,
        "upsample_linear1d": w.upsample_linear1d_wrapper,
        "interpolate": w.interpolate_wrapper,
        "adaptive_avg_pool2d": w.adaptive_avg_pool2d_wrapper,
        "avg_pool2d": w.avg_pool2d_wrapper,
        "max_pool2d": w.max_pool2d_wrapper,

        # --- Conv ---
        "conv2d": w.conv2d_wrapper,
        "convolution": w.conv2d_wrapper,
        "conv1d": w.conv1d_wrapper,
        "depthwise_conv2d": w.conv_depthwise2d_wrapper,
        "conv_depthwise2d": w.conv_depthwise2d_wrapper,
        "conv_transpose2d": w.conv_depthwise2d_wrapper,

        # --- Indexing ---
        "index_select": w.index_select_wrapper,
        "gather": w.gather_wrapper,
        "scatter": w.scatter_wrapper,
        "scatter_add": w.scatter_add_wrapper,
        "scatter_reduce": w.scatter_reduce_wrapper,
        "index_add": w.index_add_wrapper,
        "bincount": w.bincount_wrapper,

        # --- Triangular ---
        "triu": w.triu_wrapper,
        "tril": w.tril_wrapper,

        # --- Cumulative ---
        "cumsum": w.cumsum_wrapper,

        # --- Dropout (inference = identity) ---
        "dropout": w.dropout_wrapper,
        "native_dropout": w.dropout_wrapper,

        # --- Misc element-wise ---
        "remainder": w.remainder_wrapper,
        "exp2": w.exp2_wrapper,
        "tan": w.tan_wrapper,
        "celu": w.celu_wrapper,
        "log_sigmoid": w.log_sigmoid_wrapper,
        "softplus": w.softplus_wrapper,
        "isfinite": w.isfinite_wrapper,
        "isinf": w.isinf_wrapper,
        "isnan": w.isnan_wrapper,
        "nan_to_num": w.nan_to_num_wrapper,
        "threshold": w.threshold_wrapper,
        "addcdiv": w.addcdiv_wrapper,
        "addcmul": w.addcmul_wrapper,
        "lerp": w.lerp_wrapper,

        # --- Logic/bitwise ---
        "all": w.all_wrapper,
        "any": w.any_wrapper,
        "bitwise_and": w.bitwise_and_wrapper,
        "bitwise_or": w.bitwise_or_wrapper,
        "bitwise_not": w.bitwise_not_wrapper,
        "logical_and": w.logical_and_wrapper,
        "logical_or": w.logical_or_wrapper,
        "logical_not": w.logical_not_wrapper,
        "logical_xor": w.logical_xor_wrapper,
        "bitwise_left_shift": w.bitwise_left_shift_wrapper,
        "bitwise_right_shift": w.bitwise_right_shift_wrapper,

        # --- Linear algebra ---
        "dot": w.dot_wrapper,
        "mv": w.mv_wrapper,
        "baddbmm": w.baddbmm_wrapper,
        "addmv": w.addmv_wrapper,
        "linalg_vector_norm": w.vector_norm_wrapper,
        "addr": w.addr_wrapper,
        "trace": w.trace_wrapper,

        # --- Loss ---
        "mse_loss": w.mse_loss_wrapper,
        "nll_loss": w.nll_loss_wrapper,
        "cross_entropy_loss": w.cross_entropy_wrapper,

        # --- Sort/topk ---
        "sort": w.sort_wrapper,
        "topk": w.topk_wrapper,
        "flip": w.flip_wrapper,

        # --- RoPE ---
        "rope": w.rope_wrapper,
        "pixel_shuffle": w.pixel_shuffle_wrapper,
        "pixel_unshuffle": w.pixel_unshuffle_wrapper,

        # --- Clamp ---
        "clamp_min": w.clamp_min_wrapper,

        # --- Weight norm ---
        "_weight_norm_interface": w.weight_norm_interface_wrapper,
        "repeat_interleave": w.repeat_interleave_self_int_wrapper,

        # --- RNG ---
        "rand": w.rand_wrapper,
        "randn": w.randn_wrapper,
        "rand_like": w.rand_like_wrapper,
        "randn_like": w.randn_like_wrapper,
        "normal": w.normal_wrapper,
        "uniform": w.uniform_wrapper,
        "bernoulli": w.bernoulli_wrapper,
        "multinomial": w.multinomial_wrapper,

        # --- Padding ---
        "constant_pad_nd": w.constant_pad_nd_wrapper,
        "pad": w.pad_wrapper,
        "reflection_pad1d": w.reflection_pad1d_wrapper,
        "reflection_pad2d": w.reflection_pad2d_wrapper,
        "reflection_pad3d": w.reflection_pad2d_wrapper,
        "replication_pad1d": w.replication_pad1d_wrapper,
        "replication_pad2d": w.replication_pad2d_wrapper,
        "replication_pad3d": w.replication_pad2d_wrapper,

        # --- FFT ---
        "_fft_r2c": w.fft_r2c_wrapper,
        "_fft_c2r": w.fft_c2r_wrapper,
        "_fft_c2c": w.fft_c2c_wrapper,
        "fft_rfft": w.fft_rfft_wrapper,
        "fft_irfft": w.fft_irfft_wrapper,

        # --- Complex ---
        "angle": w.angle_wrapper,

        # Attention — Dao-AILab Flash Attention v2 Triton kernel
        "scaled_dot_product_attention": w.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_attention": w.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_flash_attention": w.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_efficient_attention": w.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_cudnn_attention": w.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_flash_attention_for_cpu": w.scaled_dot_product_attention_wrapper,
        "complex": w.complex_wrapper,
        "fold": w.fold_wrapper,
        "unfold_backward": w.unfold_backward_wrapper,

        # =================================================================
        # METADATA OPS — CPU stride math, NBXTensor methods
        # =================================================================

        # View/reshape
        "view": _meta_view,
        "_unsafe_view": _meta_view,
        "reshape": _meta_reshape,
        "flatten": _meta_flatten,
        "squeeze": _meta_squeeze,
        "unsqueeze": _meta_unsqueeze,
        "permute": _meta_permute,
        "transpose": _meta_transpose,
        "t": _meta_t,
        "contiguous": _meta_contiguous,
        "expand": _meta_expand,
        "expand_as": _meta_expand_as,
        "repeat": _meta_repeat,
        "narrow": _meta_narrow,
        "slice": _meta_slice,
        "select": _meta_select,
        "unbind": _meta_unbind,
        "split_with_sizes": _meta_slice,  # simplified
        "view_as": _meta_view_as,
        "reshape_as": _meta_reshape_as,
        "as_strided": _meta_as_strided,
        "unfold": _meta_unfold,
        "movedim": _meta_movedim,

        # Memory
        "clone": _meta_clone,
        "copy": _meta_copy,
        "copy_": _meta_copy,
        "detach": _meta_detach,
        "alias": _meta_alias,
        "lift_fresh": _meta_alias,

        # Fill
        "fill": _meta_fill,
        "fill_": _meta_fill,

        # Type conversion
        "to": _meta_to,
        "_to_copy": _meta_to,
        "type_as": _meta_type_as,
        "float": lambda x: x.float(),
        "half": lambda x: x.half(),
        "bfloat16": lambda x: x.bfloat16(),
        "int": lambda x: x.int(),
        "long": lambda x: x.long(),

        # Math rounding ops — proper Triton kernels
        "floor": w.floor_wrapper,
        "ceil": w.ceil_wrapper,
        "round": w.round_wrapper,
        "trunc": w.trunc_wrapper,

        # Scalar
        "item": _meta_select_item,
        "_local_scalar_dense": _meta_select_item,
        "equal": _meta_equal,

        # Complex views
        "view_as_real": lambda x: x,  # reinterpret
        "view_as_complex": lambda x: x,  # reinterpret
        "real": lambda x: x,  # view
        "imag": lambda x: x,  # view

        # Queries (return Python scalars, not tensors)
        "size": lambda x, dim=None: x.size(dim) if dim is not None else x.size(),
        "dim": lambda x: x.dim(),
        "numel": lambda x: x.numel(),
        "stride": lambda x, dim=None: x.stride(dim) if dim is not None else x.stride(),
        "is_contiguous": lambda x: x.is_contiguous,

        # Broadcast
        "broadcast_tensors": lambda *t: t,

        # Advanced indexing — aten::index(tensor, [idx0, idx1, ...])
        "index": _meta_index,
        "index_put": lambda x, indices, values, acc=False: x,  # TODO
        "index_put_": lambda x, indices, values, acc=False: x,  # TODO

        # Split/chunk (use narrow internally)
        "split": lambda x, size, dim=0: tuple(x.narrow(dim, i * size, size) for i in range(x.shape[dim] // size)),
        "split_with_sizes": lambda x, sizes, dim=0: tuple(
            x.narrow(dim, sum(sizes[:i]), s) for i, s in enumerate(sizes)),
        "chunk": lambda x, chunks, dim=0: x.unbind(dim) if chunks == x.size(dim) else (x,),
        "tensor_split": lambda x, sections, dim=0: (x,),

        # Atleast
        "atleast_1d": lambda *t: t[0] if len(t) == 1 else t,
        "atleast_2d": lambda *t: t[0] if len(t) == 1 else t,

        # Set/pack (identity)
        "set": lambda x, *a, **k: x,
        "_pack_padded_sequence": lambda inp, lengths, bf=False: inp,
    }

    # Add creation ops
    ops.update(_build_creation_ops())

    return ops


# ============================================================================
# PUBLIC API — ONE function, ONE path
# ============================================================================

def dispatch(op_type: str) -> Optional[Callable]:
    """Look up handler for ANY aten op.

    Returns function if found, None if not.
    In --triton mode, None = CRASH (compiled_ops.py enforces this).

    ONE table. Compute + metadata + creation. No second path.
    """
    global _OP_MAP
    if _OP_MAP is None:
        _OP_MAP = _build_op_map()

    op_name = op_type.split("::")[-1] if "::" in op_type else op_type
    return _OP_MAP.get(op_name)


def has_kernel(op_type: str) -> bool:
    return dispatch(op_type) is not None


def list_kernels() -> list:
    global _OP_MAP
    if _OP_MAP is None:
        _OP_MAP = _build_op_map()
    return sorted(_OP_MAP.keys())
