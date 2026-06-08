"""Triton Dispatch — ONE table for ALL ops.

Every op goes through this table: compute (GPU kernel) + metadata (CPU stride math).
If an op is not here → CRASH. No fallback, no second path.

Compute wrappers: wrappers.py. Metadata: NBXTensor methods.
Dependencies: wrappers.py, NBXTensor. Used exclusively in --triton mode.
"""

from typing import Optional, Callable, Dict
import os
import triton

from .nbx_tensor import _set_device

# Diagnostic (default-off): log when the metadata expand/view runtime-resolution
# fixes actually fire. Used for R23 footprint screening (does a fix fire on a
# trace==runtime model like an LLM?). Zero cost when unset.
_META_RESOLVE_DEBUG = os.environ.get("NBX_DEBUG_META_RESOLVE")
def _meta_resolve_log(msg):
    if _META_RESOLVE_DEBUG:
        import sys
        sys.stderr.write(f"[META_RESOLVE] {msg}\n")

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

def _resolve_view_shape(x, shape):
    """Reconcile a trace-baked view/reshape shape with the runtime numel.

    NBXTensor.view does NOT validate numel — it re-strides blindly over the same
    data_ptr, so a baked trace dim that differs at runtime becomes a silent
    out-of-bounds view (not an error, unlike torch). The graph bakes trace-time
    shapes; a variable-length model's runtime numel can differ (Kokoro predictor
    view::9: input [1,640,14], baked shape [1,640,23]). Mirror of compiled_ops.py
    view_or_reshape's -1 inference: when the baked product mismatches the input
    numel, infer the single changed dim, preferring the axis whose inferred size
    matches the input's actual dim there. Bit-identical when product == numel
    (trace==runtime) → returns the shape unchanged. Returns a list.
    """
    shape = list(shape)
    if -1 in shape or len(shape) < 2:
        return shape  # explicit infer dim, or scalar/1D — view handles it
    numel = 1
    for d in x.shape:
        numel *= d
    prod = 1
    for d in shape:
        prod *= d
    if prod == numel:
        return shape  # exact (trace==runtime) — no change, byte-identical
    # numel mismatch: collect positions whose -1 inference divides evenly
    candidates = []
    for i in range(len(shape)):
        rest = 1
        for j, d in enumerate(shape):
            if j != i:
                rest *= d
        if rest > 0 and numel % rest == 0:
            candidates.append((i, numel // rest))
    if not candidates:
        return shape  # unresolvable (≥2 dims changed) — leave to caller
    _orig = tuple(shape)
    # prefer the axis whose inferred size matches the input's actual dim
    for i, inferred in candidates:
        if i < len(x.shape) and inferred == x.shape[i]:
            shape[i] = inferred
            _meta_resolve_log(f"view infer: in={tuple(x.shape)} baked={_orig} -> {tuple(shape)} (axis {i}, match)")
            return shape
    i, inferred = candidates[0]
    shape[i] = inferred
    _meta_resolve_log(f"view infer: in={tuple(x.shape)} baked={_orig} -> {tuple(shape)} (axis {i}, first)")
    return shape


def _meta_view(x, shape):
    if isinstance(shape, (list, tuple)):
        return x.view(*_resolve_view_shape(x, shape))
    return x.view(shape)

def _meta_reshape(x, shape):
    if isinstance(shape, (list, tuple)):
        return x.reshape(*_resolve_view_shape(x, shape))
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
    # Multi-resolution fix — mirror of compiled_ops.py _make_expand. The graph
    # bakes trace-time expand sizes; at runtime a variable-length model's actual
    # input dim can differ (Kokoro predictor: d.T seq 23->14, alignment frames
    # 34->62). expand can only broadcast from 1, so when the input's actual dim is
    # non-1 and differs from the baked target, the actual dim is the only valid
    # size — use it. Bit-identical for trace==runtime models (actual == target →
    # no override); restores R30 parity with compiled.
    if isinstance(size, (list, tuple)) and len(size) == len(x.shape):
        new_size = list(size)
        for i, (actual, target) in enumerate(zip(x.shape, size)):
            if actual != 1 and actual != target and target != -1:
                new_size[i] = actual
        if _META_RESOLVE_DEBUG and new_size != list(size):
            _meta_resolve_log(f"expand override: in={tuple(x.shape)} baked={tuple(size)} -> {tuple(new_size)}")
        return x.expand(*new_size)
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
    step = int(_force_int(step)) if step is not None else 1
    if end > dim_size:
        end = dim_size
    if start < 0:
        start = max(0, dim_size + start)
    if end < 0:
        end = max(0, dim_size + end)
    length = max(0, end - start)
    narrowed = x.narrow(dim, start, length)
    if step == 1:
        return narrowed
    # Strided slice x[..., start:end:step]. narrow() only covers step==1; for a
    # step>1 slice (chatterbox s3gen CFM downsamples the mel by 2 via a `[::2]`)
    # build a step-strided view (size=ceil(length/step), stride[dim]*=step) and
    # materialise it — downstream flat-indexed kernels (split_with_sizes) need a
    # contiguous buffer (CLAUDE.md contiguous-guard). Without this the step was
    # silently dropped and the dim kept its full length (348 vs 174).
    new_size = list(narrowed.shape)
    new_stride = list(narrowed._strides)
    new_size[dim] = (length + step - 1) // step
    new_stride[dim] = narrowed._strides[dim] * step
    return narrowed.as_strided(new_size, new_stride, narrowed._offset).contiguous()


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
    # Functional aten::fill: output is a FRESH tensor shaped like x, all elements
    # = value (x supplies only shape/dtype/device — its data is irrelevant, and it
    # may be a non-contiguous view, e.g. a slice). Never mutate x in place: that
    # would corrupt other consumers of the input and (on a strided view) write the
    # wrong addresses under flat-indexed fill.
    from neurobrix.kernels.nbx_tensor import NBXTensor
    if isinstance(value, NBXTensor):
        value = value.item()
    if value is None:
        # Orphan scalar constant: an in-forward `mask[slice] = cnt` count baked
        # as a `param::constant` with NO data in the container, consumed by
        # aten::fill. The native compiled path materialises it as a 0-dim default
        # (bind_weights `[]`, commit c0a1445) and the triton-sequential resolver
        # mirrors that (graph_executor._resolve_sequential_arg) — but the triton
        # COMPILED arena leaves the slot None, so the value reaches fill as None.
        # Default to 0 here so both triton paths match the native 0-default (HAT
        # OCAB counter; the value is a trace-time artefact, not runtime data).
        value = 0
    out = NBXTensor.empty(x._shape, x._dtype, f"cuda:{x._device_idx}")
    return out.fill_(value)

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
    """aten::index — advanced indexing with a list of optional index tensors.

    Single index tensor on dim d: x[..., idx, ...] via index_select; output
    shape = x.shape[:d] + idx.shape + x.shape[d+1:].

    TWO+ index tensors = advanced ("diagonal"/joint) indexing: PyTorch selects
    elements jointly, out[k] = x[i0[k], i1[k], ...], NOT the outer product.
    The CLIP text pooler is exactly this: last_hidden_state[arange(B),
    input_ids.argmax(-1)] picks the EOS token per batch → [B, hidden]. The old
    loop returned after the FIRST index tensor, so it kept all 77 text tokens
    ([1,77,768]) instead of pooling to [1,768] — which then broadcast the 77
    dim through the FLUX/Flex temb (silu::3 [1,77,3072] → addmm [1,77,18432] →
    split(dim=1) yields 0 chunks → adaLN slice crash). Implement the joint
    select via a flat row-major index over the consecutive indexed dims.
    """
    from neurobrix.kernels import wrappers as w
    if not isinstance(indices, (list, tuple)):
        return w.index_select_wrapper(x, 0, indices)

    present = [(d, i) for d, i in enumerate(indices) if i is not None]
    if not present:
        return x

    if len(present) == 1:
        # Single advanced index — unchanged behaviour (byte-identical path).
        dim, idx = present[0]
        orig_idx_shape = idx.shape
        flat_idx = idx.reshape(-1) if idx.ndim > 1 else idx
        selected = w.index_select_wrapper(x, dim, flat_idx)
        if idx.ndim > 1:
            out_shape = list(orig_idx_shape) + list(x.shape[dim + 1:])
            selected = selected.reshape(*out_shape)
        return selected

    # --- Advanced indexing with >=2 index tensors (joint select) ---
    dims = [d for d, _ in present]
    if dims != list(range(dims[0], dims[0] + len(dims))):
        raise NotImplementedError(
            f"[--triton] aten::index advanced indexing requires consecutive "
            f"index dims; got {dims}")
    first = dims[0]
    idx_tensors = [i for _, i in present]
    base_shape = list(idx_tensors[0].shape)
    if any(list(t.shape) != base_shape for t in idx_tensors[1:]):
        # PyTorch broadcasts differing index shapes; the pooler/batched-gather
        # cases use identical shapes. Defer general broadcasting until a model
        # needs it (fail loud, never silently wrong).
        raise NotImplementedError(
            f"[--triton] aten::index advanced indexing with differing index "
            f"shapes not yet supported: {[t.shape for t in idx_tensors]}")
    # Row-major flat index across the consecutive indexed dims:
    # flat = ((i0)*n1 + i1)*n2 + i2 ...   (ni = x.shape[indexed dim])
    sizes = [x.shape[d] for d in dims]
    flats = [t.reshape(-1) for t in idx_tensors]
    flat = flats[0]
    for k in range(1, len(flats)):
        flat = flat * sizes[k] + flats[k]
    flat = flat.long()
    block = 1
    for s in sizes:
        block *= s
    collapsed = list(x.shape[:first]) + [block] + list(x.shape[first + len(dims):])
    xc = x.reshape(*collapsed)
    sel = w.index_select_wrapper(xc, first, flat)
    out_shape = list(x.shape[:first]) + base_shape + list(x.shape[first + len(dims):])
    return sel.reshape(*out_shape)


def _meta_sdpa_efficient(*args, **kwargs):
    """aten::_scaled_dot_product_efficient_attention / _flash_attention.

    These fused-backend SDPA variants have a DIFFERENT positional signature
    than plain ``scaled_dot_product_attention``: an extra ``compute_log_sumexp``
    bool sits at arg[4], shifting ``dropout_p``→[5], ``is_causal``→[6],
    ``scale``→[7]. The plain-SDPA wrapper expects
    ``(q, k, v, attn_mask, dropout_p, is_causal, scale)``, so a *direct*
    positional call mis-reads ``is_causal`` (from ``dropout_p``) and ``scale``
    (from ``is_causal``) — silently dropping the decoder's causal mask and
    using scale=1.0. Invisible at seq_len=1 (single-element softmax), it
    corrupts every seq_len>=2 forward (constant-token garbage on whisper).

    Remap explicitly. Mirrors ``triton_sequential`` (sequential.py:198) so the
    two triton modes stay symmetric (R30).
    """
    from neurobrix.kernels import wrappers as w
    q, k, v = args[0], args[1], args[2]
    attn_mask = args[3] if len(args) > 3 and args[3] is not None else None
    dropout_p = args[5] if len(args) > 5 else 0.0
    is_causal = args[6] if len(args) > 6 else False
    scale = kwargs.get("scale", args[7] if len(args) > 7 else None)
    if not isinstance(dropout_p, float):
        dropout_p = float(dropout_p)
    if not isinstance(is_causal, bool):
        is_causal = bool(is_causal)
    return w.scaled_dot_product_attention_wrapper(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale)


def _meta_weight_norm(v, g, dim=0, **kwargs):
    """aten::_weight_norm(v, g, dim) → w = v * g / ‖v‖, where ‖v‖ is the L2 norm
    over ALL dims except `dim` (keepdim). Pure-Triton meta-op via the existing
    mul/sum/sqrt/div wrappers (no dedicated kernel needed). Used by vocoders'
    weight-normalized convs (chatterbox s3gen, HiFi-GAN-style).
    """
    from neurobrix.kernels import wrappers as w
    nd = v.ndim
    d = dim % nd if nd else 0
    vsq = w.mul(v, v)
    nsq = vsq
    for rd in range(nd):
        if rd == d:
            continue
        nsq = w.sum_wrapper(nsq, dim=rd, keepdim=True)
    norm = w.sqrt_wrapper(nsq)
    # w = v * (g / norm); g and norm share the norm-shape (1s except dim d),
    # broadcast against the full-shape v.
    return w.mul(v, w.div(g, norm))


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
        _set_device(out)
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
        _set_device(out)
        fill_kernel[_1d_grid_fixed(n)](out, float(fill_value), n, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_scalar_tensor(value=0, **kwargs):
    """aten::scalar_tensor — single-element tensor."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.ops.fill_op import fill_kernel
    dtype = kwargs.get('dtype')
    device = kwargs.get('device', 'cuda')
    out = NBXTensor.empty((1,), dtype=dtype, device=device or 'cuda')
    _set_device(out)
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
        _set_device(out)
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
        _set_device(out)
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
        _set_device(out)
        fill_kernel[(triton.cdiv(n, _EW_BLOCK),)](out, float(fill_value), n, BLOCK_SIZE=_EW_BLOCK)
    return out


def _create_linspace(start, end, steps, *, dtype=None, layout=None,
                     device=None, pin_memory=None, **kwargs):
    """aten::linspace — evenly spaced values via Triton kernel.

    Reference: FlagGems ops/linspace.py — bidirectional kernel
    (`kernels/ops/linspace_op.py`), forward from start for the first
    half and backward from end for the second half so both endpoints
    are exact (matches torch.linspace semantics).

    steps == 1 degenerates to a single `start` value (torch returns
    [start], not [end]); the bidirectional formula would yield `end`
    for that lone element, so it is special-cased through fill_kernel.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from neurobrix.kernels.ops.linspace_op import linspace_kernel
    from neurobrix.kernels.ops.fill_op import fill_kernel
    steps = int(steps)
    start = float(start)
    end = float(end)
    out = NBXTensor.empty((steps,), dtype=dtype or NBXDtype.float32,
                          device=device or 'cuda')
    if steps <= 0:
        return out
    BLOCK_SIZE = 128
    grid = (triton.cdiv(steps, BLOCK_SIZE),)
    _set_device(out)
    if steps == 1:
        fill_kernel[grid](out, start, 1, BLOCK_SIZE=BLOCK_SIZE)
        return out
    step_size = (end - start) / (steps - 1)
    mid = steps // 2
    linspace_kernel[grid](out, out.stride(0), start, mid, end,
                          step_size, steps, BLOCK_SIZE=BLOCK_SIZE)
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

        # --- Recurrent (triton-pure assembly; self-manages dtype) ---
        "lstm": w.lstm_wrapper,

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
        "clip": w.clamp,  # aten::clip is an alias of aten::clamp
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
        "layer_norm": w.layer_norm_wrapper,
        "rms_norm": w.rms_norm,
        "swiglu_fused": w.swiglu_fused_wrapper,
        "rope_fused": w.rope_fused_wrapper,
        "native_group_norm": w.native_group_norm_wrapper,
        "group_norm": w.group_norm_wrapper,
        "cudnn_batch_norm": w.batch_norm_wrapper,
        "native_batch_norm": w.batch_norm_wrapper,
        "batch_norm": w.batch_norm_wrapper,

        # --- Softmax ---
        "softmax": w.softmax,
        "_softmax": w.softmax,
        "_safe_softmax": w.softmax,
        "log_softmax": w.log_softmax,
        "_log_softmax": w.log_softmax,

        # --- Matmul ---
        "mm": w.mm,
        "bmm": w.bmm,
        "addmm": w.addmm,
        "matmul": w.matmul_wrapper,
        "linear": w.linear_wrapper,
        "isin": w.isin_wrapper,
        "is_nonzero": w.is_nonzero_wrapper,

        # --- Embedding ---
        "embedding": w.embedding,

        # --- Reductions ---
        "mean": w.mean_wrapper,
        "sum": w.sum_wrapper,
        "norm": w.norm_wrapper,
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
        "conv_transpose1d": w.conv_transpose1d_wrapper,
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
        "_weight_norm": _meta_weight_norm,
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
        "roll": w.roll_wrapper,

        "pixel_shuffle": w.pixel_shuffle_wrapper,
        "pixel_unshuffle": w.pixel_unshuffle_wrapper,

        # --- Clamp ---
        "clamp_min": w.clamp_min_wrapper,

        # --- Weight norm ---
        "_weight_norm_interface": w.weight_norm_interface_wrapper,
        "repeat_interleave": w.repeat_interleave_wrapper,

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
        "stft": w.stft_wrapper,
        "istft": w.istft_wrapper,

        # --- Complex ---
        "angle": w.angle_wrapper,

        # Attention — Dao-AILab Flash Attention v2 Triton kernel
        "scaled_dot_product_attention": w.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_attention": w.scaled_dot_product_attention_wrapper,
        # Fused-backend variants have a shifted positional signature
        # (extra compute_log_sumexp / return_debug_mask arg). Route through the
        # remap shim so is_causal / scale read from the right slots — mirrors
        # triton_sequential (sequential.py:198) for R30 symmetry.
        # CAVEAT: _meta_sdpa_efficient reads is_causal from arg[6], correct for
        # efficient + cudnn. The flash variant actually carries is_causal at
        # arg[4] (no attn_bias slot), so a CAUSAL flash op would lose its mask
        # here. No flash-backend model is in the current sweep; if one appears,
        # give flash a dedicated remap that reads arg[4] (cf. the KV-cache path
        # kv_cache.py::intercept_flash, which already does).
        "_scaled_dot_product_flash_attention": _meta_sdpa_efficient,
        "_scaled_dot_product_efficient_attention": _meta_sdpa_efficient,
        "_scaled_dot_product_cudnn_attention": _meta_sdpa_efficient,
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
        "im2col": w.im2col_wrapper,
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

        # Complex views — real reinterprets (complex64 ↔ float[..,2]), see
        # NBXTensor.view_as_real/view_as_complex/.real/.imag.
        "view_as_real": lambda x: x.view_as_real(),
        "view_as_complex": lambda x: x.view_as_complex(),
        "real": lambda x: x.real,
        "imag": lambda x: x.imag,

        # Queries (return Python scalars, not tensors)
        "size": lambda x, dim=None: x.size(dim) if dim is not None else x.size(),
        "dim": lambda x: x.dim(),
        "numel": lambda x: x.numel(),
        "stride": lambda x, dim=None: x.stride(dim) if dim is not None else x.stride(),
        "is_contiguous": lambda x: x.is_contiguous,

        # Broadcast
        "broadcast_tensors": w.broadcast_tensors_wrapper,

        # Advanced indexing — aten::index(tensor, [idx0, idx1, ...])
        "index": _meta_index,
        "index_put": w.index_put_wrapper,
        "index_put_": w.index_put_wrapper,

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
