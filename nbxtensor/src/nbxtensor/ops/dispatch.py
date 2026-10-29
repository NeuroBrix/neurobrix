"""Triton Dispatch — ONE table for ALL ops.

Every op goes through this table: compute (GPU kernel) + metadata (CPU stride math).
If an op is not here → CRASH. No fallback, no second path.

Compute wrappers: wrappers.py. Metadata: NBXTensor methods.
Dependencies: wrappers.py, NBXTensor. Used exclusively in --triton mode.
"""
# The launch seam (R33, universal since 2026-09-05): every `kernel[grid](...)`
# of the house library goes through the NeuroBrix launcher (Triton compiles,
# the dispatch layer launches through a vendor driver of its own).
# NBX_LAUNCHER=triton keeps upstream's launcher for the differential arm.


from typing import Optional, Callable, Dict
import os
import triton

from nbxtensor.backend.runtime import _set_device
from nbxtensor.shape import view_strides

# Diagnostic (default-off): log when the metadata expand/view runtime-resolution
# fixes actually fire. Used for R23 footprint screening (does a fix fire on a
# trace==runtime model like an LLM?). Zero cost when unset.
_META_RESOLVE_DEBUG = os.environ.get("NBX_DEBUG_META_RESOLVE")
def _meta_resolve_log(msg):
    if _META_RESOLVE_DEBUG:
        import sys
        sys.stderr.write(f"[META_RESOLVE] {msg}\n")

_OP_MAP: Optional[Dict[str, Callable]] = None


def _ew(n):
    """An element-wise launch's certified tile and warps (`ops/elementwise_launch.ew_meta`)."""
    from nbxtensor.ops.elementwise_launch import ew_meta
    return ew_meta(n)


def _1d_grid_fixed(n):
    """Grid for element-wise kernels at the certified tile of the count's class."""
    return (triton.cdiv(n, _ew(n)['BLOCK_SIZE']),)


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

def _report_invented_shape(x, shape):
    """Report-only twin of the instrument in `metadata_ops._reshape`, at MODE 2's site.

    Step one of the owner's two-step decision (2026-09-24): know which containers a refusing
    reshape would reject before writing the refusal. The first instrument sat in the compiled
    op set, which a `--modes triton,triton-sequential` census never enters, so it could not
    fire (a Kokoro-82M census with it set never created the file). `NBX_RESHAPE_REPORT=<file>`
    appends one record per call whose target has no -1 and whose element count does not match
    the input: an INVENTED shape. A -1 target is an inference and is not recorded. A 1-D
    target is not exempt, because `NBXTensor.view` does not validate numel. The call has no
    op uid here, so a record carries shapes, not a uid. Default off, one env lookup per call,
    no behaviour change."""
    rep = os.environ.get("NBX_RESHAPE_REPORT")
    if not rep or -1 in shape:
        return
    numel = 1
    for d in x.shape:
        numel *= d
    target = 1
    for d in shape:
        target *= d
    if numel == target:
        return
    try:
        import json
        with open(rep, "a") as fh:
            json.dump({
                "model": os.environ.get("NBX_RESHAPE_REPORT_MODEL", ""),
                "site": "dispatch._resolve_view_shape",
                "op_uid": "",
                "in_shape": [int(d) for d in x.shape],
                "in_numel": int(numel),
                "target": [int(d) for d in shape],
                "target_numel": int(target),
                "ratio": (numel / target) if target else None,
            }, fh)
            fh.write("\n")
    except Exception:
        pass          # an instrument must never change the run it measures


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
    _report_invented_shape(x, shape)
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
    """The graph's aten.view: a VIEW whenever ATen's would be one (`nbxtensor.shape.view_strides`), a strided input included — the
    graph was traced under ATen, where this op copies nothing. `NBXTensor.view` copies a non-contiguous input, and
    mochi-1-preview's VAE held a 3 GB copy per permute -> view the graph never priced (8 GB rung, 2026-10-28)."""
    if isinstance(shape, (list, tuple)):
        shape = tuple(_resolve_view_shape(x, shape))
    else:
        shape = (shape,)
    if not x.is_contiguous():
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape = tuple(x.numel() // known if s == -1 else s for s in shape)
        strides = view_strides(tuple(x.shape), tuple(x.stride()), shape)
        if strides is not None:
            return x.as_strided(shape, tuple(strides))
    return x.view(*shape)

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
    # step>1 slice build a step-strided VIEW (size=ceil(length/step),
    # stride[dim]*=step) and return it as a view — NOT materialised. A slice is
    # a view in PyTorch/compiled (sequential_dispatcher maps aten::copy to
    # inputs[0].copy_(inputs[1]) so an in-place write through the slice reaches
    # the parent buffer). The interleaved-complex Wan RoPE relies on exactly
    # that: `buf[...,0::2] = real; buf[...,1::2] = imag` lowers to step-2 slice
    # + copy with the copy output orphaned — the real effect is mutation of the
    # empty_like buffer that permute reads. Materialising the slice here made
    # copy_ write into a throwaway contiguous tensor, leaving the buffer unwritten
    # → zero self-attn Q/K → washed-out VACE/Wan2.2 (2026-06-28). The
    # as_strided VIEW makes copy_ route through NBXTensor.copy_'s strided-dst
    # branch (_strided_scatter) and scatter into the parent at the right stride.
    # Read consumers of a step>1 slice are flat-indexed kernels that already
    # self-guard with .contiguous() (the binary/elementwise _prepare_binary
    # path — the rope-rotation muls — and the CLAUDE.md contiguous-guard at
    # split/matmul wrappers). The as_strided encodes the correct strided shape
    # (the historical "348 vs 174" bug was the MISSING as_strided, not the
    # missing materialisation).
    new_size = list(narrowed.shape)
    new_stride = list(narrowed._strides)
    new_size[dim] = (length + step - 1) // step
    new_stride[dim] = narrowed._strides[dim] * step
    return narrowed.as_strided(new_size, new_stride, narrowed._offset)


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


def _meta_slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    """aten.slice_scatter: a copy of `x` with `src` written where `x[..., start:end:step]` lies (along `dim`) —
    the functional form of an in-place write into a slice view (Forge `GraphBuilder.write_back_view`: Swin's
    shifted-window mask `img_mask[:, h, w, :] = cnt`). The slice is `_meta_slice`'s, the same bounds rules."""
    out = x.clone()
    _meta_slice(out, dim, start, end, step).copy_(src)
    return out


def _meta_select_scatter(x, src, dim, index):
    """aten.select_scatter: a copy of `x` with `src` written at `index` along `dim` (the functional form of an
    in-place write into a select view)."""
    out = x.clone()
    d, i = _tensor_to_int(dim), _tensor_to_int(index)
    out.select(d, i if i >= 0 else out.shape[d] + i).copy_(src)
    return out

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
    from nbxtensor.tensor import NBXTensor
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

def _chunk_narrow(x, chunks, dim=0):
    """torch.chunk semantics via narrow views.

    chunk_size = ceil(n / chunks); ceil(n / chunk_size) chunks are
    returned, the LAST carrying the remainder — exactly torch.chunk.
    The former placeholder (`x.unbind(dim) if chunks == x.size(dim)
    else (x,)`) silently returned the WHOLE tensor for any real
    chunks > 1 request (same "simplified" landmine class as the dead
    split_with_sizes → _meta_slice entry removed in P-TRITON-MLA).
    """
    chunks = int(_force_int(chunks))
    n = x.shape[dim]
    if n == 0 or chunks <= 1:
        return (x,)
    size = -(-n // chunks)  # ceil division
    return tuple(x.narrow(dim, i * size, min(size, n - i * size))
                 for i in range(-(-n // size)))

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
    from nbxtensor.ops import wrappers as w
    from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import index_wrappers as _nbx_index_wrappers  # the family the moved names live in (M2 split)
    if not isinstance(indices, (list, tuple)):
        return _nbx_index_wrappers.index_select_wrapper(x, 0, indices)

    present = [(d, i) for d, i in enumerate(indices) if i is not None]
    if not present:
        return x

    if len(present) == 1:
        # Single advanced index — unchanged behaviour (byte-identical path).
        dim, idx = present[0]
        _idx_dtn = str(getattr(idx, "dtype", "")).lower()
        if (("bool" in _idx_dtn or "uint8" in _idx_dtn)
                and dim == 0 and idx.ndim >= 1
                and list(idx.shape) == list(x.shape[:idx.ndim])):
            # Boolean-mask advanced indexing: x[mask] gathers the rows
            # where the leading-dim-prefix mask is True —
            # x.flatten(masked dims)[nonzero(mask.flatten())]. The
            # DeepStack injection read (hidden[visual_pos_masks, :]).
            # Deterministic compaction; integer masks keep the historic
            # index_select route below untouched.
            xc = x.contiguous()
            flat_rows = 1
            for _s in idx.shape:
                flat_rows *= int(_s)
            x_flat = xc.reshape([flat_rows] + list(x.shape[idx.ndim:]))
            rows = _nbx_index_wrappers.nonzero_1d_wrapper(idx.reshape([flat_rows]))
            if int(rows.shape[0]) == 0:
                # All-False mask (the text-mode empty-visual stub path):
                # a [0, tail...] gather — allocated directly, the select
                # kernel has no zero-length grid.
                from nbxtensor.tensor import NBXTensor
                return NBXTensor.empty(
                    [0] + list(x.shape[idx.ndim:]),
                    dtype=x.dtype, device=x.device)
            return _nbx_index_wrappers.index_select_wrapper(x_flat, 0, rows)
        orig_idx_shape = idx.shape
        flat_idx = idx.reshape(-1) if idx.ndim > 1 else idx
        selected = _nbx_index_wrappers.index_select_wrapper(x, dim, flat_idx)
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
        # PyTorch broadcasts differing index shapes together (right-aligned,
        # size-1 dims stretch). First real consumer: the T5 attention-mask
        # application traced with a pad-realistic stimulus (Allegro) —
        # extended_mask indexing uses [(1,1,1,1), (1,1,1,512)]. Compute the
        # common broadcast shape, expand every index (NBXTensor.expand is a
        # zero-copy stride view), and fall through to the joint flat-select.
        ndim = max(t.ndim for t in idx_tensors)
        shapes = []
        for t in idx_tensors:
            s = [1] * (ndim - t.ndim) + list(t.shape)
            shapes.append(s)
        bshape = []
        for ax in range(ndim):
            sizes_ax = {s[ax] for s in shapes} - {1}
            if len(sizes_ax) > 1:
                raise NotImplementedError(
                    f"[--triton] aten::index index shapes not broadcastable: "
                    f"{[t.shape for t in idx_tensors]}")
            bshape.append(sizes_ax.pop() if sizes_ax else 1)
        # contiguous() after expand: the stride-0 view feeds reshape(-1)
        # below (flat-indexed path) — the mandatory contiguous-guard
        # (no-op for already-contiguous tensors via the early-return).
        idx_tensors = [
            t.reshape(*([1] * (ndim - t.ndim) + list(t.shape)))
             .expand(*bshape).contiguous()
            if list(t.shape) != bshape else t
            for t in idx_tensors
        ]
        base_shape = bshape
    # Row-major flat index across the consecutive indexed dims:
    # flat = ((i0)*n1 + i1)*n2 + i2 ...   (ni = x.shape[indexed dim])
    sizes = [x.shape[d] for d in dims]
    # torch semantics: negative entries count from the end of THEIR dim —
    # wrap each index before the row-major flatten (a raw negative would
    # corrupt the joint offset; the single-dim gather wraps in-kernel).
    flats = [_nbx_elementwise_wrappers.remainder_wrapper(t.reshape(-1), int(sz)) for t, sz in zip(idx_tensors, sizes)]
    flat = flats[0]
    for k in range(1, len(flats)):
        flat = flat * sizes[k] + flats[k]
    flat = flat.long()
    block = 1
    for s in sizes:
        block *= s
    collapsed = list(x.shape[:first]) + [block] + list(x.shape[first + len(dims):])
    xc = x.reshape(*collapsed)
    sel = _nbx_index_wrappers.index_select_wrapper(xc, first, flat)
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
    from nbxtensor.ops import wrappers as w
    from nbxtensor.ops import attention_wrappers as _nbx_attention_wrappers  # the family the moved names live in (M2 split)
    q, k, v = args[0], args[1], args[2]
    attn_mask = args[3] if len(args) > 3 and args[3] is not None else None
    dropout_p = args[5] if len(args) > 5 else 0.0
    is_causal = args[6] if len(args) > 6 else False
    scale = kwargs.get("scale", args[7] if len(args) > 7 else None)
    if not isinstance(dropout_p, float):
        dropout_p = float(dropout_p)
    if not isinstance(is_causal, bool):
        is_causal = bool(is_causal)
    return _nbx_attention_wrappers.scaled_dot_product_attention_wrapper(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale,
        # The graph's recorded K layout, when the caller bound it onto this
        # meta-op. Dropping it here would leave the wrapper reading the
        # layout off a shape that cannot express it at seq_len == head_dim.
        k_pre_transposed=kwargs.get("k_pre_transposed"))


def _meta_weight_norm(v, g, dim=0, **kwargs):
    """aten::_weight_norm(v, g, dim) → w = v * g / ‖v‖, where ‖v‖ is the L2 norm
    over ALL dims except `dim` (keepdim). Pure-Triton meta-op via the existing
    mul/sum/sqrt/div wrappers (no dedicated kernel needed). Used by vocoders'
    weight-normalized convs (chatterbox s3gen, HiFi-GAN-style).
    """
    from nbxtensor.ops import wrappers as w
    from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import reduction_wrappers as _nbx_reduction_wrappers  # the family the moved names live in (M2 split)
    nd = v.ndim
    d = dim % nd if nd else 0
    vsq = _nbx_elementwise_wrappers.mul(v, v)
    nsq = vsq
    for rd in range(nd):
        if rd == d:
            continue
        nsq = _nbx_reduction_wrappers.sum_wrapper(nsq, dim=rd, keepdim=True)
    norm = _nbx_elementwise_wrappers.sqrt_wrapper(nsq)
    # w = v * (g / norm); g and norm share the norm-shape (1s except dim d),
    # broadcast against the full-shape v.
    return _nbx_elementwise_wrappers.mul(v, _nbx_elementwise_wrappers.div(g, norm))


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
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.fill_op import fill_kernel
    size, dtype, device = _extract_creation_args(args, kwargs)
    out = NBXTensor.empty(size, dtype=dtype, device=device or 'cuda')
    n = out.numel()
    if n > 0:
        _set_device(out)
        fill_kernel[_1d_grid_fixed(n)](out, 1.0, n, **_ew(n))
    return out


def _create_zeros(*args, **kwargs):
    """aten::zeros — cudaMemset zero."""
    from nbxtensor.tensor import NBXTensor
    size, dtype, device = _extract_creation_args(args, kwargs)
    return NBXTensor.zeros(size, dtype=dtype, device=device or 'cuda')


def _create_empty(*args, **kwargs):
    """aten::empty — uninitialized allocation."""
    from nbxtensor.tensor import NBXTensor
    size, dtype, device = _extract_creation_args(args, kwargs)
    return NBXTensor.empty(size, dtype=dtype, device=device or 'cuda')


def _create_full(*args, **kwargs):
    """aten::full — fill with arbitrary value via Triton fill kernel."""
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.fill_op import fill_kernel
    size = args[0] if args else kwargs.get('size', ())
    fill_value = args[1] if len(args) > 1 else kwargs.get('fill_value', 0)
    dtype = kwargs.get('dtype')
    device = kwargs.get('device', 'cuda')
    out = NBXTensor.empty(size, dtype=dtype, device=device or 'cuda')
    n = out.numel()
    if n > 0:
        _set_device(out)
        fill_kernel[_1d_grid_fixed(n)](out, float(fill_value), n, **_ew(n))
    return out


def _create_scalar_tensor(value=0, **kwargs):
    """aten::scalar_tensor — single-element tensor."""
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.fill_op import fill_kernel
    dtype = kwargs.get('dtype')
    device = kwargs.get('device', 'cuda')
    out = NBXTensor.empty((1,), dtype=dtype, device=device or 'cuda')
    _set_device(out)
    fill_kernel[(1,)](out, float(value) if value is not None else 0.0, 1, **_ew(1))
    return out


def _create_arange(start_or_end, end=None, step=1, *, dtype=None,
                   layout=None, device=None, pin_memory=None, **kwargs):
    """aten::arange — sequential values via Triton kernel.

    Reference: FlagGems ops/arange.py — uses triton arange_func kernel.
    """
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.arange_op import arange_kernel
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
        from nbxtensor.dtype.nbx_dtype import NBXDtype
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
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.zeros_like(x, dtype=dtype, device=device)


def _create_ones_like(x, *, dtype=None, layout=None, device=None,
                      pin_memory=None, **kwargs):
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.fill_op import fill_kernel
    out = NBXTensor.empty_like(x, dtype=dtype, device=device)
    n = out.numel()
    if n > 0:
        _set_device(out)
        fill_kernel[_1d_grid_fixed(n)](out, 1.0, n, **_ew(n))
    return out


def _create_empty_like(x, *, dtype=None, layout=None, device=None,
                       pin_memory=None, **kwargs):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.empty_like(x, dtype=dtype, device=device)


def _create_full_like(x, fill_value, *, dtype=None, layout=None, device=None,
                      pin_memory=None, **kwargs):
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops.fill_op import fill_kernel
    out = NBXTensor.empty_like(x, dtype=dtype, device=device)
    n = out.numel()
    if n > 0:
        _set_device(out)
        fill_kernel[_1d_grid_fixed(n)](out, float(fill_value), n, **_ew(n))
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
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops.linspace_op import linspace_kernel
    from nbxtensor.ops.fill_op import fill_kernel
    steps = int(steps)
    start = float(start)
    end = float(end)
    out = NBXTensor.empty((steps,), dtype=dtype if dtype is not None else NBXDtype.float32,
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
    from nbxtensor.tensor import NBXTensor

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
    from nbxtensor.ops import wrappers as w
    from nbxtensor.ops import attention_wrappers as _nbx_attention_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import conv_wrappers as _nbx_conv_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import index_wrappers as _nbx_index_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import linalg_wrappers as _nbx_linalg_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import loss_wrappers as _nbx_loss_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import norm_wrappers as _nbx_norm_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import random_wrappers as _nbx_random_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import reduction_wrappers as _nbx_reduction_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import shape_wrappers as _nbx_shape_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.ops import spectral_wrappers as _nbx_spectral_wrappers  # the family the moved names live in (M2 split)

    ops = {
        # =================================================================
        # COMPUTE OPS — Triton GPU kernels (wrappers.py)
        # =================================================================

        # --- Activations ---
        "relu": _nbx_elementwise_wrappers.relu,
        "silu": _nbx_elementwise_wrappers.silu,
        "gelu": _nbx_elementwise_wrappers.gelu,
        "sigmoid": _nbx_elementwise_wrappers.sigmoid_wrapper,
        "tanh": _nbx_elementwise_wrappers.tanh_wrapper,
        "hardsigmoid": _nbx_elementwise_wrappers.hardsigmoid,
        "hardswish": _nbx_elementwise_wrappers.hardswish,
        "leaky_relu": _nbx_elementwise_wrappers.leaky_relu,
        "elu": _nbx_elementwise_wrappers.elu,
        "mish": _nbx_elementwise_wrappers.mish,
        "selu": _nbx_elementwise_wrappers.selu_wrapper,

        # --- Recurrent (triton-pure assembly; self-manages dtype) ---
        "lstm": _nbx_elementwise_wrappers.lstm_wrapper,

        # --- Unary element-wise ---
        "neg": _nbx_elementwise_wrappers.neg,
        "exp": _nbx_elementwise_wrappers.exp,
        "sin": _nbx_elementwise_wrappers.sin,
        "cos": _nbx_elementwise_wrappers.cos,
        "rsqrt": _nbx_elementwise_wrappers.rsqrt,
        "sqrt": _nbx_elementwise_wrappers.sqrt_wrapper,
        "abs": _nbx_elementwise_wrappers.abs_wrapper,
        "log": _nbx_elementwise_wrappers.log_wrapper,
        "reciprocal": _nbx_elementwise_wrappers.reciprocal,
        "pow": _nbx_elementwise_wrappers.pow_wrapper,
        "clamp": _nbx_elementwise_wrappers.clamp,
        "clip": _nbx_elementwise_wrappers.clamp,  # aten::clip is an alias of aten::clamp
        "erf": _nbx_elementwise_wrappers.erf,

        # --- Binary element-wise ---
        "add": _nbx_elementwise_wrappers.add,
        "mul": _nbx_elementwise_wrappers.mul,
        "div": _nbx_elementwise_wrappers.div,
        "floor_divide": _nbx_elementwise_wrappers.floor_divide_wrapper,
        "sub": _nbx_elementwise_wrappers.sub,
        "rsub": _nbx_elementwise_wrappers.rsub,
        "where": _nbx_index_wrappers.where_wrapper,
        "maximum": _nbx_elementwise_wrappers.maximum_wrapper,
        "minimum": _nbx_elementwise_wrappers.minimum_wrapper,
        "masked_fill": _nbx_index_wrappers.masked_fill,
        "masked_scatter": _nbx_index_wrappers.masked_scatter_wrapper,

        # --- Comparisons ---
        "gt": _nbx_elementwise_wrappers.gt,
        "ge": _nbx_elementwise_wrappers.ge,
        "lt": _nbx_elementwise_wrappers.lt,
        "eq": _nbx_elementwise_wrappers.eq,
        "ne": _nbx_elementwise_wrappers.ne,
        "le": _nbx_elementwise_wrappers.le,

        # --- Normalization ---
        "native_layer_norm": _nbx_norm_wrappers.native_layer_norm,
        "layer_norm": _nbx_norm_wrappers.layer_norm_wrapper,
        "rms_norm": _nbx_norm_wrappers.rms_norm,
        "swiglu_fused": _nbx_elementwise_wrappers.swiglu_fused_wrapper,
        "rope_fused": _nbx_elementwise_wrappers.rope_fused_wrapper,
        "native_group_norm": _nbx_norm_wrappers.native_group_norm_wrapper,
        "group_norm": _nbx_norm_wrappers.group_norm_wrapper,
        "cudnn_batch_norm": _nbx_norm_wrappers.batch_norm_wrapper,
        "native_batch_norm": _nbx_norm_wrappers.batch_norm_wrapper,
        "batch_norm": _nbx_norm_wrappers.batch_norm_wrapper,

        # --- Softmax ---
        "softmax": _nbx_attention_wrappers.softmax,
        "_softmax": _nbx_attention_wrappers.softmax,
        "_safe_softmax": _nbx_attention_wrappers.softmax,
        "log_softmax": _nbx_attention_wrappers.log_softmax,
        "_log_softmax": _nbx_attention_wrappers.log_softmax,

        # --- Matmul ---
        "mm": _nbx_linalg_wrappers.mm,
        "bmm": _nbx_linalg_wrappers.bmm,
        "addmm": _nbx_linalg_wrappers.addmm,
        "mm_epilogue": _nbx_linalg_wrappers.mm_epilogue_wrapper,
        "addmm_epilogue": _nbx_linalg_wrappers.addmm_epilogue_wrapper,
        "matmul": _nbx_linalg_wrappers.matmul_wrapper,
        "linear": _nbx_linalg_wrappers.linear_wrapper,
        "isin": _nbx_elementwise_wrappers.isin_wrapper,
        "is_nonzero": _nbx_index_wrappers.is_nonzero_wrapper,

        # --- Embedding ---
        "embedding": _nbx_index_wrappers.embedding,

        # --- Reductions ---
        "mean": _nbx_reduction_wrappers.mean_wrapper,
        "sum": _nbx_reduction_wrappers.sum_wrapper,
        "norm": _nbx_norm_wrappers.norm_wrapper,
        "amax": _nbx_reduction_wrappers.amax_wrapper,
        "argmax": _nbx_reduction_wrappers.argmax_wrapper,
        "argmin": _nbx_reduction_wrappers.argmin_wrapper,
        "prod": _nbx_reduction_wrappers.prod_wrapper,
        "min": _nbx_elementwise_wrappers.min_wrapper,
        "max": _nbx_elementwise_wrappers.max_wrapper,
        "std": _nbx_reduction_wrappers.std_wrapper,
        "var": _nbx_reduction_wrappers.var_wrapper,

        # --- GLU ---
        "glu": _nbx_elementwise_wrappers.glu_wrapper,

        # --- Spatial ---
        "upsample_nearest1d": _nbx_shape_wrappers.upsample_nearest1d_wrapper,
        "upsample_nearest2d": _nbx_shape_wrappers.upsample_nearest2d_wrapper,
        # nearest-exact variants: bit-identical to nearest for integer ×2
        # (every Wan VAE upsample); reuse the nearest wrappers.
        "_upsample_nearest_exact1d": _nbx_shape_wrappers.upsample_nearest1d_wrapper,
        "_upsample_nearest_exact2d": _nbx_shape_wrappers.upsample_nearest2d_wrapper,
        "upsample_nearest3d": _nbx_shape_wrappers.upsample_nearest3d_wrapper,
        "upsample_bilinear2d": _nbx_linalg_wrappers.upsample_bilinear2d_wrapper,
        "upsample_linear1d": _nbx_linalg_wrappers.upsample_linear1d_wrapper,
        "interpolate": _nbx_shape_wrappers.interpolate_wrapper,
        "adaptive_avg_pool2d": _nbx_conv_wrappers.adaptive_avg_pool2d_wrapper,
        "avg_pool2d": _nbx_conv_wrappers.avg_pool2d_wrapper,
        "max_pool2d": _nbx_conv_wrappers.max_pool2d_wrapper,
        # with_indices variants return (values, indices) tuples — the
        # sequence binds them to ::out_0 / ::out_1.
        "max_pool2d_with_indices": (
            lambda x, kernel_size, stride=None, padding=0, dilation=1,
                   ceil_mode=False:
            _nbx_conv_wrappers.max_pool2d_wrapper(x, kernel_size, stride, padding, dilation,
                                 ceil_mode, return_indices=True)),
        "max_pool3d": _nbx_conv_wrappers.max_pool3d_wrapper,
        "max_pool3d_with_indices": _nbx_conv_wrappers.max_pool3d_with_indices_wrapper,

        # --- Conv ---
        "conv2d": _nbx_conv_wrappers.conv2d_wrapper,
        "convolution": _nbx_conv_wrappers.conv2d_wrapper,
        "conv1d": _nbx_conv_wrappers.conv1d_wrapper,
        "conv_transpose1d": _nbx_conv_wrappers.conv_transpose1d_wrapper,
        "depthwise_conv2d": _nbx_conv_wrappers.conv_depthwise2d_wrapper,
        "conv_depthwise2d": _nbx_conv_wrappers.conv_depthwise2d_wrapper,
        "conv_transpose2d": _nbx_conv_wrappers.conv_depthwise2d_wrapper,

        # --- Indexing ---
        "index_select": _nbx_index_wrappers.index_select_wrapper,
        "gather": _nbx_index_wrappers.gather_wrapper,
        "scatter": _nbx_index_wrappers.scatter_wrapper,
        "scatter_add": _nbx_index_wrappers.scatter_add_wrapper,
        "scatter_reduce": _nbx_reduction_wrappers.scatter_reduce_wrapper,
        "index_add": _nbx_index_wrappers.index_add_wrapper,
        "bincount": _nbx_elementwise_wrappers.bincount_wrapper,

        # --- Triangular ---
        "triu": _nbx_shape_wrappers.triu_wrapper,
        "tril": _nbx_shape_wrappers.tril_wrapper,

        # --- Cumulative ---
        "cumsum": _nbx_reduction_wrappers.cumsum_wrapper,

        # --- Dropout (inference = identity) ---
        "dropout": _nbx_random_wrappers.dropout_wrapper,
        "native_dropout": _nbx_random_wrappers.dropout_wrapper,

        # --- Misc element-wise ---
        "remainder": _nbx_elementwise_wrappers.remainder_wrapper,
        "exp2": _nbx_elementwise_wrappers.exp2_wrapper,
        "tan": _nbx_elementwise_wrappers.tan_wrapper,
        "celu": _nbx_elementwise_wrappers.celu_wrapper,
        "log_sigmoid": _nbx_elementwise_wrappers.log_sigmoid_wrapper,
        "softplus": _nbx_elementwise_wrappers.softplus_wrapper,
        "isfinite": _nbx_elementwise_wrappers.isfinite_wrapper,
        "isinf": _nbx_elementwise_wrappers.isinf_wrapper,
        "isnan": _nbx_elementwise_wrappers.isnan_wrapper,
        "nan_to_num": _nbx_elementwise_wrappers.nan_to_num_wrapper,
        "threshold": _nbx_elementwise_wrappers.threshold_wrapper,
        "addcdiv": _nbx_elementwise_wrappers.addcdiv_wrapper,
        "addcmul": _nbx_elementwise_wrappers.addcmul_wrapper,
        "lerp": _nbx_elementwise_wrappers.lerp_wrapper,

        # --- Logic/bitwise ---
        "all": _nbx_reduction_wrappers.all_wrapper,
        "any": _nbx_reduction_wrappers.any_wrapper,
        "bitwise_and": _nbx_elementwise_wrappers.bitwise_and_wrapper,
        "bitwise_or": _nbx_elementwise_wrappers.bitwise_or_wrapper,
        "bitwise_not": _nbx_elementwise_wrappers.bitwise_not_wrapper,
        "logical_and": _nbx_elementwise_wrappers.logical_and_wrapper,
        "logical_or": _nbx_elementwise_wrappers.logical_or_wrapper,
        "logical_not": _nbx_elementwise_wrappers.logical_not_wrapper,
        "logical_xor": _nbx_elementwise_wrappers.logical_xor_wrapper,
        "bitwise_left_shift": _nbx_elementwise_wrappers.bitwise_left_shift_wrapper,
        "bitwise_right_shift": _nbx_elementwise_wrappers.bitwise_right_shift_wrapper,

        # --- Linear algebra ---
        "dot": _nbx_linalg_wrappers.dot_wrapper,
        "mv": _nbx_linalg_wrappers.mv_wrapper,
        "baddbmm": _nbx_linalg_wrappers.baddbmm_wrapper,
        "_weight_norm": _meta_weight_norm,
        "addmv": _nbx_linalg_wrappers.addmv_wrapper,
        "linalg_vector_norm": _nbx_norm_wrappers.vector_norm_wrapper,
        "addr": _nbx_elementwise_wrappers.addr_wrapper,
        "trace": _nbx_elementwise_wrappers.trace_wrapper,

        # --- Loss ---
        "mse_loss": _nbx_loss_wrappers.mse_loss_wrapper,
        "nll_loss": _nbx_loss_wrappers.nll_loss_wrapper,
        "cross_entropy_loss": _nbx_loss_wrappers.cross_entropy_wrapper,

        # --- Sort/topk ---
        "sort": _nbx_elementwise_wrappers.sort_wrapper,
        "topk": _nbx_elementwise_wrappers.topk_wrapper,
        "flip": _nbx_shape_wrappers.flip_wrapper,
        "roll": _nbx_shape_wrappers.roll_wrapper,

        "pixel_shuffle": _nbx_shape_wrappers.pixel_shuffle_wrapper,
        "pixel_unshuffle": _nbx_shape_wrappers.pixel_unshuffle_wrapper,

        # --- Clamp ---
        "clamp_min": _nbx_elementwise_wrappers.clamp_min_wrapper,

        # --- Weight norm ---
        "_weight_norm_interface": _nbx_norm_wrappers.weight_norm_interface_wrapper,
        "repeat_interleave": _nbx_shape_wrappers.repeat_interleave_wrapper,

        # --- RNG ---
        "rand": _nbx_random_wrappers.rand_wrapper,
        "randn": _nbx_random_wrappers.randn_wrapper,
        "rand_like": _nbx_random_wrappers.rand_like_wrapper,
        "randn_like": _nbx_random_wrappers.randn_like_wrapper,
        "normal": _nbx_norm_wrappers.normal_wrapper,
        "uniform": _nbx_random_wrappers.uniform_wrapper,
        "bernoulli": _nbx_elementwise_wrappers.bernoulli_wrapper,
        "multinomial": _nbx_random_wrappers.multinomial_wrapper,

        # --- Padding ---
        "constant_pad_nd": _nbx_shape_wrappers.constant_pad_nd_wrapper,
        "pad": _nbx_shape_wrappers.pad_wrapper,
        "reflection_pad1d": _nbx_shape_wrappers.reflection_pad1d_wrapper,
        "reflection_pad2d": _nbx_shape_wrappers.reflection_pad2d_wrapper,
        "reflection_pad3d": _nbx_shape_wrappers.reflection_pad3d_wrapper,
        "replication_pad1d": _nbx_shape_wrappers.replication_pad1d_wrapper,
        "replication_pad2d": _nbx_shape_wrappers.replication_pad2d_wrapper,
        "replication_pad3d": _nbx_shape_wrappers.replication_pad3d_wrapper,

        # --- FFT ---
        "_fft_r2c": _nbx_spectral_wrappers.fft_r2c_wrapper,
        "_fft_c2r": _nbx_spectral_wrappers.fft_c2r_wrapper,
        "_fft_c2c": _nbx_spectral_wrappers.fft_c2c_wrapper,
        "fft_rfft": _nbx_spectral_wrappers.fft_rfft_wrapper,
        "fft_irfft": _nbx_spectral_wrappers.fft_irfft_wrapper,
        "stft": _nbx_spectral_wrappers.stft_wrapper,
        "istft": _nbx_spectral_wrappers.istft_wrapper,

        # --- Complex ---
        "angle": _nbx_elementwise_wrappers.angle_wrapper,

        # Attention — Dao-AILab Flash Attention v2 Triton kernel
        "scaled_dot_product_attention": _nbx_attention_wrappers.scaled_dot_product_attention_wrapper,
        "_scaled_dot_product_attention": _nbx_attention_wrappers.scaled_dot_product_attention_wrapper,
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
        "_scaled_dot_product_flash_attention_for_cpu": _nbx_attention_wrappers.scaled_dot_product_attention_wrapper,
        "complex": _nbx_elementwise_wrappers.complex_wrapper,
        "fold": _nbx_elementwise_wrappers.fold_wrapper,
        "unfold_backward": _nbx_conv_wrappers.unfold_backward_wrapper,

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
        "slice_scatter": _meta_slice_scatter,
        "select_scatter": _meta_select_scatter,
        "unbind": _meta_unbind,
        # NOTE: split_with_sizes is bound ONCE, in the Split/chunk section
        # below (narrow-based multi-output tuple). A historical
        # `"split_with_sizes": _meta_slice  # simplified` entry sat here and
        # was only harmless because the later duplicate key in this same
        # dict literal shadowed it — removed so a reorder can never
        # resurrect the wrong (single-output, wrong-signature) binding
        # (P-TRITON-MLA: the MLA q nope/rope split [128, 64] rides on the
        # narrow-based binding).
        "bucketize": _nbx_index_wrappers.bucketize_wrapper,
        "view_as": _meta_view_as,
        "reshape_as": _meta_reshape_as,
        "as_strided": _meta_as_strided,
        "unfold": _meta_unfold,
        "im2col": _nbx_conv_wrappers.im2col_wrapper,
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
        "floor": _nbx_elementwise_wrappers.floor_wrapper,
        "ceil": _nbx_elementwise_wrappers.ceil_wrapper,

        # 2D grid sampler (VLM pos-embed adaptation: bicubic/bilinear)
        "grid_sampler_2d": _nbx_shape_wrappers.grid_sampler_2d_wrapper,
        "round": _nbx_elementwise_wrappers.round_wrapper,
        "trunc": _nbx_elementwise_wrappers.trunc_wrapper,

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
        "broadcast_tensors": _nbx_shape_wrappers.broadcast_tensors_wrapper,

        # Advanced indexing — aten::index(tensor, [idx0, idx1, ...])
        "index": _meta_index,
        "index_put": _nbx_index_wrappers.index_put_wrapper,
        "index_put_": _nbx_index_wrappers.index_put_wrapper,

        # Split/chunk (use narrow internally). torch.split semantics:
        # ceil(shape/size) chunks, the LAST carries the remainder — the
        # floor-division form returned an EMPTY tuple whenever
        # split_size > shape[dim] (the audio stem's conv_chunksize 500
        # over 5 windows → None into the first convolution) and silently
        # DROPPED the tail chunk in the general case.
        "split": lambda x, size, dim=0: tuple(
            x.narrow(dim, i * size, min(size, x.shape[dim] - i * size))
            for i in range((x.shape[dim] + size - 1) // size)),
        "split_with_sizes": lambda x, sizes, dim=0: tuple(
            x.narrow(dim, sum(sizes[:i]), s) for i, s in enumerate(sizes)),
        "chunk": _chunk_narrow,
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
    from nbxtensor.host import need as _need   # the application's kernels.classification (nbxtensor.host)
    canonical_aten = _need("canonical_aten")
    return _OP_MAP.get(canonical_aten(op_name))


def has_kernel(op_type: str) -> bool:
    return dispatch(op_type) is not None


def list_kernels() -> list:
    global _OP_MAP
    if _OP_MAP is None:
        _OP_MAP = _build_op_map()
    return sorted(_OP_MAP.keys())
