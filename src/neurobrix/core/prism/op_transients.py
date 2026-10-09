"""The transient each op holds on top of its output — the ONE function Prism prices it with.

Why this exists. The placement estimate (`ActivationProfiler.estimate_peak_memory`) walked the
live set op by op and never added what an op holds WHILE it runs beyond its output: the Triton
engines were priced at zero workspace (`memory_estimator.estimate_op_workspace_bytes` returns 0
for them), so the math route's fp32 scores, the chunked route's per-chunk scores, the conv2d band,
the conv3d one-shot folds and the cast copies of an AMP island were all free in the plan and paid
at the launch. Measured on the Mac (2026-10-08): mochi-1-preview's VAE at 7 frames priced
2 564 MB untiled and peaked at 11 039 MB.

What it does. `op_transient_bytes` answers one op's transient from the engines' own sizing
functions and nothing else:
  * the TilingEngine (`core/module/tiling_sizes`): the attention route and its scores
    (`sdpa_route`, `sdpa_transient_bytes` — the function the wrapper and the derived census route
    with), the conv2d band (`conv2d_band_transient_bytes`), the planned tiled conv2d
    (`tiled_conv2d_transient_bytes`), the conv3d one-shot or chunked peak
    (`conv3d_transient_bytes`);
  * the DtypeEngine (`core/dtype/itemsize.cast_copy_bytes`): the copy its AMP wrap makes of an
    input held at another dtype than the op executes in (`execution_dtype`, the engines' own AMP
    class sets), and the result held at the execution dtype before a cast back, at the dtypes
    `core/prism/runtime_widths.runtime_dtypes` says the engine runs;
  * the compiled engine's vendor library (`library_layout_transient_bytes`): a convolution its
    library runs on the matrix unit in the channels-last layout while the graph holds it
    channels-first is transposed in and out — the input and the weight at the execution dtype,
    the result before its transpose back — at the dtypes and, per spatial rank, the buffers the
    vendor file declares (`conv.library_layout_copies`: `dtypes`, `conv<N>d` = the buffers among
    input / weight / output). NVIDIA: "NCHW ... incur[s] automatic transposes" on Tensor Cores
    (https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional). MEASURED
    (cards 2/3, 2026-10-09): mochi-1-preview's VAE `aten.convolution::33` at [1,128,14,322,578]
    fp32 -> fp16 held 1 177 MB above its output and cast copy = fp16 input 636 + output 540 +
    weight 0.8. Apple MPS (M4 Pro, 2026-10-09, the Mac's differential probe): a 2D convolution
    holds nothing, a 3D one holds one input-sized copy at fp16, bf16 and fp32;
  * the Triton wrappers' contiguity guard (`contiguous_copy_bytes`): a reader that copies its
    input contiguous before its kernel (`CONTIGUOUS_COPY_READERS`) holds that copy when the input
    is a strided view (`strided_view_tensors`: a permute, a transpose). MEASURED (card 2,
    2026-10-09, pool off): Sana_1600M_4Kpx_BF16's VAE `aten.relu::18` on the [0,3,1,2] permute of a
    [1,3072,4096,128] fp32 tensor held 6 144 MB above its output — the copy, at the component's peak;
The compiled engine's library workspace proper (cuDNN im2col, the mem-efficient logsumexp) is not
a TilingEngine split and stays where it was, in the overflow scan's
`memory_estimator.estimate_op_workspace_bytes`.

What it reads from the hardware profile: the attention budget the wrapper routes on
(`memory.sdpa_math_max_scores_bytes`, `memory.sdpa_math_scores_device_fraction` of the device,
`memory.sdpa_math_min_chunk_rows`, `memory.sdpa_math_max_chunks`) and the matrix unit's flash rows
(`matrix_unit.flash`, `matrix_unit.operand_dtype`) — the same keys `kernels/wrappers.py` reads, so
the route priced is the route launched. An architecture with no vendor file is priced as the
wrapper runs without one: no budget, no chunk, no unit.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence

from neurobrix.core.dtype.itemsize import cast_copy_bytes, itemsize
from neurobrix.core.module import tiling_sizes as _ts

#: Op families whose output IS their input re-represented (an explicit cast): the copy is the
#: output, already in the live set, never a second transient.
_CAST_OPS = frozenset({"aten::_to_copy", "aten::to", "aten::type_as"})
_FLOATS = frozenset({"float16", "bfloat16", "float32", "float64"})


#: The ops whose TRITON wrapper copies a strided input contiguous before its kernel runs, by the
#: canonical ATen name: every elementwise unary wrapper of kernels/wrappers.py that opens on
#: `x = x.contiguous()` (relu ... logical_not), `conv2d_wrapper` (`x_c = x.contiguous()`) and
#: `NBXTensor.cat` (`tensors = [t.contiguous() for t in tensors]`). The binary elementwise
#: wrappers, the norms and the contractions read a strided operand by its strides and are not
#: here (measured: Sana 4K VAE's add, rms_norm and mm on permuted inputs held no copy).
#: Held to the wrappers' source by tests/unit/prism/test_a_strided_input_is_copied_where_its_reader_copies_it.py.
CONTIGUOUS_COPY_READERS = frozenset({
    "relu", "silu", "gelu", "sigmoid", "tanh", "hardsigmoid", "hardswish", "leaky_relu", "elu",
    "mish", "selu", "neg", "sin", "cos", "rsqrt", "sqrt", "log", "reciprocal", "erf", "floor",
    "ceil", "round", "trunc", "softplus", "exp2", "tan", "celu", "log_sigmoid", "isfinite",
    "isinf", "isnan", "threshold", "logical_not",
    "convolution", "cat",
})


def strided_view_tensors(ops: Mapping[str, Dict[str, Any]]) -> FrozenSet[str]:
    """The tensor ids a graph's views leave strided: the output of a permute that moves an axis and
    of a transpose of two distinct axes (`aten::t` included). A weight's transpose read in place by
    a contraction is no reader of `CONTIGUOUS_COPY_READERS`, so it costs nothing here."""
    out = set()
    for op in ops.values():
        t = str(op.get("op_type", ""))
        at = op.get("attributes") or {}
        if t == "aten::permute":
            dims = at.get("dims")
            if dims is None or list(dims) == sorted(dims):
                continue
        elif t == "aten::transpose":
            d0, d1 = at.get("dim0"), at.get("dim1")
            if d0 is not None and d1 is not None and d0 == d1:
                continue
        elif t != "aten::t":
            continue
        out.update(op.get("output_tensor_ids") or ())
    return frozenset(out)


@dataclass(frozen=True)
class TransientContext:
    """What one component's ops are priced under: the engine, the attention route's arch inputs on
    the executing device, and the splits the plan makes (`conv3d_chunks`, `tiled_ops`)."""
    engine: str                                   # "triton" or "compiled" (host_footprint.engine_of)
    compute_dtype: str = ""                       # the component's plan compute dtype
    sdpa_budget_bytes: int = 0
    sdpa_min_chunk_rows: int = 0
    sdpa_max_chunks: int = 0
    matrix_unit: Mapping[str, Any] = field(default_factory=dict)   # the arch yml's matrix_unit
    band_bytes: int = 0
    conv3d_chunks: FrozenSet[str] = field(default_factory=frozenset)
    tiled_ops: Mapping[str, int] = field(default_factory=dict)
    # the component's precision contract (`runtime_widths.PrecisionContract`): the SAME object
    # `runtime_dtypes` reads, so the dtypes the walk holds and the dtype each op executes at come
    # from one record. None = no record: nothing safe, no island, nothing narrowed.
    contract: Optional[Any] = None
    # the compiled engine's vendor library: the execution dtypes at which it copies a convolution's
    # buffers to its own layout, and which buffers per spatial rank (the vendor file's
    # `conv.library_layout_copies`: `dtypes`, `conv<N>d`).
    conv_layout_copy_dtypes: FrozenSet[str] = field(default_factory=frozenset)
    conv_layout_copy_buffers: Mapping[int, FrozenSet[str]] = field(default_factory=dict)
    # the component's strided views (`strided_view_tensors`), bound by the walk from its graph.
    strided_views: FrozenSet[str] = field(default_factory=frozenset)


def context_for(profile, engine: str, compute_dtype: str, contract=None) -> TransientContext:
    """The context of `profile`'s first device under `engine` at `compute_dtype` — the keys the
    wrappers read — for a component under `contract` (its plan-time precision contract)."""
    compute_dtype = str(compute_dtype).replace("torch.", "")
    devices = getattr(profile, "devices", None) or []
    band = _ts.conv2d_band_bytes()
    if not devices:
        return TransientContext(engine=engine, compute_dtype=compute_dtype, band_bytes=band,
                                contract=contract)
    dev = devices[0]
    from neurobrix.core.config.loader import UnsupportedArchitectureError, get_vendor_config
    vendor = getattr(dev.brand, "value", dev.brand)
    try:
        cfg = get_vendor_config(vendor, dev.architecture)
    except UnsupportedArchitectureError:
        # The wrapper's own answer with no vendor file: no budget, no chunk, no unit.
        return TransientContext(engine=engine, compute_dtype=compute_dtype, band_bytes=band,
                                contract=contract)
    mem = cfg.get("memory") or {}
    copies = (cfg.get("conv") or {}).get("library_layout_copies") or {}
    base = int(mem.get("sdpa_math_max_scores_bytes", 0) or 0)
    budget = _ts.sdpa_device_scores_budget(
        base, float(mem.get("sdpa_math_scores_device_fraction", 0.0) or 0.0), dev.memory_mb)
    return TransientContext(
        engine=engine, compute_dtype=compute_dtype, sdpa_budget_bytes=budget,
        sdpa_min_chunk_rows=int(mem.get("sdpa_math_min_chunk_rows", 0) or 0),
        sdpa_max_chunks=int(mem.get("sdpa_math_max_chunks", 0) or 0),
        matrix_unit=dict(cfg.get("matrix_unit") or {}),
        band_bytes=band, contract=contract,
        conv_layout_copy_dtypes=frozenset(copies.get("dtypes") or ()),
        conv_layout_copy_buffers=layout_copy_buffers(copies))


_CONV_BUFFERS = ("input", "weight", "output")


def layout_copy_buffers(copies: Mapping[str, Any]) -> Dict[int, FrozenSet[str]]:
    """`conv.library_layout_copies`' `conv<N>d` entries as {spatial rank: buffers}; a buffer name
    outside input / weight / output is refused by name."""
    out: Dict[int, FrozenSet[str]] = {}
    for key, bufs in copies.items():
        m = re.fullmatch(r"conv(\d)d", str(key))
        if m is None:
            if key != "dtypes":
                raise ValueError(f"conv.library_layout_copies: unknown key {key!r}")
            continue
        bad = sorted(set(bufs or ()) - set(_CONV_BUFFERS))
        if bad:
            raise ValueError(f"conv.library_layout_copies.{key}: unknown buffers {bad}")
        out[int(m.group(1))] = frozenset(bufs or ())
    return out


def _unit_flash(ctx: TransientContext, D: int, dtypes: Sequence[Optional[str]]) -> bool:
    """`launch_keys.unit_flash_takes` over the profile's matrix unit: a flash row holds head dim D
    (`matrix_unit_flash_tile`, the function the kernel picks its tile with) and q, k, v — after the
    wrapper's operand alignment (`launch_keys.sdpa_operand_dtypes`) — are its operand dtype."""
    if not ctx.matrix_unit or any(d is None for d in dtypes):
        return False
    from neurobrix.kernels.ops._configs import matrix_unit_flash_tile
    if matrix_unit_flash_tile(D, dict(ctx.matrix_unit)) is None:
        return False
    if len(set(dtypes)) > 1:
        from neurobrix.kernels import launch_keys as _lk
        from neurobrix.kernels.nbx_tensor import NBXDtype
        dtypes = [d.name for d in _lk.sdpa_operand_dtypes(*(NBXDtype[d] for d in dtypes))[:3]]
    return all(d == ctx.matrix_unit.get("operand_dtype") for d in dtypes)


def _num(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _pair(v, i: int) -> int:
    if isinstance(v, (list, tuple)):
        return int(v[min(i, len(v) - 1)])
    return int(v)


def tiling_transient_bytes(uid: str, op: Dict[str, Any], in_shapes: List[List[int]],
                           out_shapes: List[List[int]], in_dtypes: List[Optional[str]],
                           out_dtype: Optional[str], ctx: TransientContext) -> int:
    """The TilingEngine's half of an op's transient on the Triton engines (0 on the compiled one,
    whose library workspace `op_transient_bytes` asks the compiled estimator for)."""
    if ctx.engine != "triton":
        return 0
    op_type = str(op.get("op_type", ""))
    at = op.get("attributes") or {}
    if "scaled_dot_product" in op_type and len(in_shapes) >= 3:
        q, k, v = in_shapes[0], in_shapes[1], in_shapes[2]
        if len(q) != 4 or len(k) != 4 or len(v) != 4:
            return 0
        B, H, Tq, D = (int(d) for d in q)
        Tk, Dv = int(k[2]), int(v[3])
        unit = _unit_flash(ctx, D, list(in_dtypes[:3]))
        route, rows = _ts.sdpa_route(B, H, Tq, Tk, D, Dv, ctx.sdpa_budget_bytes, ctx.sdpa_min_chunk_rows,
                                     ctx.sdpa_max_chunks,
                                     force_math=os.environ.get("NBX_FORCE_MATH_ATTENTION") == "1",
                                     unit_flash=unit)
        return _ts.sdpa_transient_bytes(route, rows, B, H, Tq, Tk)
    if "convolution" in op_type and len(in_shapes) >= 2 and out_shapes:
        x, w, y = in_shapes[0], in_shapes[1], out_shapes[0]
        if at.get("transposed") or not in_dtypes or in_dtypes[0] is None:
            return 0
        ib = itemsize(in_dtypes[0])
        ob = itemsize(out_dtype) if out_dtype else ib
        stride, padding, dilation = at.get("stride", 1), at.get("padding", 0), at.get("dilation", 1)
        if len(x) == 5 and len(w) == 5:
            return _ts.conv3d_transient_bytes(x, w, stride, padding, dilation, ib, ob,
                                              chunked=uid in ctx.conv3d_chunks)
        if len(x) == 4 and len(w) == 4 and len(y) == 4:
            N, in_c, IH, IW = (int(d) for d in x)
            out_c, _cg, kh, _kw = (int(d) for d in w)
            out_h, out_w = int(y[2]), int(y[3])
            groups = int(at.get("groups", 1) or 1)
            sh, dh = _pair(stride, 0), _pair(dilation, 0)
            if uid in ctx.tiled_ops:
                return _ts.tiled_conv2d_transient_bytes(
                    N, in_c, IH, IW, ib, out_c, out_h, out_w, ob, kh, sh, dh,
                    _pair(padding, 0), _pair(padding, 1), len(in_shapes) > 2, int(ctx.tiled_ops[uid]))
            # `launch_keys.conv2d_route`: depthwise first (no band), then the band over `band_bytes`.
            if groups == in_c == out_c and dh == 1 and _pair(dilation, 1) == 1:
                return 0
            return _ts.conv2d_band_transient_bytes(N, in_c, IW, ib, out_c, out_h, out_w, ob, kh, sh, dh,
                                                   ctx.band_bytes)
    return 0


def execution_dtype(uid: str, op: Dict[str, Any], in_dtypes: Sequence[Optional[str]],
                    ctx: TransientContext) -> Optional[str]:
    """The dtype the DtypeEngine's wrap makes op `uid` compute in, or None when it wraps none (a
    self-managed wrapper, a full-precision compute dtype, an op of no AMP class). The engines'
    own rules, in their order, under the component's precision contract (`ctx.contract`):
      * both engines: a calibration island (`fp32_op_uids`) computes in fp32 whatever its class;
      * Triton engines (`triton/dtype.TritonDtypeEngine.wrap_op`): a narrowed fp32-class op
        computes in fp32 (only its store narrows); `_SELF_MANAGED_OPS` are unwrapped; an fp32 op
        (and, under fp16, `div`) computes in fp32; a half op in the compute dtype; a promote op in
        its widest floating input. The contract moves no other op's compute;
      * compiled (`core/dtype/engine.py` `_resolve_args`, as `runtime_widths` mirrors it): under a
        safe fp16 contract the fp16-IO norms/softmax (`_FP32_OPS_HALF_IO`) and the matmuls
        (`_FP16_GEMM_OPS`) take the compute dtype, and so does a narrowed `div`; otherwise as
        the Triton classes with the engine's `_FP16_NEED_FP32` (mm, bmm, addmm, div)."""
    c = ctx.compute_dtype
    if c not in ("float16", "bfloat16"):
        return None
    from neurobrix.core.prism import runtime_widths as _rw
    from neurobrix.kernels.classification import canonical_aten
    k = ctx.contract
    islands = frozenset(getattr(k, "fp32_op_uids", ()) or ())
    narrow = frozenset(getattr(k, "narrow_op_uids", ()) or ())
    if uid in islands:
        return "float32"
    name = canonical_aten(str(op.get("op_type", "")).split("::")[-1].split(".")[0])
    if ctx.engine == "triton":
        from neurobrix.triton import dtype as _tdt
        if uid in narrow and name in _tdt.AMP_FP32_OPS:
            return "float32"
        if name in _tdt._SELF_MANAGED_OPS:
            return None
        if name in _tdt.AMP_FP32_OPS or (c == "float16" and name in _tdt._FP16_NEED_FP32
                                          and name in _tdt.AMP_FP16_OPS):
            return "float32"
        if name in _tdt.AMP_FP16_OPS:
            return c
        promote = _tdt.AMP_PROMOTE_OPS
    else:
        safe = bool(getattr(k, "safe", False)) and c == "float16"
        if name in _rw.ATEN_AMP_FP32_OPS:
            return c if safe and name in _rw.ATEN_FP32_OPS_HALF_IO else "float32"
        if name in _rw.ATEN_AMP_FP16_OPS:
            if c == "float16" and name in _rw.ATEN_FP16_NEED_FP32:
                if safe and (name in _rw.ATEN_FP16_GEMM_OPS or uid in narrow):
                    return c
                return "float32"
            return c
        promote = _rw.ATEN_AMP_PROMOTE_OPS
    if name in promote:
        floats = [d for d in in_dtypes if d in _FLOATS]
        return max(floats, key=itemsize) if floats else None
    return None


def cast_transient_bytes(uid: str, op: Dict[str, Any], in_shapes: List[List[int]], out_shapes: List[List[int]],
                         in_dtypes: List[Optional[str]], out_dtype: Optional[str],
                         ctx: TransientContext) -> int:
    """The DtypeEngine's half: each floating input held at another dtype than the op's execution
    dtype (`execution_dtype`) is copied at it first (`cast_copy_bytes`), and a result computed at
    another dtype than the one it is stored at (the fp32-internal wrap's cast back) is held at its
    execution dtype before the cast. An explicit cast's copy is its output, already live."""
    if str(op.get("op_type", "")) in _CAST_OPS:
        return 0
    ex = execution_dtype(uid, op, in_dtypes, ctx)
    if ex is None:
        return 0
    total = 0
    for sh, dt in zip(in_shapes, in_dtypes):
        if dt in _FLOATS:
            total += cast_copy_bytes(dt, ex, _num(sh))
    if out_dtype in _FLOATS and out_shapes:
        total += cast_copy_bytes(out_dtype, ex, _num(out_shapes[0]))
    return total


def library_layout_transient_bytes(uid: str, op: Dict[str, Any], in_shapes: List[List[int]],
                                   out_shapes: List[List[int]], in_dtypes: List[Optional[str]],
                                   ctx: TransientContext) -> int:
    """The compiled engine's vendor-library half: a convolution executing at a dtype the vendor file
    declares in `conv.library_layout_copies.dtypes` holds, at the execution dtype, the copies of
    the buffers its `conv<N>d` entry names for its spatial rank N while it runs (0 on the Triton
    engines, which run the graph's layout, and on an arch or rank whose file declares none)."""
    if ctx.engine == "triton" or not ctx.conv_layout_copy_dtypes:
        return 0
    if "convolution" not in str(op.get("op_type", "")) or len(in_shapes) < 2 or not out_shapes:
        return 0
    ex = execution_dtype(uid, op, in_dtypes, ctx) or (in_dtypes[0] if in_dtypes else None)
    if ex not in ctx.conv_layout_copy_dtypes:
        return 0
    bufs = ctx.conv_layout_copy_buffers.get(len(in_shapes[1]) - 2, frozenset())
    sizes = {"input": _num(in_shapes[0]), "weight": _num(in_shapes[1]), "output": _num(out_shapes[0])}
    return itemsize(ex) * sum(sizes[b] for b in bufs)


def contiguous_copy_bytes(op: Dict[str, Any], in_shapes: List[List[int]],
                          in_dtypes: List[Optional[str]], ctx: TransientContext) -> int:
    """The Triton wrappers' contiguity guard: a reader of `CONTIGUOUS_COPY_READERS` holds a dense
    copy of each input that is a strided view (`ctx.strided_views`) at that input's dtype (0 on the
    compiled engine, whose library reads by strides)."""
    if ctx.engine != "triton" or not ctx.strided_views:
        return 0
    from neurobrix.kernels.classification import canonical_aten
    name = canonical_aten(str(op.get("op_type", "")).split("::")[-1].split(".")[0])
    if name not in CONTIGUOUS_COPY_READERS:
        return 0
    total = 0
    for tid, sh, dt in zip(op.get("input_tensor_ids") or (), in_shapes, in_dtypes):
        if tid in ctx.strided_views and dt is not None:
            total += _num(sh) * itemsize(dt)
    return total


def op_transient_bytes(uid: str, op: Dict[str, Any], in_shapes: List[List[int]], out_shapes: List[List[int]],
                       in_dtypes: List[Optional[str]], out_dtype: Optional[str], ctx: TransientContext) -> int:
    """ONE op's transient on top of its output: the TilingEngine's split bytes (Triton engines)
    plus the DtypeEngine's cast copies (both engines) plus the compiled library's layout copies
    plus the Triton wrappers' contiguous copies of strided inputs."""
    return (tiling_transient_bytes(uid, op, in_shapes, out_shapes, in_dtypes, out_dtype, ctx)
            + cast_transient_bytes(uid, op, in_shapes, out_shapes, in_dtypes, out_dtype, ctx)
            + library_layout_transient_bytes(uid, op, in_shapes, out_shapes, in_dtypes, ctx)
            + contiguous_copy_bytes(op, in_shapes, in_dtypes, ctx))
