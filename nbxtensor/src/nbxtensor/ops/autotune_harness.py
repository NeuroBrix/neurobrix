"""The kernel-level harness of a certification: a key's operands drawn, the launch that forms it, its float64
reference, and every candidate configuration launched, measured against the reference and timed.

A certifier asks, for one autotune key of one kernel, which configuration to serve. This module answers the tensor half:

- `synthesize(qual, tuner, key, rng, ...)`: (call, oracle_fn, output argument) — a callable that runs the kernel's own
  wrapper on operands of the key's shapes and dtypes (so the wrapper forms the key itself), and a LAZY callable that
  computes the reference of the same operands (`oracles.fp64`). A key is drawn at the size that forms it: a GEMM at a
  row count whose launch forms its bucket (`gemm_rows`), a padded-row class at the shortest row of the class where the
  caller's capacity cannot hold its top (`Room`), an element-wise class at its floor likewise.
- `key_operands(qual, tuner, key)`: the (shape, dtype) of every operand and of the output that `synthesize` draws, read
  from the same key fields, for a caller that prices a key before any draw.
- `LaunchUnderTest`: the intercepted launch, run once per candidate — its output poisoned (0xFF: NaN in every float
  dtype) before each, launched, synchronised, its deviation read on the reference's windows only — then timed.
- `launch_oracle`: the reference of THE LAUNCH the autotuner sees, from its own operands, for a kernel the screen
  row-windows (a product the wrapper streams in bands is proven band by band).
- `unit_runs`, `time_launch`, `witness_ms`, `candidate_configs`, `stored_config`, `config_dict`.

An operand is drawn at the KERNEL'S width through a bounded chunk (bf16 as its bits in a uint16 array, `Synth`): the
host holds the operand once, never a float32 copy beside it. The caller decides which keys, at which tolerance, under
which timing protocol, against which memory budget, and writes the result; this module never does.

Moved from the NeuroBrix certifier (2026-11-12).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nbxtensor.dtype import itemsize as _isz
from nbxtensor.dtype.convert import bf16_bits_to_f32, f32_to_bf16_bits
from nbxtensor.ops import autotune_certified as C
from nbxtensor.ops.oracles import fp64 as F

__all__ = ["NP_DTYPES", "INTEGRAL", "Synth", "ArgSpec", "draw", "spec", "upload", "KeyTooLargeForClass", "Room",
           "shortest_row", "gemm_rows", "key_operands", "synthesize", "launch_oracle", "LaunchUnderTest",
           "launch_and_measure", "unit_runs", "fused_moe_unit_runs", "time_launch", "witness_ms", "candidate_configs",
           "stored_config", "config_dict"]

#: The numpy array every dtype a key carries is drawn into: bf16 as its BITS (uint16), Triton's `int1` as a boolean.
#: Every name `autotune_certified` recognises must be drawable here, or a key carrying it is recognised and then
#: unsynthesisable (a batched GEMM's bias is a `uint8` attention mask on some models).
NP_DTYPES = {"fp16": np.float16, "bf16": np.uint16, "fp32": np.float32, "fp64": np.float64,
             "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64,
             "int1": np.bool_, "int8": np.int8, "int16": np.int16,
             "int32": np.int32, "int64": np.int64, "bool": np.bool_}

#: The integral dtypes: a mask or an index, drawn as zeros and ones.
INTEGRAL = {"uint8", "uint16", "uint32", "uint64", "int1", "int8", "int16", "int32", "int64", "bool"}

#: A key's dtype spelling where NBXTensor's differs (Triton's `int1` is a boolean tensor).
_SPEC_NBX = {"int1": "bool"}

#: The batch a batched GEMM key is drawn at.
_BMM_BATCH = 2

_HALF = ("fp16", "bf16", "fp32")


def _nbx(name: str):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    return {"fp16": NBXDtype.float16, "bf16": NBXDtype.bfloat16, "fp32": NBXDtype.float32}[name]


class Synth(np.ndarray):
    """An array that remembers the dtype name its values belong to (`key_dtype`). numpy has no bfloat16: a bf16
    operand is carried as its BITS, in a uint16 array at the kernel's own width, and its numbers are read through
    `oracles.fp64.values`, which decodes them exactly (a bf16 value is exact in float32 and float64)."""

    def __new__(cls, arr, key_dtype):
        obj = np.asarray(arr).view(cls)
        obj.key_dtype = key_dtype
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.key_dtype = getattr(obj, "key_dtype", None)

    def astype(self, dtype, *args, **kwargs):
        # bf16 BITS cast to a float are integers, not values: refused by name, never read silently
        if getattr(self, "key_dtype", None) == "bf16" and np.dtype(dtype).kind == "f":
            raise TypeError("a bf16 operand is carried as its bits: read its numbers through "
                            "oracles.fp64.values(), not astype")
        return super().astype(dtype, *args, **kwargs)


class ArgSpec:
    """An operand known by its shape and dtype only (`synthesize(values=False)`)."""
    __slots__ = ("shape", "dtype_name")

    def __init__(self, shape, dtype_name):
        self.shape, self.dtype_name = tuple(int(v) for v in shape), dtype_name


def _known(dtype_name: str) -> None:
    if dtype_name not in NP_DTYPES:
        # no fallback: float32 for an unknown dtype would certify another key than the one asked for
        raise RuntimeError(
            f"certify: no synthesis for dtype {dtype_name!r}. A key carrying it cannot be proven until this table "
            f"knows it; defaulting to float32 would certify a different key than the one asked for.")


def draw(rng, shape, dtype_name, scale=0.1) -> Synth:
    """An operand of `shape` at the kernel's width: an integral one zeros and ones, a float one a normal sample times
    `scale`, drawn in float32 (float64 for fp64) through a 1 MB chunk and written at its width (bf16 rounded to
    nearest, ties to even). The generator fills sequentially, so the chunks are the whole draw's values, bit for bit."""
    _known(dtype_name)
    if dtype_name in INTEGRAL:
        a = rng.integers(0, 2, size=shape, dtype=np.int8)
        return Synth(a.astype(NP_DTYPES[dtype_name], copy=False), dtype_name)
    kind = np.float64 if dtype_name == "fp64" else np.float32
    out = np.empty(shape, dtype=NP_DTYPES[dtype_name])
    flat = out.reshape(-1)
    step = 1 << 18
    for i in range(0, flat.size, step):
        chunk = rng.standard_normal(min(step, flat.size - i), dtype=kind)
        chunk *= kind(scale)
        flat[i:i + step] = f32_to_bf16_bits(chunk) if dtype_name == "bf16" else chunk
    return Synth(out, dtype_name)


def spec(rng, shape, dtype_name, scale=0.1) -> ArgSpec:
    """`draw`'s signature, no draw: the same refusal of a dtype the table cannot build."""
    _known(dtype_name)
    return ArgSpec(shape, dtype_name)


def upload(a):
    """The NBXTensor the kernel receives, at the dtype the key names: a bf16 operand IS its bits (`from_numpy`'s
    dtype names the dtype the bits already are), an `ArgSpec` an empty tensor of its shape."""
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    if isinstance(a, ArgSpec):
        return NBXTensor.empty(a.shape, dtype=_SPEC_NBX.get(a.dtype_name, a.dtype_name))
    if getattr(a, "key_dtype", None) == "bf16":
        return NBXTensor.from_numpy(np.ascontiguousarray(np.asarray(a)), dtype=NBXDtype.bfloat16)
    return NBXTensor.from_numpy(np.ascontiguousarray(np.asarray(a)))


def _upload_conv_weight(a, groups):
    """A convolution weight in the order the loader stores it on this arch (`launch_keys.conv_weight_layout`): a KRSC
    weight is timed and checked as the run reads it, the logical (K, C, R, S) tensor over channels-innermost memory."""
    from nbxtensor.ops import launch_keys as _lk
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    if not _lk.conv_weight_krsc(tuple(a.shape), groups, False, _lk.conv_weight_layout()):
        return upload(a)
    K, Cg, R, S = a.shape
    if isinstance(a, ArgSpec):
        return NBXTensor.empty((K, R, S, Cg), dtype=_SPEC_NBX.get(a.dtype_name, a.dtype_name)).permute(0, 3, 1, 2)
    bits = np.ascontiguousarray(np.asarray(a).transpose(0, 2, 3, 1))
    if getattr(a, "key_dtype", None) == "bf16":
        return NBXTensor.from_numpy(bits, dtype=NBXDtype.bfloat16).permute(0, 3, 1, 2)
    return NBXTensor.from_numpy(bits).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------------------------------------------------
# the size a key is drawn at
# ---------------------------------------------------------------------------------------------------------------------
class KeyTooLargeForClass(RuntimeError):
    """The key's operands and output exceed the capacity the caller certifies against: refused BEFORE a value is
    drawn. `usable` is that capacity (the card's whole bytes when not given)."""

    def __init__(self, asked: int, card: int, what: str, usable: Optional[int] = None):
        self.asked, self.card = int(asked), int(card)
        self.usable = int(usable) if usable is not None else self.card
        super().__init__(f"the key's operands and output ask {self.asked} bytes of the {self.usable} bytes a "
                         f"{self.card}-byte card plans against ({what})")


def shortest_row(feat: int) -> Optional[int]:
    """The shortest row length that forms the padded-row class `feat` (`launch_keys.softmax_feat_bucket`: every length
    of (feat/2, feat] forms it), or None when no shorter row does. Every configuration's tile, register budget and
    pruning are the padded row's, so it stands for the class as well as any."""
    from nbxtensor.ops.launch_keys import softmax_feat_bucket
    shortest = feat // 2 + 1 if feat > 1 else 1
    return shortest if softmax_feat_bucket(shortest) == feat else None


class Room:
    """The bytes a key may be drawn at on a discrete card: `card` the card's bytes, `usable` the share of it the caller
    plans against. Room() is unbounded (no card: a unified-memory device is bounded by the caller's own price)."""

    def __init__(self, card: Optional[int] = None, usable: Optional[int] = None):
        self.card = None if card is None else int(card)
        self.usable = self.card if usable is None else int(usable)

    def refuse(self, specs, what: str) -> None:
        """Refuse `specs` ((shape, dtype) of every operand AND the output) over the usable bytes."""
        if self.card is None:
            return
        asked = sum(int(np.prod(shape, dtype=np.int64)) * _isz.itemsize(_isz.from_autotune_key(dt))
                    for shape, dt in specs)
        if asked > self.usable:
            raise KeyTooLargeForClass(asked, self.card, what, usable=self.usable)

    def row_shape(self, rows: int, feat: int, in_dt: str, out_dt: str, what: str) -> Tuple[int, int]:
        """The (rows, row length) a padded-row key is drawn at: the class's top where in and out fit, else its shortest
        row (`shortest_row`); no length of the class fitting is refused."""
        if self.card is None:
            return rows, feat
        per = _isz.itemsize(_isz.from_autotune_key(in_dt)) + _isz.itemsize(_isz.from_autotune_key(out_dt))
        if rows * feat * per <= self.usable:
            return rows, feat
        shortest = shortest_row(feat)
        if shortest is None or rows * shortest * per > self.usable:
            raise KeyTooLargeForClass(rows * feat * per, self.card, what)
        return rows, int(shortest)

    def ew_count(self, top: int, dt: str, what: str) -> int:
        """The elements an element-wise class is drawn at: its top where the copy's in and out fit, else its floor
        (`autotune_bucket.class_floor`, the fewest that form it); a floor that does not fit is refused."""
        if self.card is None:
            return top
        from nbxtensor.ops.autotune_bucket import class_floor
        per = 2 * _isz.itemsize(_isz.from_autotune_key(dt))
        if top * per <= self.usable:
            return top
        floor = class_floor("elements", top)
        if floor * per > self.usable:
            raise KeyTooLargeForClass(floor * per, self.card, what)
        return floor


def gemm_rows(key: tuple, dts, unit_keyed: bool = False) -> int:
    """The row count a GEMM key is drawn at: the largest whose launch forms the key
    (`launch_keys.gemm_row_representative`), not the bucket's top — there the matrix unit may take the GEMM and the
    wrapper never reaches the autotuner. A key no row count forms is refused."""
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    a = _nbx(dts[0]) if dts and dts[0] in _HALF else NBXDtype.float16
    b = _nbx(dts[1]) if len(dts) > 1 and dts[1] in _HALF else a
    m = LK.gemm_row_representative(int(key[0]), a, b, unit_keyed)
    if m is None:
        raise ValueError(f"no row count forms the GEMM key {key}: the matrix unit takes every M of bucket {key[0]} "
                         f"— a census defect, the key should not exist")
    return m


# ---------------------------------------------------------------------------------------------------------------------
# what a key draws: its operands, read from the key's fields
# ---------------------------------------------------------------------------------------------------------------------
def _ops_mm(short, key, dts, out_dt, d):
    M, N, K = gemm_rows(key, dts, unit_keyed=True), int(key[1]), int(key[2])
    ins = [("a", (M, K), d(0)), ("b", (K, N), d(1))] + ([("bias", (N,), d(2))] if short == "addmm_kernel" else [])
    return {"family": "mm", "inputs": ins, "out": ((M, N), out_dt), "mnk": (M, N, K), "batch": 1,
            "row_windowed_launch": True}


def _ops_bmm(short, key, dts, out_dt, d):
    M, N, K = gemm_rows(key, dts), int(key[1]), int(key[2])
    B = _BMM_BATCH
    ins = [("a", (B, M, K), d(0)), ("b", (B, K, N), d(1))]
    if len(key) > 5 and isinstance(key[5], bool) and key[5]:
        ins.append(("bias", (B, M, N), d(3, d(0))))
    return {"family": "mm", "inputs": ins, "out": ((B, M, N), out_dt), "mnk": (M, N, K), "batch": B,
            "row_windowed_launch": False}


def _ops_flash(short, key, dts, out_dt, d):
    from nbxtensor.ops.attention_wrappers import flash_route_heads
    tq, tk, hd = int(key[0]), int(key[1]), int(key[2])
    B, H = 1, flash_route_heads(tq, tk, hd, d(0), d(1), d(2))
    ins = [("q", (B, H, tq, hd), d(0)), ("k", (B, H, tk, hd), d(1)), ("v", (B, H, tk, hd), d(2))]
    if d(3) != d(0):
        ins.append(("mask", (B, H, tq, tk), d(3)))
    return {"family": "attention", "inputs": ins, "out": ((B, H, tq, hd), d(0)), "mnk": (tq, hd, 2 * tk),
            "batch": B * H, "row_windowed_launch": True}


def _ops_rows(short, key, dts, out_dt, d):
    """softmax, row reductions, row norms and group norm: rows of the padded row."""
    rows, feat = int(key[0]), int(key[1])
    ins = [("x", (rows, feat), d(0))]
    if short in ("layer_norm_forward_kernel", "rms_norm_forward_kernel"):
        ins += [("w", (feat,), d(1))] if bool(key[2]) else []
        ins += [("b", (feat,), d(2))] if short == "layer_norm_forward_kernel" and bool(key[2]) and bool(key[3]) else []
    if short == "group_norm_forward_kernel":
        ins += ([("w", (rows,), d(2))] if bool(key[2]) else []) + ([("b", (rows,), d(3))] if bool(key[3]) else [])
    family = "rownorm" if "norm" in short else "softmax"
    out = (rows,) if short in ("mean_forward_kernel", "sum_forward_kernel", "amax_forward_kernel") else (rows, feat)
    return {"family": family, "inputs": ins, "out": (out, out_dt), "mnk": (rows, feat, 1), "batch": 1,
            "row_windowed_launch": True}


def _ops_ew(short, key, dts, out_dt, d):
    n = int(key[0])
    return {"family": "elementwise", "inputs": [("x", (n,), d(0))], "out": ((n,), out_dt), "mnk": (n, 1, 1),
            "batch": 1, "row_windowed_launch": False}


def _ops_gemv(short, key, dts, out_dt, d):
    N, K = int(key[0]), int(key[1])
    ins = [("mat", (N, K), d(0)), ("vec", (K,), d(1))] + ([("bias", (N,), d(2))] if short == "addmv_kernel" else [])
    return {"family": "gemv", "inputs": ins, "out": ((N,), out_dt), "mnk": (1, N, K), "batch": 1,
            "row_windowed_launch": False}


def _ops_moe(short, key, dts, out_dt, d):
    N, K, nv, n_exp = int(key[0]), int(key[1]), int(key[2]), int(key[3])
    return {"family": "moe", "inputs": [("x", (nv, K), d(0)), ("w", (n_exp * N, K), d(0)), ("score", (nv,), d(0)),
                                        ("route", (nv,), "int64")],
            "out": ((nv, N), d(0)), "mnk": (nv, N, K), "experts": n_exp}


def _ops_conv(short, key, dts, out_dt, d):
    (n, ci, h, w, co, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, groups) = [int(v) for v in key[:16]]
    ci_g = ci // max(groups, 1)
    return {"family": "conv", "inputs": [("x", (n, ci, h, w), d(0)), ("w", (co, ci_g, kh, kw), d(1))],
            "out": ((n, co, oh, ow), out_dt), "geometry": (n, ci, h, w, co, oh, ow, ci_g, kh, kw, ph, pw)}


def _ops_depthwise(short, key, dts, out_dt, d):
    (c, h, w, oh, ow, kh, kw, sh, sw, ph, pw) = [int(v) for v in key[:11]]
    return {"family": "conv", "inputs": [("x", (1, c, h, w), d(0)), ("w", (c, 1, kh, kw), d(1))],
            "out": ((1, c, oh, ow), out_dt), "geometry": (1, c, h, w, c, oh, ow, 1, kh, kw, ph, pw)}


_ROWS = ("softmax_forward_kernel", "softmax_forward_looped_kernel", "layer_norm_forward_kernel",
         "rms_norm_forward_kernel", "mean_forward_kernel", "sum_forward_kernel", "amax_forward_kernel",
         "group_norm_forward_kernel")
_KEY_OPERANDS = {"matmul_kernel": _ops_mm, "addmm_kernel": _ops_mm, "baddbmm_kernel": _ops_bmm,
                 "flash_attention_m8n8k4_kernel": _ops_flash, "flash_attention_certified_kernel": _ops_flash,
                 "ew_launch_kernel": _ops_ew, "gemv_vec_kernel": _ops_gemv, "addmv_kernel": _ops_gemv,
                 "fused_moe_kernel": _ops_moe, "conv2d_forward_kernel": _ops_conv,
                 "depthwise_conv2d_kernel": _ops_depthwise, **{k: _ops_rows for k in _ROWS}}


def key_operands(qual: str, tuner, key: tuple) -> Optional[Dict[str, Any]]:
    """The operands `synthesize` draws for `key`: {"family", "inputs": [(name, shape, dtype)], "out": (shape, dtype),
    and the geometry its reference is windowed on ("mnk", "batch", "row_windowed_launch"; "experts"; "geometry")}, read
    from the same key fields as the synthesis, so a price and the synthesis cannot disagree. None for a kernel this
    module does not price."""
    short = C.kernel_short(qual)
    ops = _KEY_OPERANDS.get(short)
    if ops is None:
        return None
    dts = C.key_dtypes(key)
    d = lambda i, default="fp16": (dts[i] if len(dts) > i else default)   # noqa: E731
    return ops(short, key, dts, C.output_dtype(tuner, key), d)


# ---------------------------------------------------------------------------------------------------------------------
# the synthesis, per kernel family
# ---------------------------------------------------------------------------------------------------------------------
class _Key:
    """One key being synthesised: its fields, its dtypes, the generator, the room, whether values are drawn."""

    def __init__(self, qual, key, rng, room, values, filed, out_dt):
        self.key, self.rng, self.room, self.values = key, rng, room, values
        self.filed, self.out_dt = filed, out_dt
        self.dts, self.short = C.key_dtypes(key), C.kernel_short(qual)

    def d(self, i: int, default):
        return self.dts[i] if len(self.dts) > i else default

    def arr(self, shape, dt):
        return (draw if self.values else spec)(self.rng, shape, dt)


def _mm(k):
    from nbxtensor.ops import linalg_wrappers as LW
    M, N, K = gemm_rows(k.key, k.dts, unit_keyed=True), int(k.key[1]), int(k.key[2])
    addmm = k.short == "addmm_kernel"
    k.room.refuse([((M, K), k.d(0, "fp16")), ((K, N), k.d(1, "fp16")),
                   ((N,), k.dts[2] if len(k.dts) > 2 and addmm else k.out_dt), ((M, N), k.out_dt)], k.short)
    a = k.arr((M, K), k.d(0, "fp16"))
    b = k.arr((K, N), k.d(1, "fp16"))
    if not addmm:
        return (lambda: LW.mm(upload(a), upload(b))), (F.matmul_oracle(a, b) if k.values else None), "c_ptr"
    bias = k.arr((N,), k.d(2, "fp16"))
    return ((lambda: LW.addmm(upload(bias), upload(a), upload(b))),
            (F.matmul_oracle(a, b, bias[None, :]) if k.values else None), "c_ptr")


def _ew(k):
    """The element-wise class's shared copy launched itself (no wrapper reaches it but `ew_meta`), filed under the
    class's top whatever is drawn; its reference is the draw itself (each element's result is its own)."""
    import triton
    from nbxtensor.ops.elementwise_launch import ew_launch_kernel
    from nbxtensor.tensor import NBXTensor
    top = int(k.filed[0])
    n = k.room.ew_count(int(k.key[0]), k.d(0, "fp32"), k.short)
    x = k.arr((n,), k.d(0, "fp32"))

    def run():
        xd = upload(x)
        y = NBXTensor.empty_like(xd)
        ew_launch_kernel[lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)](xd, y, n, top)
        return y
    return run, ((lambda: np.asarray(x, dtype=np.float64)) if k.values else None), "y_ptr"


def _gemv(k):
    from nbxtensor.ops import linalg_wrappers as LW
    N, K = int(k.key[0]), int(k.key[1])
    mat_dt, vec_dt = k.d(0, "fp16"), k.d(1, "fp32")
    addmv = k.short == "addmv_kernel"
    k.room.refuse([((N, K), mat_dt), ((K,), vec_dt), ((N,), k.out_dt)]
                  + ([((N,), k.d(2, k.out_dt))] if addmv else []), k.short)
    mat = k.arr((N, K), mat_dt)
    vec = k.arr((K,), vec_dt)
    if not addmv:
        return (lambda: LW.mv_wrapper(upload(mat), upload(vec))), (F.gemv_oracle(mat, vec) if k.values else None), "C_ptr"
    bias = k.arr((N,), k.d(2, k.out_dt))
    return ((lambda: LW.addmv_wrapper(upload(bias), upload(mat), upload(vec))),
            (F.gemv_oracle(mat, vec, bias) if k.values else None), "out_ptr")


def _bmm(k):
    from nbxtensor.ops import linalg_wrappers as LW
    M, N, K = gemm_rows(k.key, k.dts), int(k.key[1]), int(k.key[2])
    has_bias = bool(k.key[5]) if len(k.key) > 5 and isinstance(k.key[5], bool) else False
    B, bias_dt = _BMM_BATCH, k.d(3, k.d(0, "fp16"))
    k.room.refuse([((B, M, K), k.d(0, "fp16")), ((B, K, N), k.d(1, "fp16")), ((B, M, N), k.out_dt)]
                  + ([((B, M, N), bias_dt)] if has_bias else []), k.short)
    a = k.arr((B, M, K), k.d(0, "fp16"))
    b = k.arr((B, K, N), k.d(1, "fp16"))
    if not has_bias:
        return (lambda: LW.bmm(upload(a), upload(b))), (F.matmul_oracle(a, b) if k.values else None), "out_ptr"
    bias = k.arr((B, M, N), bias_dt)
    return ((lambda: LW.baddbmm_wrapper(upload(bias), upload(a), upload(b))),
            (F.matmul_oracle(a, b, bias) if k.values else None), "out_ptr")


def _flash(k):
    """Through the attention wrapper itself (the matrix-unit route it takes for these dtypes and head dim), at the
    buckets' top lengths and the fewest heads that take the flash; a mask only where the key's bias dtype is not Q's.
    K is drawn (B, H, tk, hd), never transposed, and said so: where tk == hd its layout cannot be read from its shape."""
    from nbxtensor.ops import attention_wrappers as AW
    tq, tk, hd = int(k.key[0]), int(k.key[1]), int(k.key[2])
    qd = k.d(0, "fp16")
    B, H = 1, AW.flash_route_heads(tq, tk, hd, qd, k.d(1, qd), k.d(2, qd))
    shapes = [((B, H, tq, hd), qd), ((B, H, tk, hd), k.d(1, qd)), ((B, H, tk, hd), k.d(2, qd)), ((B, H, tq, hd), qd)]
    masked = len(k.dts) > 3 and k.dts[3] != qd
    k.room.refuse(shapes + ([((B, H, tq, tk), k.dts[3])] if masked else []), k.short)
    q, kk, v = (k.arr(sh, dt) for sh, dt in shapes[:3])
    mask = k.arr((B, H, tq, tk), k.dts[3]) if masked else None
    scale = 1.0 / float(np.sqrt(hd))
    run = lambda: AW.scaled_dot_product_attention_wrapper(  # noqa: E731
        upload(q), upload(kk), upload(v), attn_mask=upload(mask) if masked else None, k_pre_transposed=False)
    return run, (F.attention_oracle(q, kk, v, mask, scale) if k.values else None), "Out"


def _row_reduce(k):
    from nbxtensor.ops import reduction_wrappers as RW
    x_dt = k.d(0, "fp32")
    rows, feat = k.room.row_shape(int(k.key[0]), int(k.key[1]), x_dt, x_dt, k.short)
    k.room.refuse([((rows, feat), x_dt), ((rows,), k.out_dt)], k.short)
    x = k.arr((rows, feat), x_dt)
    red = k.short.split("_")[0]
    fn = {"mean": RW.mean_wrapper, "sum": RW.sum_wrapper, "amax": RW.amax_wrapper}[red]
    return (lambda: fn(upload(x), -1)), (F.row_reduce_oracle(x, red) if k.values else None), "output_ptr"


def _align_reference(route, n_exp: int, nb: int, bs: int, stage: int) -> F.Exact:
    """The routing alignment's host sort, the reference its stages are met EXACTLY against."""
    counts = np.bincount(route, minlength=n_exp)
    padded = -(-counts // bs) * bs
    offsets = np.concatenate([[0], np.cumsum(padded)[:-1]])
    if stage == 1:
        return F.Exact(offsets)
    bound = min(nb * bs, nb + n_exp * (bs - 1))
    max_total = -(-bound // bs) * bs
    if stage == 2:
        starts = np.arange(max_total // bs) * bs
        own = (starts[:, None] >= offsets[None, :]) & (starts[:, None] < (offsets + padded)[None, :])
        return F.Exact(np.where(own.any(axis=1), own.argmax(axis=1), -1))
    ref = np.full(max_total, -1, dtype=np.int64)          # stage 3 writes the routed slots only (the poison)
    for e in range(n_exp):
        tok = np.nonzero(route == e)[0]
        ref[offsets[e]:offsets[e] + tok.size] = tok
    return F.Exact(ref)


def _align(k):
    """The class's (token, expert) pairs at its floor, routed uniformly over the key's experts, through the library's
    own `moe_align_block_size` (stage 3, whose key names no block, at the FMA kernel's block)."""
    from nbxtensor.ops.moe_wrappers import moe_align_block_size, MOE_BLOCK_M
    from nbxtensor.ops.autotune_bucket import class_floor
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.memory.allocator import DeviceAllocator
    stage = int(k.short[len("moe_align_stage")])
    key = k.key
    bs, n_exp, nb = (MOE_BLOCK_M, int(key[0]), int(key[1])) if stage == 3 else (int(key[0]), int(key[2]), int(key[3]))
    nb = class_floor("elements", nb)
    route = k.rng.integers(0, n_exp, size=nb).astype(np.int64)

    def run():
        sids, eids, _npad = moe_align_block_size(NBXTensor.from_numpy(route), bs, n_exp, DeviceAllocator.get_device())
        return eids if stage == 2 else sids
    oracle = (lambda: _align_reference(route, n_exp, nb, bs, stage)) if k.values else None
    return run, oracle, {1: "offsets_ptr", 2: "expert_ids_ptr", 3: "sorted_ids_ptr"}[stage]


_EXPERT_TABLES: List[Any] = []


def _expert_table(experts: List[Any]) -> Any:
    """The device-address table the MoE kernels read their experts' weights through, built as the runtime builds it
    (`memory.pointer_tables.build_ptr_tables`): on Metal a shader dereferences captured addresses held under a pinned
    scope, never a buffer's `data_ptr()`. The last few keys' tables are kept (their pins with them)."""
    from nbxtensor.memory.pointer_tables import build_ptr_tables
    from nbxtensor.memory.allocator import DeviceAllocator
    tables = build_ptr_tables(experts, experts, experts)
    _EXPERT_TABLES.append(tables)
    while len(_EXPERT_TABLES) > 8:
        _EXPERT_TABLES.pop(0).release_pins()
    return tables.gate_ptrs[DeviceAllocator.get_device()]


def _fused_moe_run(x, ws, score, route, w_dt, N, K, mul, divide):
    from nbxtensor.ops import moe_wrappers as MW
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.memory.allocator import DeviceAllocator
    nv, n_exp = int(x.shape[0]), len(ws)
    ptrs = _expert_table([upload(w) for w in ws])
    sids, eids, npad = MW.moe_align_block_size(NBXTensor.from_numpy(route), MW.MOE_BLOCK_M, n_exp,
                                               DeviceAllocator.get_device())
    out = NBXTensor.zeros((nv, N), dtype=_nbx(w_dt))
    MW.invoke_fused_moe(upload(x), ptrs, out, upload(score), sids, eids, npad, N, K, 1, K, 1,
                        mul_routed_weight=mul, topk_divide=divide)
    return out


def _fused_moe(k):
    """The bucket's valid (token, expert) pairs at its top, one expert per token, the key's experts of [N, K] weights
    routed uniformly at random and aligned by the library's own `moe_align_block_size`, through `invoke_fused_moe`."""
    N, K, nv, n_exp = int(k.key[0]), int(k.key[1]), int(k.key[2]), int(k.key[3])
    mul, divide = bool(k.key[4]), bool(k.key[5])
    w_dt = k.d(0, "fp16")
    k.room.refuse([((nv, K), w_dt), ((n_exp * N, K), w_dt), ((nv, N), w_dt)], k.short)
    x = k.arr((nv, K), w_dt)
    ws = [k.arr((N, K), w_dt) for _ in range(n_exp)]
    score = k.arr((nv,), w_dt)
    route = k.rng.integers(0, n_exp, size=nv).astype(np.int64)
    return ((lambda: _fused_moe_run(x, ws, score, route, w_dt, N, K, mul, divide)),
            (F.moe_oracle(x, ws, score, route, mul) if k.values else None), "c_ptr")


def _moe_part_reduce(k):
    import triton
    from nbxtensor.ops import moe_decode_vec as MV
    from nbxtensor.tensor import NBXTensor
    n_out, top_k = int(k.key[0]), int(k.key[1])
    part = k.arr((top_k, n_out), "fp32")

    def run():
        out = NBXTensor.empty((n_out,), dtype=_nbx("fp32"))
        MV.moe_part_reduce_kernel[lambda META: (triton.cdiv(n_out, META["BLOCK_N"]),)](upload(part), out, n_out,
                                                                                       TOP_K=top_k)
        return out
    return run, ((lambda: F.f64(part).sum(axis=0)) if k.values else None), "out_ptr"


def _moe_vec_run(gate, x, ids, rw, ws, k_in, n_out, top_k, bf):
    import triton
    from nbxtensor.ops import moe_decode_vec as MV
    from nbxtensor.tensor import NBXTensor
    tabs = [_expert_table(proj) for proj in [[upload(w) for w in proj] for proj in ws]]
    out = NBXTensor.empty((top_k, n_out), dtype=_nbx("fp32"))
    grid = lambda META: (top_k, triton.cdiv(n_out, META["BLOCK_N"]))     # noqa: E731
    if gate:
        MV.moe_gateup_vec_half_kernel[grid](upload(x), upload(ids), tabs[0], tabs[1], out, k_in, n_out, 1, k_in,
                                            top_k, W_BF16=bf)
    else:
        MV.moe_down_split_vec_half_kernel[grid](upload(x), upload(ids), upload(rw), tabs[0], out, k_in, n_out, 1, k_in,
                                                top_k, W_BF16=bf)
    return out


def _moe_vec_reference(gate, x, ids, rw, ws, n_out, top_k):
    y = np.empty((top_k, n_out), dtype=np.float64)
    for p_, e in enumerate(ids):
        if gate:
            g, u = F.f64(x) @ F.f64(ws[0][e]).T, F.f64(x) @ F.f64(ws[1][e]).T
            y[p_] = g / (1.0 + np.exp(-g)) * u
        else:
            y[p_] = (F.f64(x)[p_] @ F.f64(ws[0][e]).T) * F.f64(rw)[p_]
    return y


def _moe_vec_half(k):
    """The one-token MoE band (`moe_wrappers.moe_decode_vec_half_pass`), each launch on its own: TOP_K experts (a
    permutation of as many) of [N, K] half weights, the routed rows fp32."""
    k_in, n_out, top_k, bf = int(k.key[0]), int(k.key[1]), int(k.key[2]), bool(k.key[3])
    w_dt = "bf16" if bf else "fp16"
    gate = k.short == "moe_gateup_vec_half_kernel"
    k.room.refuse([((top_k * (2 if gate else 1) * n_out, k_in), w_dt), ((top_k, n_out), "fp32")], k.short)
    ids = k.rng.permutation(top_k).astype(np.int64)
    x = k.arr((k_in,) if gate else (top_k, k_in), w_dt if gate else "fp32")
    rw = k.arr((top_k,), w_dt)
    ws = [[k.arr((n_out, k_in), w_dt) for _ in range(top_k)] for _ in range(2 if gate else 1)]
    oracle = (lambda: _moe_vec_reference(gate, x, ids, rw, ws, n_out, top_k)) if k.values else None
    return ((lambda: _moe_vec_run(gate, x, ids, rw, ws, k_in, n_out, top_k, bf)), oracle,
            ("h_ptr" if gate else "part_ptr"))


def _rounded_to(qs, dt: str) -> np.ndarray:
    """Q rounded to the cache's dtype on load, as the decode kernels do."""
    if dt == "fp16":
        return qs.astype(np.float16).astype(np.float64)
    if dt == "bf16":
        return bf16_bits_to_f32(f32_to_bf16_bits(qs.astype(np.float32))).astype(np.float64)
    return qs.astype(np.float32).astype(np.float64)


def _split_partials(sc, vs, T: int, split: int, seg: int) -> np.ndarray:
    """Each split's unnormalised partial output [G, split, Dv]: exp(scores - the split's max) @ its values."""
    out = np.zeros((sc.shape[0], split, vs.shape[1]), dtype=np.float64)
    for s_ in range(split):
        lo, hi = s_ * seg, min((s_ + 1) * seg, T)
        if lo >= hi:
            continue
        blk = sc[:, lo:hi]
        out[:, s_] = np.exp(blk - blk.max(axis=1, keepdims=True)) @ vs[lo:hi]
    return out


def _decode_vec_reference(q, kk, v, b, q_round, k_dt, T, D, Dv, G, scale):
    from nbxtensor.ops.decode_attn_vec import decode_vec_splits
    split = decode_vec_splits(T)
    qs, ks, vs = F.f64(q).reshape(G, D), F.f64(kk).reshape(T, D), F.f64(v).reshape(T, Dv)
    if q_round is not None:
        qs = _rounded_to(qs, k_dt)
    sc = qs @ ks.T * scale + (F.f64(b)[None, :] if b is not None else 0.0)
    return _split_partials(sc, vs, T, split, -(-T // split))


def _decode_vec(k):
    """One KV head and its GQA group of query rows over the bucket's top of keys, through `_decode_attn_vec` (its
    split, the reduce after); the reference is each split's unnormalised partial output (`opart`)."""
    import math
    from nbxtensor.ops import attention_wrappers as AW
    T, D, Dv, G, has_b = int(k.key[0]), int(k.key[1]), int(k.key[2]), int(k.key[3]), bool(k.key[4])
    q_dt, k_dt, v_dt = (k.dts + ["fp16"] * 3)[:3]
    b_dt = k.dts[3] if has_b and len(k.dts) > 3 else None
    k.room.refuse([((T, D), k_dt), ((T, Dv), v_dt)], k.short)
    q, kk, v = k.arr((1, G, 1, D), q_dt), k.arr((1, 1, T, D), k_dt), k.arr((1, 1, T, Dv), v_dt)
    b = k.arr((T,), b_dt) if has_b else None
    scale = 1.0 / math.sqrt(D)
    q_round = None if q_dt == k_dt else _nbx(k_dt)
    run = lambda: AW._decode_attn_vec(upload(q), upload(kk), upload(v), upload(b) if has_b else None,  # noqa: E731
                                      scale, 1, G, 1, T, D, q_round=q_round)
    oracle = (lambda: _decode_vec_reference(q, kk, v, b, q_round, k_dt, T, D, Dv, G, scale)) if k.values else None
    return run, oracle, "opart_ptr"


def _reduce_reference(opart, mpart, lpart, rows, dv):
    """The log-sum-exp combine of the splits' partial outputs, running maxima and sums (a split that saw only masked
    keys has max -inf and sum 0)."""
    o, m, l_ = F.f64(opart), F.f64(mpart), F.f64(lpart)
    top = np.max(np.where(np.isfinite(m), m, -np.inf), axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    c = np.where(np.isfinite(m), np.exp(m - top), 0.0)
    den = (l_ * c).sum(axis=1)
    num = (o * c[:, :, None]).sum(axis=1)
    return (num / np.where(den == 0.0, 1.0, den)[:, None]).reshape(rows, 1, dv)


def _decode_reduce(k):
    """The combine after both split decodes, through `decode_reduce`: a batch of query rows, each split's partial
    output, running max and sum of exponentials drawn, some splits dead."""
    from nbxtensor.ops import attention_wrappers as AW
    from nbxtensor.tensor import NBXTensor
    split, groups, dv = int(k.key[0]), int(k.key[1]), int(k.key[2])
    if groups != 1:
        raise ValueError(f"no launch forms the reduce key {k.key}: every decode views its partials at GROUPS 1")
    out_dt, rows = k.d(3, "fp16"), 64
    opart, mpart, lpart = k.arr((rows, split, dv), "fp32"), k.arr((rows, split), "fp32"), k.arr((rows, split), "fp32")
    if k.values:
        mpart, lpart = np.array(mpart, dtype=np.float32) * 10.0, np.abs(np.array(lpart, dtype=np.float32)) * 10.0 + 0.5
        dead = k.rng.random((rows, split)) < (0.1 if split > 1 else 0.0)
        mpart[dead], lpart[dead] = -np.inf, 0.0

    def run():
        out = NBXTensor.empty((rows, 1, dv), dtype=_nbx(out_dt))
        AW.decode_reduce(upload(opart), upload(mpart), upload(lpart), out)
        return out
    return run, ((lambda: _reduce_reference(opart, mpart, lpart, rows, dv)) if k.values else None), "out_ptr"


def _paged_reference(q, kp, vp, order, q_dt, kv_dt, T, G, D, scale):
    from nbxtensor.ops import paged_decode_attn as PD
    split = PD.paged_decode_splits(T)
    seg = -(-(-(-T // split)) // PD.PAGED_DECODE_SPLIT_ALIGN) * PD.PAGED_DECODE_SPLIT_ALIGN
    pages = len(order)
    ks = np.concatenate([F.f64(kp)[order[i], 0] for i in range(pages)])[:T]
    vs = np.concatenate([F.f64(vp)[order[i], 0] for i in range(pages)])[:T]
    qs = F.f64(q).reshape(G, D)
    if q_dt != kv_dt:
        qs = _rounded_to(qs, kv_dt)
    return _split_partials(qs @ ks.T * scale, vs, T, split, seg)


def _paged(k):
    """One sequence of the class's top (its splits' positions) over shuffled pages, one KV head and its group, through
    `paged_decode_attention`; the reference is each split's unnormalised partial output, the split the kernel cuts."""
    import math
    from nbxtensor.ops import attention_wrappers as AW
    from nbxtensor.ops import paged_decode_attn as PD
    from nbxtensor.tensor import NBXTensor
    splits, pt, D, Dv, G = (int(x) for x in k.key[:5])
    T = splits * PD.PAGED_DECODE_SPLIT_POSITIONS
    if PD.paged_decode_splits(T) != splits:
        raise ValueError(f"no length forms the paged decode key {k.key}: {splits} splits is not a split count")
    q_dt, kv_dt = (k.dts + ["fp16"] * 2)[:2]
    pages = -(-T // pt)
    k.room.refuse([((pages * pt, D), kv_dt), ((pages * pt, Dv), kv_dt)], k.short)
    q = k.arr((1, G, 1, D), q_dt)
    kp, vp = k.arr((pages, 1, pt, D), kv_dt), k.arr((pages, 1, pt, Dv), kv_dt)
    order = k.rng.permutation(pages).astype(np.int32)
    scale = 1.0 / math.sqrt(D)
    run = lambda: AW.paged_decode_attention(  # noqa: E731
        upload(q), upload(kp), upload(vp), NBXTensor.from_numpy(order.reshape(1, pages)),
        NBXTensor.from_numpy(np.array([T], dtype=np.int32)), scale, T, q_round=_nbx(kv_dt))
    oracle = (lambda: _paged_reference(q, kp, vp, order, q_dt, kv_dt, T, G, D, scale)) if k.values else None
    return run, oracle, "opart_ptr"


def _group_norm(k):
    """N = 1, one channel per group, the groups the programs' bucket, the group's extent the padded class; the draw is
    uploaded once and the host keeps only the rows the reference reads (`oracles.fp64.oracle_rows`)."""
    from nbxtensor.ops import norm_wrappers as NW
    x_dt = k.d(0, "fp32")
    rows, feat = k.room.row_shape(int(k.key[0]), int(k.key[1]), x_dt, x_dt, k.short)
    has_w, has_b = bool(k.key[2]), bool(k.key[3])
    k.room.refuse([((rows, feat), x_dt), ((rows, feat), x_dt)], k.short)
    x = k.arr((rows, feat), x_dt)
    w = k.arr((rows,), k.dts[2]) if has_w else None
    b = k.arr((rows,), k.dts[3]) if has_b else None
    xd, oracle = upload(x), (F.groupnorm_oracle(F.oracle_rows(x), w, b) if k.values else None)
    del x
    run = lambda: NW.group_norm_wrapper(xd.view(1, rows, feat), rows, upload(w) if has_w else None,  # noqa: E731
                                        upload(b) if has_b else None)[0].view(rows, feat)
    return run, oracle, "Y"


def _row_norm(k):
    """The row norms through their wrappers (`native_layer_norm`, `rms_norm`) at the wrappers' own epsilon."""
    from nbxtensor.ops import norm_wrappers as NW
    rows, feat, has_w = int(k.key[0]), int(k.key[1]), bool(k.key[2])
    ln = k.short == "layer_norm_forward_kernel"
    has_b = ln and has_w and bool(k.key[3])
    x_dt = k.d(0, "fp32")
    k.room.refuse([((rows, feat), x_dt), ((rows, feat), k.out_dt)], k.short)
    x = k.arr((rows, feat), x_dt)
    w = k.arr((feat,), k.dts[1]) if has_w else None
    b = k.arr((feat,), k.dts[2]) if has_b else None
    xd, oracle = upload(x), (F.rownorm_oracle(F.oracle_rows(x), w, b, ln) if k.values else None)
    del x
    if ln:
        return (lambda: NW.native_layer_norm(xd, [feat], upload(w) if has_w else None,
                                             upload(b) if has_b else None)[0]), oracle, "output_ptr"
    return (lambda: NW.rms_norm(xd, upload(w) if has_w else None)), oracle, "output_ptr"


def _softmax(k):
    from nbxtensor.ops import attention_wrappers as AW
    log = bool(k.key[2])
    rows, feat = k.room.row_shape(int(k.key[0]), int(k.key[1]), k.d(0, "fp32"), k.out_dt, k.short)
    k.room.refuse([((rows, feat), k.d(0, "fp32")), ((rows, feat), k.out_dt)], k.short)
    x = k.arr((rows, feat), k.d(0, "fp32"))
    fn = AW.log_softmax if log else AW.softmax
    xd, oracle = upload(x), (F.softmax_oracle(F.oracle_rows(x), log) if k.values else None)
    del x
    return (lambda: fn(xd, -1)), oracle, "output_ptr"


def _conv(k):
    from nbxtensor.ops import conv_wrappers as CW
    (n, ci, h, w, co, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, groups) = [int(v) for v in k.key[:16]]
    x_dt, w_dt = k.d(0, "fp16"), k.d(1, "fp16")
    k.room.refuse([((n, ci, h, w), x_dt), ((co, ci // max(groups, 1), kh, kw), w_dt), ((n, co, oh, ow), k.out_dt)],
                  k.short)
    x = k.arr((n, ci, h, w), x_dt)
    wt = k.arr((co, ci // max(groups, 1), kh, kw), w_dt)
    run = lambda: CW.conv2d_wrapper(upload(x), _upload_conv_weight(wt, groups), None, (sh, sw), (ph, pw),  # noqa: E731
                                    (dh, dw), False, 0, groups)
    return run, (F.conv_oracle(x, wt, (sh, sw), (ph, pw), (dh, dw), groups) if k.values else None), "output_pointer"


def _depthwise(k):
    from nbxtensor.ops import conv_wrappers as CW
    (c, h, w, oh, ow, kh, kw, sh, sw, ph, pw) = [int(v) for v in k.key[:11]]
    x_dt, w_dt = k.d(0, "fp16"), k.d(1, "fp16")
    k.room.refuse([((1, c, h, w), x_dt), ((c, 1, kh, kw), w_dt), ((1, c, oh, ow), k.out_dt)], k.short)
    x = k.arr((1, c, h, w), x_dt)
    wt = k.arr((c, 1, kh, kw), w_dt)
    run = lambda: CW.conv2d_wrapper(upload(x), upload(wt), None, (sh, sw), (ph, pw), (1, 1), False, 0, c)  # noqa: E731
    return run, (F.conv_oracle(x, wt, (sh, sw), (ph, pw), (1, 1), c) if k.values else None), "out_ptr"


_FAMILIES = {"matmul_kernel": _mm, "addmm_kernel": _mm, "ew_launch_kernel": _ew, "gemv_vec_kernel": _gemv,
             "addmv_kernel": _gemv, "baddbmm_kernel": _bmm, "flash_attention_m8n8k4_kernel": _flash,
             "flash_attention_certified_kernel": _flash, "mean_forward_kernel": _row_reduce,
             "sum_forward_kernel": _row_reduce, "amax_forward_kernel": _row_reduce,
             "moe_align_stage1_kernel": _align, "moe_align_stage2_kernel": _align, "moe_align_stage3_kernel": _align,
             "fused_moe_kernel": _fused_moe, "moe_part_reduce_kernel": _moe_part_reduce,
             "moe_gateup_vec_half_kernel": _moe_vec_half, "moe_down_split_vec_half_kernel": _moe_vec_half,
             "decode_attn_vec_split_kernel": _decode_vec, "flash_decode_reduce_kernel": _decode_reduce,
             "paged_decode_attn_split_kernel": _paged, "group_norm_forward_kernel": _group_norm,
             "layer_norm_forward_kernel": _row_norm, "rms_norm_forward_kernel": _row_norm,
             "softmax_forward_kernel": _softmax, "softmax_forward_looped_kernel": _softmax,
             "conv2d_forward_kernel": _conv, "depthwise_conv2d_kernel": _depthwise}


def synthesize(qual: str, tuner, key: tuple, rng, room: Optional[Room] = None, values: bool = True,
               filed_key: Optional[tuple] = None,
               out_dt: Optional[str] = None) -> Optional[Tuple[Callable[[], Any], Optional[Callable[[], Any]], str]]:
    """(call, oracle_fn, output argument name) for `key` of the kernel `qual`, or None when this module has no
    synthesis for it. `call` runs the kernel's wrapper on operands drawn from `rng` at the key's shapes and dtypes;
    `oracle_fn` computes their float64 reference LAZILY (only once the wrapper's key is known to be the one asked for).

    `room`: the capacity a discrete card's key is drawn within (`Room`; unbounded by default). `filed_key`: the key the
    result is filed under when `key` is the size it is drawn at (an element-wise class drawn at its floor). `out_dt`:
    the output's dtype where the caller sizes the key (`autotune_certified.output_dtype`). `values=False`: the
    arguments only — each operand allocated on the device with the key's shape and dtype (so its strides, alignment and
    integer specialisation are the real key's), nothing drawn on the host, no reference (None)."""
    family = _FAMILIES.get(C.kernel_short(qual))
    if family is None:
        return None
    return family(_Key(qual, key, rng, room or Room(), values, filed_key or key, out_dt))


# ---------------------------------------------------------------------------------------------------------------------
# the launch under test: every candidate configuration against the reference, then the clock
# ---------------------------------------------------------------------------------------------------------------------
def launch_oracle(qual: str, tuner, meta, out_tensor) -> Optional[F.RowWindowedOracle]:
    """The float64 reference of THE LAUNCH THE AUTOTUNER SEES, from the kernel's own named operands
    (`screen_oracle.windowed_reference`, row-windowed on the launch's own rows) — or None for a launch this does not
    cover (a kernel the screen does not row-window, or an output that is not a 2-D matrix), whose synthesised reference
    then stands. A product the wrapper streams in bands is keyed on its whole shape: each band is proven against its
    own operands, never against windows cut from one whole-shape reference."""
    from nbxtensor.ops import screen_oracle as _so
    if C.kernel_short(qual) not in _so.ROW_WINDOWABLE:
        return None
    try:
        shape = tuple(int(x) for x in out_tensor.shape)
    except Exception:                                    # noqa: BLE001
        return None
    if len(shape) != 2:
        return None
    m, n = shape
    named = {**dict(getattr(tuner, "nargs", None) or {}), **dict(meta or {})}
    k = int(named.get("K", 0) or 0)
    blocks = []
    for r0, r1 in (F.row_windows(m, n, k) if k else None) or [(0, m)]:
        ref = _so.windowed_reference(tuner, meta, (r0, r1))
        if ref is None:
            raise RuntimeError(f"{qual}: the launch's operands could not be read for the oracle (rows {r0}-{r1} of {m})")
        blocks.append(((r0, r1), ref))
    return F.RowWindowedOracle(blocks, m)


def _sync() -> None:
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.stream_synchronize(0)


def launch_and_measure(tuner, args, kwargs, cfg, out, oracle):
    """(the launch's return, its deviation): `args` launched at configuration `cfg`, synchronised, `out` measured
    against `oracle` (a later launch of a key inside one wrapper call, run at the configuration already chosen)."""
    ret = tuner.fn.run(*args, **{**kwargs, **cfg.all_kwargs()})
    _sync()
    return ret, F.deviation_against(out, oracle)


class LaunchUnderTest:
    """One intercepted launch of an autotuned kernel, as the autotuner receives it (`args`, `kwargs`), whose output is
    the argument `out_name`. Every run poisons the output first — every byte 0xFF, NaN in every float dtype, on the
    device — so a configuration that writes only part of it cannot inherit its predecessor's correct values."""

    def __init__(self, tuner, args, kwargs, out_name: str, label: str):
        from nbxtensor.backend import launcher as L
        idx = next((i for i, n in enumerate(tuner.arg_names) if n == out_name), None)
        if idx is None or idx >= len(args):
            raise RuntimeError(f"{label}: output argument {out_name!r} not among the kernel's arguments")
        if L._writable_buffers(args) is None:
            raise RuntimeError(f"{label}: a strided view among the arguments — not certifiable")
        self.tuner, self.args, self.kwargs, self.out = tuner, args, kwargs, args[idx]
        self._span = (int(self.out.data_ptr()), int(self.out.nbytes()))

    def poison(self) -> None:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.memset_cuda(self._span[0], 0xFF, self._span[1])

    def launch(self, cfg):
        return self.tuner.fn.run(*self.args, **{**self.kwargs, **cfg.all_kwargs()})

    def final(self, cfg):
        """The launch the wrapper's call returns: the chosen configuration, on a poisoned output."""
        self.poison()
        return self.launch(cfg)

    def time(self, cfg, bench) -> float:
        return float(bench(lambda: self.launch(cfg)))

    def sweep(self, configs, oracle, tolerance: float):
        """Every configuration run once, measured, its run timed: (accepted [(cfg, deviation, run_s)], excluded
        [{config, deviation, tolerance}] beyond `tolerance`, unrun [{config, error}] the backend refused)."""
        results, excluded, unrun = [], [], []
        for cfg in configs:
            self.poison()
            try:
                t_one = time.time()
                self.launch(cfg)
                _sync()
                run_s = time.time() - t_one
                dev = F.deviation_against(self.out, oracle)
            except Exception as exc:                     # a config the backend refuses: counted, never trusted
                unrun.append({"config": config_dict(cfg), "error": str(exc)[:200]})
                continue
            if not (dev <= tolerance):
                excluded.append({"config": config_dict(cfg), "deviation": dev, "tolerance": tolerance})
            else:
                results.append((cfg, dev, run_s))
        return results, excluded, unrun

    def sweep_units(self, runs, oracle, tolerance: float, bench, excluded: list, unrun: list) -> list:
        """The matrix unit's tiles (`unit_runs`), each proven against the same reference and timed with the same bench:
        [(tile, deviation, ms)]; a refused or diverging tile is appended to `unrun` / `excluded`."""
        units = []
        for tile, run_unit in runs:
            self.poison()
            try:
                run_unit()
                _sync()
                udev = F.deviation_against(self.out, oracle)
            except Exception as exc:                 # a tile the backend refuses: counted, never trusted
                unrun.append({"unit_tile": tile, "error": str(exc)[:200]})
                continue
            if not (udev <= tolerance):
                excluded.append({"unit_tile": tile, "deviation": udev, "tolerance": tolerance})
                continue
            units.append((tile, udev, float(bench(run_unit))))
        return units


def fused_moe_unit_runs(tuner, args, kwargs) -> List[Tuple[Dict[str, Any], Any]]:
    """(tile, run) for each `matrix_unit.mm` tile of the profile for a `fused_moe_kernel` launch on experts of the
    unit's operand dtype: the routing the launch's alignment holds, read back and aligned again at the tile's rows,
    each run writing the launch's output through `invoke_fused_moe_m8n8k4`. Empty for any other launch."""
    if getattr(getattr(tuner, "base_fn", None), "__name__", None) != "fused_moe_kernel":
        return []
    from nbxtensor.ops import moe_wrappers as MW
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.memory.allocator import DeviceAllocator
    named = dict(zip(tuner.arg_names, args))
    a = named["a_ptr"]
    tiles = MW.fused_moe_unit_tiles(a)
    if not tiles:
        return []
    route_t = NBXTensor.from_numpy(_routing_of(named))
    out = []
    for tile in tiles:
        s_, e_, n_ = MW.moe_align_block_size(route_t, tile["block_m"], int(named["NUM_EXPERTS"]),
                                             DeviceAllocator.get_device())
        out.append((tile, (lambda t=tile, s_=s_, e_=e_, n_=n_: MW.invoke_fused_moe_m8n8k4(
            a, named["expert_ptrs_ptr"], named["c_ptr"], named["topk_weights_ptr"], s_, e_, n_, int(named["N"]),
            int(named["K"]), int(named["stride_bk"]), int(named["stride_bn"]), int(kwargs["top_k"]), t,
            mul_routed_weight=bool(kwargs["MUL_ROUTED_WEIGHT"]), topk_divide=bool(kwargs.get("TOPK_DIVIDE", True))))))
    return out


def _routing_of(named) -> np.ndarray:
    """Each valid pair's expert, read back from a fused MoE launch's alignment (its blocks of MOE_BLOCK_M pairs)."""
    from nbxtensor.ops.moe_wrappers import MOE_BLOCK_M
    nv = int(named["num_valid_tokens"])
    sids = np.asarray(named["sorted_token_ids_ptr"].numpy()).ravel()
    eids = np.asarray(named["expert_ids_ptr"].numpy()).ravel()
    npad = int(np.asarray(named["num_tokens_post_padded_ptr"].numpy()).ravel()[0])
    route = np.zeros(nv, dtype=np.int64)
    for b in range(npad // MOE_BLOCK_M):
        pairs = sids[b * MOE_BLOCK_M:(b + 1) * MOE_BLOCK_M]
        route[pairs[pairs < nv]] = int(eids[b])
    return route


def unit_runs(tuner, args, kwargs) -> List[Tuple[Dict[str, Any], Any]]:
    """(tile, run) for every matrix-unit candidate of this launch: a GEMM's (`linalg_wrappers.matrix_unit_runs`) and a
    fused MoE's (`fused_moe_unit_runs`); empty when the unit cannot take it."""
    from nbxtensor.ops.linalg_wrappers import matrix_unit_runs
    return matrix_unit_runs(tuner, args, kwargs) + fused_moe_unit_runs(tuner, args, kwargs)


def time_launch(fn, warmup: int, rep: int) -> float:
    """`fn` timed by the sweep's timer (`launcher.do_bench`, warm-up and repetitions in milliseconds)."""
    from nbxtensor.backend import launcher as L
    return float(L.do_bench(fn, warmup=warmup, rep=rep))


def witness_ms(M: int, N: int, K: int, dtype: str, warmup: int, rep: int) -> float:
    """A fixed-shape matmul timed by the sweep's own timer (`launcher.do_bench`): same shape, same inputs (a fixed
    seed), same timer every time, so a change in it is a change in the machine's regime."""
    from nbxtensor.backend import launcher as L
    from nbxtensor.ops import linalg_wrappers as LW
    from nbxtensor.tensor import NBXTensor
    npd = NP_DTYPES.get(dtype, np.float16)
    rng = np.random.default_rng(0)
    a = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((M, K)) * 0.1).astype(npd)))
    b = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((K, N)) * 0.1).astype(npd)))
    return float(L.do_bench(lambda: LW.mm(a, b), warmup=warmup, rep=rep))


def candidate_configs(tuner, kwargs) -> list:
    """The autotuner's own candidates for this launch, pruned by Triton's own pruning (the library's screen is not
    applied: the certification measures every candidate itself)."""
    from triton.runtime.autotuner import Autotuner
    upstream = getattr(Autotuner.prune_configs, "_nbx_upstream", Autotuner.prune_configs)
    return list(upstream(tuner, kwargs))


def stored_config(d: Dict[str, Any]):
    """A configuration as a certified directory records it, back as a Triton config."""
    from nbxtensor.ops.autotune_cache import config_from_dict
    return config_from_dict(d)


def config_dict(cfg) -> Dict[str, Any]:
    """A Triton config as a certified directory records it."""
    from nbxtensor.ops.autotune_cache import config_to_dict
    return config_to_dict(cfg)
