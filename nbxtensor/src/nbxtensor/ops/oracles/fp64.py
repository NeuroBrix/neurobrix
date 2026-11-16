"""The kernel families' float64 references for a certification, whole or on windows of their output, and a kernel's
output measured against one.

A reference must not share an implementation with the kernels it checks: numpy in float64, no Triton, no torch. A
reference is computed WHOLE while it costs at most `ORACLE_MAX_MACS` multiply-adds (or, for the row-wise families, that
many bytes of float64), and on WINDOWS of the output above it: the first, middle and last rows of a product or a row
op, the corners and the centre of a convolution's plane on its first, middle and last batch element. A kernel
configuration's arithmetic is the same at every output position (one tile shape, one accumulation order) and a tiling or
boundary fault reaches an edge, so a window carries the same per-element bound at a fraction of the cost (a whole
44 544 x 3 072 x 8 192 product in float64 held 155 GB of host memory and eighty cores, 2026-09-21).

An operand is read through `values`: a bf16 operand is carried as its bits (numpy has no bfloat16) and decoded exactly
to float32 here. The deviation (`oracle_deviation`) is max |out - ref| over the reference's own magnitude, measured on
the windows only, and a windowed reference reads only its windows off the device (`deviation_against`).

Moved from the NeuroBrix certifier (2026-11-12), which keeps which keys, at which tolerance, under which protocol.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from nbxtensor.dtype.convert import bf16_bits_to_f32

__all__ = ["ORACLE_MAX_MACS", "ROW_F64_ARRAYS", "values", "f64", "row_windows", "conv_windows", "gemv_rows",
           "Exact", "WindowedOracle", "RowWindowedOracle", "HostWindows", "oracle_rows", "conv_out_hw", "conv2d_oracle",
           "matmul_oracle", "moe_oracle", "attention_oracle", "softmax_oracle", "gemv_oracle", "rownorm_oracle",
           "row_reduce_oracle", "groupnorm_oracle", "conv_oracle", "head_windows", "HeadWindowedOracle",
           "delta_rule_oracle", "moe_route_oracle", "host_values", "materialise", "deviation_against",
           "causal_conv1d_oracle",
           "oracle_deviation"]

#: The cost under which a reference is computed whole: multiply-adds for a product or a convolution, bytes of float64
#: for a row-wise family (read at call time: a caller may lower it).
ORACLE_MAX_MACS = 2_000_000_000

#: The full-size float64 arrays a row-wise reference holds per element of its window (the row, its shifted or centred
#: values, the result): a softmax does a few operations per element, so it is windowed by its BYTES, never as one
#: multiply-add (priced at one, a 131 072 x 4 096 key computed 17 GB of float64 whole, 2026-10-21).
ROW_F64_ARRAYS = 3


def values(a) -> np.ndarray:
    """An operand's NUMBERS as a plain array: a bf16 operand (tagged `key_dtype == "bf16"`, its bits in a uint16 array)
    decoded exactly to float32, any other operand as it is."""
    if getattr(a, "key_dtype", None) == "bf16":
        return bf16_bits_to_f32(np.asarray(a))
    return np.asarray(a)


def f64(a) -> np.ndarray:
    """An operand's numbers in float64."""
    return values(a).astype(np.float64)


def _distinct(wins: list) -> list:
    seen, out = set(), []
    for w in wins:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def row_windows(m: int, n: int, k: int, cap: Optional[int] = None) -> Optional[List[Tuple[int, int]]]:
    """None when an m x n x k product fits the cap (`ORACLE_MAX_MACS` by default); else the row windows its reference
    is computed on — the first rows, the middle rows, the last rows — each of at most cap/3 multiply-adds."""
    cap = ORACLE_MAX_MACS if cap is None else cap
    if m * n * k <= cap:
        return None
    rows = max(1, min(m, int(cap / 3 // max(1, n * k))))
    return _distinct([(0, rows), ((m - rows) // 2, (m - rows) // 2 + rows), (m - rows, m)])


def conv_windows(n, oh, ow, ci_g, co, kh, kw) -> Optional[list]:
    """None when a convolution's whole reference fits `ORACLE_MAX_MACS`; else the (batch, r0, r1, c0, c1) windows of
    its output — the top-left corner of the first batch element, the centre of the middle one, the bottom-right corner
    of the last — each of at most cap/3 multiply-adds."""
    cap = ORACLE_MAX_MACS
    per_position = ci_g * co * kh * kw
    if n * oh * ow * per_position <= cap:
        return None
    budget = max(1, int(cap / 3 // per_position))
    side = max(1, int(budget ** 0.5))
    rh, rw = min(oh, side), min(ow, side)
    if rh * rw > budget:
        rw = max(1, budget // rh)
    return _distinct([(0, 0, rh, 0, rw),
                      (n // 2, (oh - rh) // 2, (oh - rh) // 2 + rh, (ow - rw) // 2, (ow - rw) // 2 + rw),
                      (n - 1, oh - rh, oh, ow - rw, ow)])


def gemv_rows(n: int, k: int) -> int:
    """The matrix rows the GEMV reference casts to float64 at once: all of them under a third of the cap in bytes,
    else that third's worth of rows."""
    return max(1, min(n, int(ORACLE_MAX_MACS / 3 // max(1, k * 8))))


class Exact:
    """A reference of integers, met exactly: the deviation is 0 where every value equals, else inf."""
    describe = "exact integers"

    def __init__(self, ref):
        self.ref = np.asarray(ref, dtype=np.int64).reshape(-1)

    def slices(self, out):
        got = np.asarray(out).astype(np.int64).reshape(-1)
        if got.shape != self.ref.shape:
            return [(np.ones(1), np.zeros(1))]
        return [((got != self.ref).astype(np.float64), np.zeros(got.shape))]


class WindowedOracle:
    """The float64 reference on windows of a convolution's output; a result is measured on those windows only, and
    `describe` names them."""

    def __init__(self, blocks, oh, ow, n):
        self.blocks = blocks                              # [(window, float64 array [1, co, rh, rw])]
        self.describe = (f"on {len(blocks)} window(s) of the output (" +
                         "; ".join(f"batch {w[0]} rows {w[1]}-{w[2]} cols {w[3]}-{w[4]}" for w, _ in blocks) +
                         f") of {n}x{oh}x{ow}")

    def slices(self, out):
        for (ni, r0, r1, c0, c1), ref in self.blocks:
            yield out[ni:ni + 1, :, r0:r1, c0:c1], ref

    def device_slices(self, t):
        """The same windows, cut on the tensor WHERE IT LIVES: never the whole output across to the host for three
        small windows of it (1.73 GB a candidate on a 49x128x96x720 convolution)."""
        for (ni, r0, r1, c0, c1), ref in self.blocks:
            yield materialise(t[ni:ni + 1, :, r0:r1, c0:c1]), ref


class RowWindowedOracle:
    """The float64 reference on row windows of an output whose rows are its second-to-last axis (a product's [M, N] or
    [B, M, N], an attention's [B, H, Tq, D], a row op's [rows, feat]); measured on those rows only."""

    def __init__(self, blocks, m):
        self.blocks = blocks                              # [((r0, r1), float64 array [..., r1-r0, N])]
        self.describe = (f"on {len(blocks)} row window(s) of the output (" +
                         "; ".join(f"rows {r0}-{r1}" for (r0, r1), _ in blocks) + f") of {m} rows")

    def slices(self, out):
        for (r0, r1), ref in self.blocks:
            yield out[..., r0:r1, :], ref

    def device_slices(self, t):
        """The same row windows, cut on the tensor where it lives, whatever its rank."""
        for (r0, r1), ref in self.blocks:
            cut = t[(slice(None),) * (len(tuple(t.shape)) - 2) + (slice(r0, r1), slice(None))]
            yield materialise(cut), ref


class HostWindows:
    """The rows of a host draw a row-windowed reference reads, and nothing else, as numbers: built from the draw before
    it is dropped (on a unified-memory device the draw and its device copy share one pool), read as `x[r0:r1, :]`."""

    def __init__(self, x, wins):
        self.shape = tuple(x.shape)
        self._rows = {(int(r0), int(r1)): np.array(values(x[r0:r1])) for r0, r1 in wins}

    def __getitem__(self, idx):
        rs = idx[0] if isinstance(idx, tuple) else idx
        return self._rows[(int(rs.start), int(rs.stop))]


def oracle_rows(x):
    """What a row-windowed reference keeps of a host draw: the draw itself when it reads it whole, else only the rows of
    its windows (`HostWindows`)."""
    m, n = x.shape[-2], x.shape[-1]
    wins = row_windows(m, n, ROW_F64_ARRAYS * 8)
    return x if wins is None else HostWindows(x, wins)


def _by_rows(rows: Callable[[int, int], np.ndarray], m: int, n: int) -> Callable[[], Any]:
    """A row-wise reference: `rows(r0, r1)` whole under the cap in bytes, on row windows above it."""
    wins = row_windows(m, n, ROW_F64_ARRAYS * 8)
    if wins is None:
        return lambda: rows(0, m)
    return lambda: RowWindowedOracle([((r0, r1), rows(r0, r1)) for (r0, r1) in wins], m)


def conv_out_hw(h, wd, kh, kw, stride, padding, dilation) -> Tuple[int, int]:
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    return (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1, (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1


def conv2d_oracle(x, w, stride, padding, dilation, groups, window=None) -> np.ndarray:
    """Direct convolution in float64 (NCHW, OIHW) through `conv2d_fp64.conv2d_reference_window`; `window` = (n_idx,
    r0, r1, c0, c1) computes that output block only. The input is read slab by slab, so a bf16 operand is decoded on
    the window's receptive field only."""
    from nbxtensor.ops.oracles.conv2d_fp64 import conv2d_reference_window
    return conv2d_reference_window(None, f64(w), stride=stride, padding=padding, dilation=dilation,
                                   groups=groups, window=window, shape=x.shape,
                                   slab_of=lambda n0, n1, u0, u1, v0, v1: f64(x[n0:n1, :, u0:u1, v0:v1]))


def matmul_oracle(a, b, bias=None) -> Callable[[], Any]:
    """The reference of a (..., M, K) x (..., K, N) product (+ bias over N, or a [..., M, N] bias), whole under the cap,
    row-windowed above it."""
    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    batch = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1

    def whole():
        r = f64(a) @ f64(b)
        return r if bias is None else r + f64(bias)
    if row_windows(m * batch, n, k) is None:
        return whole
    wins = row_windows(m, n, k, cap=max(1, ORACLE_MAX_MACS // batch)) or [(0, m)]

    def windowed():
        blocks = []
        for (r0, r1) in wins:
            r = f64(a[..., r0:r1, :]) @ f64(b)
            if bias is not None:
                bb = f64(bias)
                r = r + (bb[..., r0:r1, :] if bb.ndim >= 2 and bb.shape[-2] == m else bb)
            blocks.append(((r0, r1), r))
        return RowWindowedOracle(blocks, m)
    return windowed


def moe_oracle(x, ws, score, route, mul) -> Callable[[], Any]:
    """The reference of a grouped MoE product: each (token, expert) pair's row times its expert's weights [N, K] (times
    its routed weight when `mul`), on row windows of the pairs above the cap; per window its activations in float64,
    gathered per expert, one expert's weights at a time."""
    nv, K = int(x.shape[0]), int(x.shape[1])
    N = int(ws[0].shape[0])

    def window(r0, r1):
        xs, rt = f64(x[r0:r1]), route[r0:r1]
        y = np.empty((r1 - r0, N), dtype=np.float64)
        for e in np.unique(rt):
            idx = np.nonzero(rt == e)[0]
            y[idx] = xs[idx] @ f64(ws[int(e)]).T
        return y * f64(score[r0:r1])[:, None] if mul else y

    def build():
        wins = row_windows(nv, N, K)
        if wins is None:
            return window(0, nv)
        return RowWindowedOracle([((r0, r1), window(r0, r1)) for r0, r1 in wins], nv)
    return build


def attention_oracle(q, k, v, mask, scale) -> Callable[[], Any]:
    """The reference of softmax(q k^T * scale + mask) v over [B, H, Tq, D] (`attention_fp64`), on query-row windows
    above the cap (a query row costs its Tk scores and their Tk x D products)."""
    from nbxtensor.ops.oracles.attention_fp64 import attention_fp64
    tq, hd, tk = q.shape[-2], q.shape[-1], k.shape[-2]
    wins = row_windows(tq, hd, 2 * tk)
    if wins is None:
        return lambda: attention_fp64(q, k, v, scale, [(0, tq)], mask, values)[0][1]
    return lambda: RowWindowedOracle(attention_fp64(q, k, v, scale, wins, mask, values), tq)


def softmax_oracle(x, log: bool) -> Callable[[], Any]:
    """The reference of a row softmax (or log-softmax) of x [rows, feat]."""
    def rows(r0, r1):
        z = f64(x[r0:r1, :])
        z = z - z.max(axis=-1, keepdims=True)
        if log:
            return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    return _by_rows(rows, x.shape[-2], x.shape[-1])


def gemv_oracle(mat, vec, bias=None) -> Callable[[], np.ndarray]:
    """The reference of mat [N, K] @ vec [K] (+ bias [N]): every output row, the matrix cast a block of rows at a time
    (`gemv_rows`) so a vocabulary head is never one float64 copy."""
    n, k = mat.shape
    rows = gemv_rows(n, k)

    def run():
        v = f64(vec)
        out = np.empty(n, dtype=np.float64)
        for r0 in range(0, n, rows):
            out[r0:r0 + rows] = f64(mat[r0:r0 + rows]) @ v
        return out if bias is None else out + f64(bias)
    return run


def rownorm_oracle(x, w, b, layer: bool) -> Callable[[], Any]:
    """The reference of a row layer norm (eps 1e-5) or RMS norm (eps 1e-6) of x [rows, feat] with its weight and
    bias — the wrappers' own default epsilons."""
    wf = None if w is None else f64(w)
    bf = None if b is None else f64(b)

    def rows(r0, r1):
        z = f64(x[r0:r1, :])
        if layer:
            z = z - z.mean(axis=-1, keepdims=True)
            z = z / np.sqrt((z * z).mean(axis=-1, keepdims=True) + 1e-5)
        else:
            z = z / np.sqrt((z * z).mean(axis=-1, keepdims=True) + 1e-6)
        if wf is not None:
            z = z * wf
        if bf is not None:
            z = z + bf
        return z
    return _by_rows(rows, x.shape[-2], x.shape[-1])


def row_reduce_oracle(x, red: str) -> Callable[[], np.ndarray]:
    """The reference of a row reduction ("mean", "sum" or "amax") of x [rows, feat], a window of rows at a time."""
    m, n = x.shape[-2], x.shape[-1]
    op = {"mean": np.mean, "sum": np.sum, "amax": np.max}[red]

    def run():
        out = np.empty(m, dtype=np.float64)
        step = max(1, ORACLE_MAX_MACS // (8 * max(1, n)))      # a window of ORACLE_MAX_MACS bytes in float64
        for r0 in range(0, m, step):
            out[r0:r0 + step] = op(f64(x[r0:r0 + step, :]), axis=-1)
        return out
    return run


def groupnorm_oracle(x, w, b) -> Callable[[], Any]:
    """The reference of a group norm of x [groups, extent] (one channel per group, eps 1e-5) with its per-group weight
    and bias."""
    wf = None if w is None else f64(w)[:, None]
    bf = None if b is None else f64(b)[:, None]

    def rows(r0, r1):
        z = f64(x[r0:r1, :])
        z = z - z.mean(axis=-1, keepdims=True)
        z = z / np.sqrt((z * z).mean(axis=-1, keepdims=True) + 1e-5)
        if wf is not None:
            z = z * wf[r0:r1]
        if bf is not None:
            z = z + bf[r0:r1]
        return z
    return _by_rows(rows, x.shape[-2], x.shape[-1])


def conv_oracle(x, wt, stride, padding, dilation, groups) -> Callable[[], Any]:
    """The reference of a convolution, whole under the cap, on `conv_windows` above it."""
    n, ci, h, wd = x.shape
    co, ci_g, kh, kw = wt.shape
    oh, ow = conv_out_hw(h, wd, kh, kw, stride, padding, dilation)
    wins = conv_windows(n, oh, ow, ci_g, co, kh, kw)
    if wins is None:
        return lambda: conv2d_oracle(x, wt, stride, padding, dilation, groups)
    return lambda: WindowedOracle([(w, conv2d_oracle(x, wt, stride, padding, dilation, groups, window=w))
                                   for w in wins], oh, ow, n)

def causal_conv1d_oracle(x, weight, bias=None, state=None) -> Tuple[np.ndarray, np.ndarray]:
    """The reference of `causal_conv1d_wrapper` in float64: (y [B, C, L], state_out [B, C, K-1]). x [B, C, L], weight
    [C, K] or [C, 1, K], bias [C] or None, state [B, C, K-1] or None (zeros before the start). The state is joined
    before x and each tap is added as its own shifted product: y[t] = sum_j w[j] * joined[t + j] (+ bias); state_out is
    the joined input's last K-1 columns. Computed whole (B * C * L * K multiply-adds); its channels are independent, so
    a caller over the cap windows it by channels."""
    xf = f64(x)
    wf = f64(weight)
    if wf.ndim == 3:
        wf = wf[:, 0, :]
    B, C, L = xf.shape
    K = wf.shape[1]
    past = np.zeros((B, C, K - 1), dtype=np.float64) if state is None else f64(state)
    joined = np.concatenate([past, xf], axis=2)
    y = np.zeros((B, C, L), dtype=np.float64)
    for j in range(K):
        y += wf[None, :, j, None] * joined[:, :, j:j + L]
    if bias is not None:
        y += f64(bias)[None, :, None]
    return y, joined[:, :, L:L + K - 1].copy()


def head_windows(batch: int, heads: int, per_head: int) -> Optional[List[Tuple[int, int, int]]]:
    """None when a reference over `batch` rows of `heads` independent heads, `per_head` multiply-adds each, fits
    `ORACLE_MAX_MACS`; else its (batch row, h0, h1) windows — the first heads of the first row, the middle heads of the
    middle row, the last heads of the last row — each of at most cap/3 multiply-adds (at least one head: a recurrence's
    head is never cut, its whole length is kept)."""
    if batch * heads * per_head <= ORACLE_MAX_MACS:
        return None
    n = max(1, min(heads, int(ORACLE_MAX_MACS / 3 // max(1, per_head))))
    mid = (heads - n) // 2
    return _distinct([(0, 0, n), ((batch - 1) // 2, mid, mid + n), (batch - 1, heads - n, heads)])


class HeadWindowedOracle:
    """The float64 reference on (batch row, head range) windows of an output whose heads are axis `head_axis` (a
    recurrence's rows depend on every earlier row, so it is windowed over its independent heads and batch rows, never
    over rows); measured on those windows only."""

    def __init__(self, blocks, head_axis: int, batch: int, heads: int):
        self.blocks = blocks                              # [((b, h0, h1), float64 array [1, ..., h1-h0, ...])]
        self.head_axis = head_axis
        self.describe = (f"on {len(blocks)} head window(s) of the output (" +
                         "; ".join(f"batch {b} heads {h0}-{h1}" for (b, h0, h1), _ in blocks) +
                         f") of {batch} x {heads} heads")

    def _cut(self, b, h0, h1):
        idx = [slice(None)] * (self.head_axis + 1)
        idx[0], idx[self.head_axis] = slice(b, b + 1), slice(h0, h1)
        return tuple(idx)

    def slices(self, out):
        for (b, h0, h1), ref in self.blocks:
            yield out[self._cut(b, h0, h1)], ref

    def device_slices(self, t):
        """The same windows, cut on the tensor where it lives."""
        for (b, h0, h1), ref in self.blocks:
            yield materialise(t[self._cut(b, h0, h1)]), ref


def _unit_rows(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def _delta_rule_window(q, k, v, g, beta, state, b, h0, h1, qk_l2norm, eps, scale):
    """The recurrence for batch row `b`, value heads h0..h1, over the whole length, in float64 (arrays already float64,
    q and k un-expanded): (out [1, L, h, dv], state [1, h, dk, dv])."""
    kh = np.arange(h0, h1) // (v.shape[2] // q.shape[2])
    qw, kw = q[b][:, kh], k[b][:, kh]                     # [L, h, dk]: each value head's key head
    if qk_l2norm:
        qw, kw = _unit_rows(qw, eps), _unit_rows(kw, eps)
    qw = qw * scale
    vw, gw, bw = v[b][:, h0:h1], g[b][:, h0:h1], beta[b][:, h0:h1]
    s = (np.zeros((h1 - h0, q.shape[3], v.shape[3])) if state is None else state[b, h0:h1].copy())
    out = np.empty(vw.shape)
    for t in range(q.shape[1]):
        s *= np.exp(gw[t])[:, None, None]                                         # the decay
        u = (vw[t] - np.einsum("hkv,hk->hv", s, kw[t])) * bw[t][:, None]          # the correction
        s += np.einsum("hk,hv->hkv", kw[t], u)                                    # the rank-one write
        out[t] = np.einsum("hkv,hk->hv", s, qw[t])                                # the read, after the write
    return out[None], s[None]


def delta_rule_oracle(q, k, v, g, beta, state=None, *, qk_l2norm: bool, eps: float, scale: float
                      ) -> Callable[[], Tuple[Any, Any]]:
    """The reference of the gated delta rule (`ops.gated_delta_rule`) over q, k [B, L, Hk, dk], v [B, L, Hv, dv], g,
    beta [B, L, Hv] from `state` [B, Hv, dk, dv] (None: zeros): (out [B, L, Hv, dv], state [B, Hv, dk, dv]) in float64,
    each whole under the cap, a `HeadWindowedOracle` above it (a head costs L x dk x dv x 3 multiply-adds)."""
    B, L, Hk, dk = q.shape
    Hv, dv = v.shape[2], v.shape[3]
    if Hk == 0 or Hv % Hk:
        raise ValueError(f"delta_rule_oracle: {Hv} value heads are not a whole multiple of {Hk} key heads")
    q, k, v, g, beta = (f64(x) for x in (q, k, v, g, beta))
    state = None if state is None else f64(state)

    def window(b, h0, h1):
        return _delta_rule_window(q, k, v, g, beta, state, b, h0, h1, qk_l2norm, eps, scale)

    def build():
        wins = head_windows(B, Hv, L * dk * dv * 3)
        if wins is None:
            parts = [window(b, 0, Hv) for b in range(B)]
            return np.concatenate([o for o, _ in parts]), np.concatenate([s for _, s in parts])
        parts = [(w, window(*w)) for w in wins]
        return (HeadWindowedOracle([(w, o) for w, (o, _) in parts], 2, B, Hv),
                HeadWindowedOracle([(w, s) for w, (_, s) in parts], 1, B, Hv))
    return build

def _top_first(v: np.ndarray, n: int) -> np.ndarray:
    """The ids of the n largest values of each row, the largest first, the lower id first on an exact tie."""
    ids = np.broadcast_to(np.arange(v.shape[-1]), v.shape)
    return np.stack([np.lexsort((i, -r))[:n] for i, r in zip(ids, v)])


def _group_scores(c: np.ndarray, n_group: int, group_score: str) -> np.ndarray:
    g = np.sort(c.reshape(c.shape[0], n_group, -1), axis=-1)[..., ::-1]
    return g[..., :2].sum(-1) if group_score == "top2_sum" else g[..., 0]


def moe_route_oracle(hidden, router_weight, choice_bias, *, score_fn: str, top_k: int, n_group: int, topk_group: int,
                     group_score: str, fill: float, renorm_eps: Optional[float], scale: float
                     ) -> Callable[[], Tuple[Any, Any, Any, Any]]:
    """The reference of the routing op (`ops.moe_route`) of hidden [T, H] over router_weight [E, H] with choice_bias
    [E] (or None), in float64: (indices [T, top_k] int64, weights [T, top_k], the choice values the final top-k ranks
    [T, E] (a dropped group's experts at `fill`), the group scores [T, n_group]). The slots come largest first, the
    lower id first on an exact tie (the op's slot order is not its contract: compare sets, pair weights by id)."""
    x, w = f64(hidden), f64(router_weight)
    bias = None if choice_bias is None else f64(choice_bias)
    if score_fn not in ("sigmoid", "softmax") or group_score not in ("top2_sum", "max"):
        raise ValueError(f"moe_route_oracle: score_fn {score_fn!r} / group_score {group_score!r} is not one the op "
                         f"computes")

    def build():
        logits = x @ w.T
        if score_fn == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-logits))
        else:
            p = np.exp(logits - logits.max(-1, keepdims=True))
            s = p / p.sum(-1, keepdims=True)
        c = s if bias is None else s + bias
        gs = _group_scores(c, n_group, group_score)
        kept = np.zeros(gs.shape, dtype=bool)
        np.put_along_axis(kept, _top_first(gs, topk_group), True, axis=-1)
        c = np.where(np.repeat(kept, c.shape[-1] // n_group, axis=-1), c, fill)
        idx = _top_first(c, top_k)
        wt = np.take_along_axis(s, idx, -1)
        if renorm_eps is not None:
            wt = wt / (wt.sum(-1, keepdims=True) + renorm_eps)
        return idx.astype(np.int64), wt * scale, c, gs
    return build


def host_values(t) -> np.ndarray:
    """A kernel's output as NUMBERS, whatever dtype it carries: `NBXTensor.numpy()` hands bf16 back as 2-byte void
    items (numpy has no bfloat16), decoded here exactly through `bf16_bits_to_f32`."""
    host = t.numpy()
    dt = getattr(host, "dtype", None)
    if dt is not None and dt.kind == "V" and dt.itemsize == 2:
        return bf16_bits_to_f32(np.ascontiguousarray(host).view(np.uint16))
    return host


def materialise(piece):
    """A cut ready to be read as numbers: a tensor's slice is a view whose strides no flat reader follows, so it is made
    contiguous on the device before it crosses; an array is handed back."""
    contiguous = getattr(piece, "contiguous", None)
    return contiguous() if callable(contiguous) else piece


def deviation_against(out_tensor, oracle) -> float:
    """The deviation of a kernel's output against `oracle`, reading from the device only what the oracle measures (a
    windowed oracle's windows, cut where the output lives)."""
    if hasattr(oracle, "device_slices"):
        return max(oracle_deviation(host_values(piece) if hasattr(piece, "data_ptr") else piece, ref)
                   for piece, ref in oracle.device_slices(out_tensor))
    return oracle_deviation(host_values(out_tensor), oracle)


def oracle_deviation(out: np.ndarray, oracle) -> float:
    """max |out - oracle| relative to the oracle's own magnitude; inf when one side is not finite where the other is,
    or the shapes differ. A windowed oracle is measured on its windows, the largest deviation among them."""
    if hasattr(oracle, "slices"):
        return max(oracle_deviation(piece, ref) for piece, ref in oracle.slices(np.asarray(out)))
    x = np.asarray(out, dtype=np.float64).reshape(-1)
    y = np.asarray(oracle, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        return float("inf")
    fx, fy = np.isfinite(x), np.isfinite(y)
    if not np.array_equal(fx, fy):
        return float("inf")
    if not fy.any():
        return 0.0
    scale = float(np.abs(y[fy]).max())
    if scale == 0.0:
        return 0.0 if np.array_equal(x, y) else float("inf")
    return float(np.abs(x[fy] - y[fy]).max() / scale)
