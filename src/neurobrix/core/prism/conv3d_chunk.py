"""When a 3-D convolution streams its temporal output in chunks — a PLAN decision.

The triton wrapper runs conv3d as kt conv2d launches over the folded batch axis (B*T_out frames);
when the folds are large it streams them in chunks, each transient bounded by `CHUNK_BYTES`. It
decided to chunk by reading the driver's FREE bytes at that instant — a key a live run formed
depended on the memory free at that moment, which no census and no derivation can know (the
supervisor's decision 2, 2026-09-29). The decision is now Prism's, made from the plan like every
other tiling decision: `need` (the one-shot path's peak, below) against the op's planned budget
(0.85 of the card minus the component's resident weights, the budget op-level tiling already
uses). Pure arithmetic: Prism prices with it at plan time, the wrapper sizes the chunks with it.
"""
from __future__ import annotations

import os
from typing import Sequence, Tuple

#: The per-transient bound of the chunked path (and the fold size above which the one-shot path's
#: peak is worth pricing at all).
CHUNK_BYTES = int(os.environ.get("NBX_CONV3D_CHUNK_BYTES", str(1 * 1024 * 1024 * 1024)))
#: conv2d's band-streaming threshold: a one-shot conv3d whose conv2d output exceeds it holds a
#: band's worth more.
BAND_BYTES = int(os.environ.get("NBX_CONV2D_BAND_BYTES", str(4 * 1024 * 1024 * 1024)))
_SLACK = 256 * 1024 * 1024


def _triple(v) -> Tuple[int, int, int]:
    if isinstance(v, (list, tuple)):
        v = list(v) + [v[-1]] * (3 - len(v))
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


def conv3d_need(x_shape: Sequence[int], w_shape: Sequence[int], stride, padding, dilation,
                in_bytes: int, out_bytes: int) -> Tuple[int, int]:
    """(bytes the ONE-SHOT path holds at its peak, folded bytes per output frame). `need` is 0 when
    no fold exceeds `CHUNK_BYTES` — the one-shot path is taken whatever the budget. The peak: the
    temporal pad copy, then the larger of (input fold + output fold [+ a band when the output bands]
    [+ the kt accumulator]) and (two output folds [+ the accumulator]), and a fixed slack."""
    B, Cin, T, H, W = (int(d) for d in x_shape)
    Cout, _Cg, kt, kh, kw = (int(d) for d in w_shape)
    st, sh, sw = _triple(stride)
    pt, ph, pw = _triple(padding)
    dt, dh, dw = _triple(dilation)
    t_out = (T + 2 * pt - dt * (kt - 1) - 1) // st + 1
    if t_out <= 0:
        return 0, 0
    oh = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    fold_in = B * t_out * Cin * H * W * in_bytes
    fold_out = B * t_out * Cout * oh * ow * out_bytes
    frame = max(fold_in, fold_out) // max(1, t_out)
    if max(fold_in, fold_out) <= CHUNK_BYTES:
        return 0, frame
    pad_bytes = B * Cin * (T + 2 * pt) * H * W * in_bytes if pt > 0 else 0
    band_extra = BAND_BYTES if fold_out > BAND_BYTES else 0
    acc_extra = fold_out if kt >= 2 else 0
    need = pad_bytes + max(fold_out + fold_in + band_extra + acc_extra, 2 * fold_out + acc_extra) + _SLACK
    return need, frame


def chunk_frames(frame_bytes: int) -> int:
    """Output frames per chunk: each chunk's folded transient under `CHUNK_BYTES`."""
    return max(1, int(CHUNK_BYTES // max(1, frame_bytes)))
