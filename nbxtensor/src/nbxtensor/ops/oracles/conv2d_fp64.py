"""The convolution family's float64 reference — the screen's uncovered 11.5%.

WHY THIS FAMILY AND NOT ANOTHER
--------------------------------
Measured 2026-09-12 against this rack's certified directory: of 7 158 certified
keys the screen oracle covers 6 336 (88.5%), and the 822 it does not are exactly
`conv2d_forward_kernel` (770) and `depthwise_conv2d_kernel` (52). Nothing else.
Full count: `docs/reference/owed-proofs.md` item 3.

Without a reference, those keys are decided by a consensus vote, and a vote has
two failure modes it cannot see — a majority wrong in the same way, and a
unanimous space that is wrong. Both were measured on another backend: four
`addmm` shapes where the emitted code declared `alpha`/`beta` as `int`, every
candidate wrong identically, the vote unanimous, and the bare screen seating a
wrong configuration every time while the same screen with an oracle refused every
time.

WHAT MAKES A REFERENCE A REFERENCE
-----------------------------------
It must not share an implementation with the thing it checks. So: pure numpy in
float64, no Triton, no torch, and the arithmetic written out rather than
delegated. It is slow and that is irrelevant — it runs once per shape, inside a
screen that already refuses to look at anything over the profile's byte budget.

The kernels' signatures make this tractable: every dimension, every stride and
every compile-time parameter is a NAMED argument, so nothing is inferred.

    conv2d_forward_kernel(input, weight, output, N, Cin, H, W, Cout, Hout, Wout,
                          <12 strides>, kernel_h/w, stride_h/w, padding_h/w,
                          groups, fp16, dilation_h/w, <3 block sizes>)
    depthwise_conv2d_kernel(x, w, out, N, C, H_in, W_in, H_out, W_out,
                            <11 strides>, kh, kw, stride_h/w, pad_h/w, fp16,
                            <2 block sizes>)

Neither takes a bias, so the reference is the convolution alone.

HOW IT REFUSES
--------------
Silence, never a guess. An operand that cannot be read here, a declared extent
that contradicts the array it names, an output extent that does not follow from
the inputs — each returns None, and the screen then says out loud that this key
is unscreened. A reference that answers when it should not is worse than none:
it turns healthy work into a loud stop, and a refusal reads as vigilance.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

import numpy as np


def _out_extent(size: int, pad: int, dilation: int, kernel: int, stride: int) -> int:
    """The spatial extent a convolution produces, by the standard formula."""
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def conv2d_reference(x: np.ndarray, w: np.ndarray, *,
                     stride: Sequence[int], padding: Sequence[int],
                     dilation: Sequence[int], groups: int) -> np.ndarray:
    """`[N, Cin, H, W]` ⊛ `[Cout, Cin//groups, KH, KW]` → `[N, Cout, Ho, Wo]`, fp64.

    Written as an accumulation over the KH·KW taps rather than an im2col: the
    strided view per tap is exact and allocates nothing, and the loop makes the
    padding and dilation arithmetic visible instead of hidden in a reshape.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 4 or w.ndim != 4:
        raise ValueError(f"expected 4-D input and weight, got {x.shape} {w.shape}")
    n, c_in, h, width = x.shape
    c_out, c_in_group, k_h, k_w = w.shape
    if groups < 1 or c_in % groups or c_out % groups:
        raise ValueError(f"{c_in} in / {c_out} out do not divide into {groups} groups")
    if c_in // groups != c_in_group:
        raise ValueError(f"weight declares {c_in_group} input channels per group, "
                         f"the input has {c_in // groups}")
    s_h, s_w = int(stride[0]), int(stride[1])
    p_h, p_w = int(padding[0]), int(padding[1])
    d_h, d_w = int(dilation[0]), int(dilation[1])

    out_h = _out_extent(h, p_h, d_h, k_h, s_h)
    out_w = _out_extent(width, p_w, d_w, k_w, s_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"the geometry produces a {out_h}x{out_w} output")

    padded = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float64)
    per_in, per_out = c_in // groups, c_out // groups
    for g in range(groups):
        x_g = padded[:, g * per_in:(g + 1) * per_in]
        w_g = w[g * per_out:(g + 1) * per_out]
        acc = out[:, g * per_out:(g + 1) * per_out]
        for i in range(k_h):
            top = i * d_h
            rows = slice(top, top + (out_h - 1) * s_h + 1, s_h)
            for j in range(k_w):
                left = j * d_w
                cols = slice(left, left + (out_w - 1) * s_w + 1, s_w)
                acc += np.einsum("nchw,oc->nohw", x_g[:, :, rows, cols],
                                 w_g[:, :, i, j])
    return out


def receptive_slab(h: int, wd: int, k_h: int, k_w: int, *, stride, padding, dilation, window):
    """The input slab an output window `(r0, r1, c0, c1)` reads, and the padding it still needs.

    Returns `((u0, u1, v0, v1), (top, bottom, left, right))`: the UNPADDED input rows `u0:u1`
    and columns `v0:v1` the window's receptive field covers, and the zero padding to put back
    around that slab so the tap loop below sees the same coordinates it would see on the whole
    padded input. One arithmetic for the certifier and the runtime screen — written twice it
    would be two places to be wrong.
    """
    r0, r1, c0, c1 = window
    s_h, s_w = int(stride[0]), int(stride[1])
    p_h, p_w = int(padding[0]), int(padding[1])
    d_h, d_w = int(dilation[0]), int(dilation[1])
    R0, R1 = r0 * s_h, (r1 - 1) * s_h + d_h * (k_h - 1) + 1      # padded coordinates
    C0, C1 = c0 * s_w, (c1 - 1) * s_w + d_w * (k_w - 1) + 1
    u0, u1 = max(0, R0 - p_h), min(h, R1 - p_h)
    v0, v1 = max(0, C0 - p_w), min(wd, C1 - p_w)
    top, bottom = max(0, p_h - R0), max(0, (R1 - p_h) - h)
    left, right = max(0, p_w - C0), max(0, (C1 - p_w) - wd)
    return (u0, u1, v0, v1), (top, bottom, left, right)


def conv2d_reference_window(x, w: np.ndarray, *, stride: Sequence[int], padding: Sequence[int],
                            dilation: Sequence[int], groups: int, window=None,
                            shape: Optional[Sequence[int]] = None, slab_of=None) -> np.ndarray:
    """`conv2d_reference` on one output block only: `window` = `(n_idx, r0, r1, c0, c1)` selects
    `out[n_idx, :, r0:r1, c0:c1]`, exact on every position of it, reading ONLY its receptive
    field. `window=None` is the whole output.

    The input arrives either as a host array `x` or — when `slab_of` is given — through a reader
    `slab_of(n0, n1, u0, u1, v0, v1) -> float64 [n1-n0, Cin, u1-u0, v1-v0]` that fetches just the
    slab, with `shape` naming the whole input's `[N, Cin, H, W]`. The reader is what bounds the
    runtime screen: a 128-channel 2048x1024 input is 2.1 GB in float64 whole (the Mac's 27.5 GB
    footprint, 2026-09-25) and a few megabytes as the slab of a few output rows.

    One BLAS product per tap, the depthwise case as one broadcast product per tap — the
    certifier's arithmetic since 2026-09-07, now the one copy of it.
    """
    w = np.asarray(w, dtype=np.float64)
    if slab_of is None:
        x = np.asarray(x)
        shape = x.shape
    n, c_in, h, wd = (int(v) for v in shape)
    c_out, ci_g, k_h, k_w = w.shape
    if groups < 1 or c_in % groups or c_out % groups or c_in // groups != ci_g:
        raise ValueError(f"{c_in} in / {c_out} out / {ci_g} per group do not divide into {groups} groups")
    s_h, s_w = int(stride[0]), int(stride[1])
    d_h, d_w = int(dilation[0]), int(dilation[1])
    oh = _out_extent(h, int(padding[0]), d_h, k_h, s_h)
    ow = _out_extent(wd, int(padding[1]), d_w, k_w, s_w)
    if oh <= 0 or ow <= 0:
        raise ValueError(f"the geometry produces a {oh}x{ow} output")
    if window is None:
        n0, n1, r0, r1, c0, c1 = 0, n, 0, oh, 0, ow
    else:
        ni, r0, r1, c0, c1 = window
        n0, n1 = int(ni), int(ni) + 1
    (u0, u1, v0, v1), pads = receptive_slab(h, wd, k_h, k_w, stride=stride, padding=padding,
                                             dilation=dilation, window=(r0, r1, c0, c1))
    if slab_of is None:
        slab = x[n0:n1, :, u0:u1, v0:v1].astype(np.float64)
    else:
        slab = np.asarray(slab_of(n0, n1, u0, u1, v0, v1), dtype=np.float64)
        if slab.shape != (n1 - n0, c_in, u1 - u0, v1 - v0):
            raise ValueError(f"the slab reader returned {slab.shape}, "
                             f"the window needs {(n1 - n0, c_in, u1 - u0, v1 - v0)}")
    top, bottom, left, right = pads
    xp = np.pad(slab, ((0, 0), (0, 0), (top, bottom), (left, right)))
    nb, rh, rw = n1 - n0, r1 - r0, c1 - c0
    out = np.zeros((nb, c_out, rh, rw), dtype=np.float64)
    co_g = c_out // groups
    if groups == c_in == c_out and ci_g == 1:
        for i in range(k_h):
            for j in range(k_w):
                patch = xp[:, :, i * d_h:i * d_h + rh * s_h:s_h, j * d_w:j * d_w + rw * s_w:s_w]
                out += patch * w[:, 0, i, j][None, :, None, None]
        return out
    for g in range(groups):
        xg = xp[:, g * ci_g:(g + 1) * ci_g]
        wg = w[g * co_g:(g + 1) * co_g]
        for i in range(k_h):
            for j in range(k_w):
                patch = xg[:, :, i * d_h:i * d_h + rh * s_h:s_h, j * d_w:j * d_w + rw * s_w:s_w]
                prod = patch.transpose(0, 2, 3, 1).reshape(-1, ci_g) @ wg[:, :, i, j].T
                out[:, g * co_g:(g + 1) * co_g] += prod.reshape(nb, rh, rw, co_g).transpose(0, 3, 1, 2)
    return out


def _row_segments(rows, c_out: int, oh: int):
    """Flat output rows `[r0, r1)` of the `[N*Cout*Ho, Wo]` view, cut at every channel boundary:
    yields `(n, c, oh_lo, oh_hi, flat_lo, flat_hi)`, the output rows `oh_lo:oh_hi` of channel `c`
    of element `n`, which are exactly the flat rows `[flat_lo, flat_hi)`.

    One segment per `(n, c)` run, never a widening: a window that straddles a channel used to
    be widened to every row of the element (`(0, oh)`), whose receptive field is the WHOLE
    input — for a 128 x 2048 x 1024 input the 2.1 GB of float64 this file exists to avoid,
    reachable by window position alone (the guardian, 2026-09-26). The bound must not depend
    on where the windows fall."""
    r0, r1 = int(rows[0]), int(rows[1])
    per_n = c_out * oh
    r = r0
    while r < r1:
        n, rem = divmod(r, per_n)
        c, oh_lo = divmod(rem, oh)
        end = min(r1, r + (oh - oh_lo))              # the last flat row of this channel, or r1
        yield n, c, oh_lo, oh_lo + (end - r), r, end
        r = end


def _conv_geometry(named: dict):
    """The `[N, Cin, H, W]` input, `[Cout, Cin/groups, KH, KW]` weight shape, stride, padding,
    dilation, groups and declared output extents of `conv2d_forward_kernel`'s named arguments —
    refusing, with the reason, a declaration that contradicts the formula."""
    shape = (int(named["batch_dim"]), int(named["in_feat_dim"]),
             int(named["in_height"]), int(named["in_width"]))
    groups = int(named.get("groups", 1))
    k_h, k_w = int(named["kernel_height"]), int(named["kernel_width"])
    w_shape = (int(named["out_feat_dim"]), shape[1] // groups, k_h, k_w)
    stride = (int(named["stride_height"]), int(named["stride_width"]))
    padding = (int(named["padding_height"]), int(named["padding_width"]))
    dilation = (int(named.get("dilation_height", 1)), int(named.get("dilation_width", 1)))
    oh, ow = int(named["out_height"]), int(named["out_width"])
    formula = (_out_extent(shape[2], padding[0], dilation[0], k_h, stride[0]),
               _out_extent(shape[3], padding[1], dilation[1], k_w, stride[1]))
    if (oh, ow) != formula:
        raise ValueError(f"the declared output {oh}x{ow} is not the formula's {formula[0]}x{formula[1]}")
    return shape, w_shape, stride, padding, dilation, groups, oh, ow


def _depthwise_geometry(named: dict):
    c = int(named["C"])
    k_h, k_w = int(named["kh"]), int(named["kw"])
    shape = (int(named["N"]), c, int(named["H_in"]), int(named["W_in"]))
    stride = (int(named["stride_h"]), int(named["stride_w"]))
    padding = (int(named["pad_h"]), int(named["pad_w"]))
    oh, ow = int(named["H_out"]), int(named["W_out"])
    formula = (_out_extent(shape[2], padding[0], 1, k_h, stride[0]),
               _out_extent(shape[3], padding[1], 1, k_w, stride[1]))
    if (oh, ow) != formula:
        raise ValueError(f"the declared output {oh}x{ow} is not the formula's {formula[0]}x{formula[1]}")
    return shape, (c, 1, k_h, k_w), stride, padding, (1, 1), c, oh, ow


def _rows_of(geometry, slab_of, weight_f64, rows) -> np.ndarray:
    shape, w_shape, stride, padding, dilation, groups, oh, ow = geometry
    w = np.asarray(weight_f64, dtype=np.float64)
    if w.size != int(np.prod(w_shape)):
        raise ValueError(f"the weight holds {w.size} elements, the declared {w_shape} needs {int(np.prod(w_shape))}")
    w = w.reshape(w_shape)
    parts = []
    for n, c, oh_lo, oh_hi, lo, hi in _row_segments(rows, w_shape[0], oh):
        block = conv2d_reference_window(None, w, stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, window=(n, oh_lo, oh_hi, 0, ow),
                                        shape=shape, slab_of=slab_of)
        parts.append(block[0, c].reshape(hi - lo, ow))
    if not parts:
        raise ValueError(f"the rows {rows} are empty")
    return np.concatenate(parts, axis=0)


def conv2d_rows(named: dict, slab_of, weight_f64, rows) -> np.ndarray:
    """The reference for the flat output rows `rows = (r0, r1)` of `conv2d_forward_kernel`,
    the output read as `[N*Cout*Ho, Wo]` — the form the runtime screen windows every kernel in.
    Reads only the receptive field of those rows through `slab_of`. Raises `ValueError` with
    the reason when it cannot: a declaration that contradicts the arrays it names is a finding
    the screen must say out loud, never a silent consensus."""
    return _rows_of(_conv_geometry(named), slab_of, weight_f64, rows)


def depthwise_rows(named: dict, slab_of, weight_f64, rows) -> np.ndarray:
    """The same for `depthwise_conv2d_kernel` (`groups == C`, weight `[C, KH, KW]`)."""
    return _rows_of(_depthwise_geometry(named), slab_of, weight_f64, rows)


def _row_bytes_of(geometry) -> int:
    """What ONE flat output row costs the reference on the host, from the geometry: the slab of
    input it reads — every input channel, `stride + dilation*(k-1)` input rows (the row and its
    halo), the full input width, 8 bytes an element, counted TWICE because the slab is padded
    into a second array — plus the tap patch and the output row. The screen sizes its windows by
    this, not by the output row's own bytes: a 3-channel output row is 24 KB while its
    128-channel receptive field is a megabyte, and a window sized to the former reads
    gigabytes."""
    shape, w_shape, stride, _padding, dilation, groups, _oh, ow = geometry
    c_in, w_in = shape[1], shape[3]
    c_out, k_h = w_shape[0], w_shape[2]
    rows_read = max(1, int(stride[0])) + int(dilation[0]) * (k_h - 1)
    return 8 * (2 * c_in * rows_read * w_in + (c_in // groups + c_out) * ow)


class RowOracle(NamedTuple):
    """One kernel's row-windowed reference: the rows function, the names of its INPUT and WEIGHT
    arguments, the `[N, C, H, W]` shape of the input from the named arguments, and what one
    flat output row costs the reference on the host. Everything the screen needs to window the
    kernel is in this row — no branch on the kernel's name lives outside the table."""
    rows: Any
    input_name: str
    weight_name: str
    input_shape: Any
    row_bytes: Any


def oracle_row_bytes(kernel: str, named: dict) -> Optional[int]:
    """What one flat output row costs `kernel`'s reference, or None for a kernel this table does
    not window. A kernel the table DOES window whose cost cannot be formed raises `ValueError`:
    that is a refusal, and a refusal sized to the output row would be the Mac's defect one layer
    down."""
    entry = ROW_ORACLES.get(kernel)
    if entry is None:
        return None
    try:
        return int(entry.row_bytes(named))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{kernel}: the row cost of its reference cannot be formed from the "
                         f"named arguments ({type(exc).__name__}: {exc})") from None


def depthwise_reference(x: np.ndarray, w: np.ndarray, *,
                        stride: Sequence[int], padding: Sequence[int]) -> np.ndarray:
    """`[N, C, H, W]` ⊛ `[C, KH, KW]` → `[N, C, Ho, Wo]`, fp64.

    The `groups == C` case, which the engine routes to its own kernel because a
    grouped convolution with one channel per group is a stencil and not a GEMM
    (453x measured on a Sana 4Kpx VAE op). The reference keeps them separate for
    the same reason the kernels are separate: a shared path would let a defect in
    one hide behind the other.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 4 or w.ndim != 3:
        raise ValueError(f"expected [N,C,H,W] and [C,KH,KW], got {x.shape} {w.shape}")
    n, c, h, width = x.shape
    c_w, k_h, k_w = w.shape
    if c_w != c:
        raise ValueError(f"weight declares {c_w} channels, the input has {c}")
    s_h, s_w = int(stride[0]), int(stride[1])
    p_h, p_w = int(padding[0]), int(padding[1])
    out_h = _out_extent(h, p_h, 1, k_h, s_h)
    out_w = _out_extent(width, p_w, 1, k_w, s_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"the geometry produces a {out_h}x{out_w} output")

    padded = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
    out = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    for i in range(k_h):
        rows = slice(i, i + (out_h - 1) * s_h + 1, s_h)
        for j in range(k_w):
            cols = slice(j, j + (out_w - 1) * s_w + 1, s_w)
            out += padded[:, :, rows, cols] * w[None, :, i, j, None, None]
    return out


def _read(named: dict, key: str, to_f64) -> Optional[np.ndarray]:
    tensor = named.get(key)
    if tensor is None:
        return None
    return to_f64(tensor)


def conv2d_from_args(named: dict, to_f64) -> Optional[np.ndarray]:
    """The reference output for `conv2d_forward_kernel`, or None.

    `to_f64` is the caller's one host-copy path — the provider owns it, because
    reading device memory correctly is a question this module must not answer
    twice (a raw device read was measured returning bytes a kernel had not yet
    landed, and it contradicted every correct candidate as loudly as a wrong
    one).
    """
    try:
        x = _read(named, "input_pointer", to_f64)
        w = _read(named, "weight_pointer", to_f64)
        if x is None or w is None:
            return None
        shape = (int(named["batch_dim"]), int(named["in_feat_dim"]),
                 int(named["in_height"]), int(named["in_width"]))
        groups = int(named.get("groups", 1))
        k_h, k_w = int(named["kernel_height"]), int(named["kernel_width"])
        w_shape = (int(named["out_feat_dim"]), shape[1] // groups, k_h, k_w)
        if x.size != int(np.prod(shape)) or w.size != int(np.prod(w_shape)):
            # A declared extent that contradicts the array it names. Refuse:
            # reshaping anyway would produce a confident wrong reference.
            return None
        out = conv2d_reference(
            x.reshape(shape), w.reshape(w_shape),
            stride=(int(named["stride_height"]), int(named["stride_width"])),
            padding=(int(named["padding_height"]), int(named["padding_width"])),
            dilation=(int(named.get("dilation_height", 1)),
                      int(named.get("dilation_width", 1))),
            groups=groups)
        declared = (int(named["batch_dim"]), int(named["out_feat_dim"]),
                    int(named["out_height"]), int(named["out_width"]))
        if out.shape != declared:
            # The kernel and the formula disagree about the output geometry.
            # That is a finding, not something to paper over with a reshape.
            return None
        return out
    except (KeyError, TypeError, ValueError):
        return None


def depthwise_from_args(named: dict, to_f64) -> Optional[np.ndarray]:
    """The reference output for `depthwise_conv2d_kernel`, or None."""
    try:
        x = _read(named, "x_ptr", to_f64)
        w = _read(named, "w_ptr", to_f64)
        if x is None or w is None:
            return None
        shape = (int(named["N"]), int(named["C"]),
                 int(named["H_in"]), int(named["W_in"]))
        w_shape = (int(named["C"]), int(named["kh"]), int(named["kw"]))
        if x.size != int(np.prod(shape)) or w.size != int(np.prod(w_shape)):
            return None
        out = depthwise_reference(
            x.reshape(shape), w.reshape(w_shape),
            stride=(int(named["stride_h"]), int(named["stride_w"])),
            padding=(int(named["pad_h"]), int(named["pad_w"])))
        declared = (int(named["N"]), int(named["C"]),
                    int(named["H_out"]), int(named["W_out"]))
        if out.shape != declared:
            return None
        return out
    except (KeyError, TypeError, ValueError):
        return None


#: What a provider's `ORACLES` table absorbs: kernel short name -> (reference
#: from the live named arguments, the name of its OUTPUT argument).
#:
#: The output argument name is not decoration. A provider that matched the
#: output by byte length once placed the reference on an INPUT of the same size
#: and refused ten correct candidates of a shape whose certified deviation is
#: 1.7e-06. It is matched by ADDRESS, and this table is what says which address.
ORACLES = {
    "conv2d_forward_kernel": (conv2d_from_args, "output_pointer"),
    "depthwise_conv2d_kernel": (depthwise_from_args, "out_ptr"),
}

#: The same kernels on FLAT OUTPUT ROWS, for the runtime screen's windows.
ROW_ORACLES = {
    "conv2d_forward_kernel": RowOracle(
        conv2d_rows, "input_pointer", "weight_pointer",
        lambda named: _conv_geometry(named)[0],
        lambda named: _row_bytes_of(_conv_geometry(named))),
    "depthwise_conv2d_kernel": RowOracle(
        depthwise_rows, "x_ptr", "w_ptr",
        lambda named: _depthwise_geometry(named)[0],
        lambda named: _row_bytes_of(_depthwise_geometry(named))),
}
