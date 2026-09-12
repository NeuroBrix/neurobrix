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

from typing import Optional, Sequence

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
