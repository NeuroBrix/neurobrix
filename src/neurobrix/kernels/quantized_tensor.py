"""QuantizedTensor — the engine-side handle for an encoded weight.

A weight stored under the int4-g128-asym encoding rides the container
as a triplet (`X.qweight` int32 [K//8, N], `X.scales` / `X.qmins`
fp16 [K//group, N], K = in_features). At weight-load the triplet is
assembled into ONE QuantizedTensor registered under the graph's
expected key (`X.weight`), so binding, placement accounting and
lifecycle stay untouched — compute wrappers detect the type and route:

  aten::t   -> returns self with the view flag flipped. The packed
               layout is ALREADY the transposed orientation [K, N]
               (the dequant-GEMV family's natural layout), so the
               graph's `t(weight [N, K])` is a zero-cost marker.
  aten::mm  -> M == 1: fused dequant-GEMV (byte-gated family);
               M > 1: dequantize to dense fp32 via the family's
               standalone kernel, then the normal mm path (prefill is
               compute-bound; the transient dense buffer frees at op
               end).

Lives in kernels/ beside NBXTensor (wrappers route on this type; triton/ consumes it). R33: NBXTensor members only, zero torch. The logical metadata
(shape/dtype) mirrors what the dense weight would expose so generic
binding/accounting code needs no special cases.
"""

from typing import Tuple

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype

ENCODING = "int4-g128-asym"
GROUP = 128
PACK = 8


class QuantizedTensor:
    """int4-g128-asym weight triplet under one graph-facing handle.

    `logical_shape` is the DENSE weight's shape as the graph traced it
    ([out_features, in_features]); `transposed=True` marks the
    graph-side aten::t view ([in, out] — the packed layout's natural
    orientation)."""

    __slots__ = ("qweight", "scales", "qmins", "logical_shape",
                 "transposed")

    def __init__(self, qweight: NBXTensor, scales: NBXTensor,
                 qmins: NBXTensor, logical_shape: Tuple[int, int],
                 transposed: bool = False):
        self.qweight = qweight
        self.scales = scales
        self.qmins = qmins
        self.logical_shape = tuple(logical_shape)
        self.transposed = transposed

    # ── graph-facing metadata (mirrors the dense weight) ──
    @property
    def shape(self) -> Tuple[int, int]:
        n, k = self.logical_shape
        return (k, n) if self.transposed else (n, k)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self):
        return NBXDtype.float16  # the dense weight's traced dtype class

    @property
    def nbx_dtype(self):
        return NBXDtype.float16

    @property
    def device(self):
        return self.qweight.device

    @property
    def _device(self):
        return self.qweight._device

    @property
    def _device_idx(self):
        return self.qweight._device_idx

    @property
    def _nbytes(self) -> int:
        return (self.qweight._nbytes + self.scales._nbytes
                + self.qmins._nbytes)

    def numel(self) -> int:
        n, k = self.logical_shape
        return n * k

    def t(self) -> "QuantizedTensor":
        return QuantizedTensor(self.qweight, self.scales, self.qmins,
                               self.logical_shape,
                               transposed=not self.transposed)

    def __repr__(self) -> str:
        return (f"QuantizedTensor({ENCODING}, logical={self.logical_shape}, "
                f"transposed={self.transposed}, "
                f"packed={tuple(self.qweight.shape)})")


def assemble_quantized(weights: dict) -> int:
    """Fold `.qweight`/`.scales`/`.qmins` triplets in a loaded weights
    dict into QuantizedTensor entries under the graph key `<base>.weight`.
    Mutates the dict in place; returns the number of assembled weights.
    ZERO FALLBACK: an incomplete triplet raises."""
    bases = [k[: -len(".qweight")] for k in weights
             if k.endswith(".qweight")]
    for base in bases:
        qw = weights.pop(base + ".qweight")
        try:
            sc = weights.pop(base + ".scales")
            mn = weights.pop(base + ".qmins")
        except KeyError as e:
            raise RuntimeError(
                f"ZERO FALLBACK: incomplete quantized triplet for "
                f"'{base}' — missing {e}. The build gate should have "
                f"refused this artifact; re-build the variant.") from e
        kp, n = qw.shape
        weights[base + ".weight"] = QuantizedTensor(
            qw, sc, mn, logical_shape=(n, kp * PACK))
    return len(bases)
