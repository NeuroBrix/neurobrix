"""A 2-D conv op-level tiling runs band by band holds, beside its input and its whole output, its largest band's input
copied into one buffer and the band's output (`kernels/ops/fused_upsample_conv._tiled_conv2d_spatial_*`, both
engines). Prism prices it (`tiling_sizes.tiled_conv2d_band_bytes`): in the activation walk's in-op figure
(`estimate_peak_memory(tiled_convs=...)`) and on the layer partition's op curves (`LayerPartitioner(op_extra=...)`),
so a piece is cut before a banded conv its live set cannot hold beside the band.

HAT-L x4 at 2048 px, compiled, 16 GB (2026-10-22): piece 1 ended AT aten.convolution::1 with 12.7 GB live, and the
run asked the band's 482 MiB output beside its 1 446 MiB input buffer — out of memory; priced, the piece ends before it
and the values between pieces wait on the host.

Injections (seen red): the band term dropped from the walk -> red; `op_extra` ignored by the op curve -> red."""
from __future__ import annotations

from nbxtensor.ops import tiling_sizes as TS
from neurobrix.core.prism.layer_partition import LayerPartitioner
from neurobrix.core.prism.profiler import ActivationProfiler

MiB = 2 ** 20


def test_the_band_is_what_the_hat_l_run_asked():
    """[1, 180, 2048, 2048] through a 3x3, padding 1, in 2 bands at fp32: the first band reads 1 026 rows plus one
    edge row, padded to 2 050 columns — the 1 446 MiB buffer the live dump showed — and writes 1 024 output rows of 60
    channels, 480 MiB (the allocator was asked 482 MiB: it rounds a request up to its block)."""
    b = TS.tiled_conv2d_band_bytes([1, 180, 2048, 2048], [60, 180, 3, 3], [1, 1], [1, 1], [1, 1], 2, 4, 4)
    band_in = 180 * 1027 * 2050 * 4
    band_out = 60 * 1024 * 2048 * 4
    assert b == band_in + band_out
    assert round(band_in / MiB) == 1446 and round(band_out / MiB) == 480


def _t(shape, dtype="float32", **kw):
    return {"shape": list(shape), "dtype": dtype, **kw}


def _op(uid, op_type, ins, outs, **attrs):
    attrs.setdefault("args", [{"type": "tensor", "tensor_id": t} for t in ins])
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins), "output_tensor_ids": list(outs),
            "attributes": attrs}


def _graph():
    T = {"input::x": _t([1, 8, 64, 64], is_input=True), "param::w": _t([8, 8, 3, 3], is_parameter=True),
         "a": _t([1, 8, 64, 64]), "c": _t([1, 8, 64, 64]), "o": _t([1, 8, 64, 64])}
    O = [_op("relu::0", "aten::relu", ["input::x"], ["a"]),
         _op("conv::0", "aten::convolution", ["a", "param::w"], ["c"], stride=[1, 1], padding=[1, 1],
             dilation=[1, 1], groups=1),
         _op("add::0", "aten::add", ["c", "a"], ["o"])]
    return {"tensors": T, "ops": {o["op_uid"]: o for o in O}, "execution_order": [o["op_uid"] for o in O],
            "input_tensor_ids": ["input::x"], "output_tensor_ids": ["o"], "torch_dtype": "float32"}


def test_the_walk_holds_the_band_while_the_conv_runs():
    g = _graph()
    p = ActivationProfiler(g)
    plain = p.estimate_peak_memory(dtype_bytes=4)
    banded = p.estimate_peak_memory(dtype_bytes=4, tiled_convs={"conv::0": 2})
    band = TS.tiled_conv2d_band_bytes([1, 8, 64, 64], [8, 8, 3, 3], [1, 1], [1, 1], [1, 1], 2, 4, 4)
    assert banded.peak_bytes >= plain.peak_bytes and banded.peak_op_uid == "conv::0"
    assert banded.peak_bytes == max(plain.peak_bytes, _in_op_at_conv(p) + band)


def _in_op_at_conv(p):
    # the conv's in-op figure without a band: its input `a` (the graph input is its caller's) and its output `c`
    return 2 * 8 * 64 * 64 * 4


def test_the_partition_curves_carry_the_band():
    g = _graph()
    extra = {"conv::0": 12345}
    plain = LayerPartitioner(g, {})
    banded = LayerPartitioner(g, {}, op_extra=extra)
    i = g["execution_order"].index("conv::0")
    assert banded.op_peak_curve()[i] == plain.op_peak_curve()[i] + 12345
    assert [v for j, v in enumerate(banded.op_peak_curve()) if j != i] == \
        [v for j, v in enumerate(plain.op_peak_curve()) if j != i]
    assert banded._parked_op_curve()[i] == plain._parked_op_curve()[i] + 12345
