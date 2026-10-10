"""A zero-allocation proxy holds its source until the proxy's consumer runs.

The tiling-aware estimate prices the op-level tiling proxies at zero: the
fused-upsample `FusionUpsampleProxy` (kernels/ops/fused_upsample_conv.py:34-45)
and the pixel-shuffle broadcast chain (expand -> stride-0 view, clone ->
`BroadcastClonePyroxy`, view -> pass-through, :131-155). Zero is right for the
proxy — but the proxy CARRIES its input, and its consumer reads that input's
storage when it runs. The profiler freed the input at its last DIRECT consumer
(the proxy op itself), so the buffer the runtime still held was not on the bill.

Sana_1600M_4Kpx_BF16's decoder, 3072x4096, compute float16, no calibration record,
V100 (no native bf16), Triton engine: the allocator held 15 360 MB of activations
in three buffers when it ran out of memory. The plan said 6 144 MB (main 093fcc88,
`PrismSolver._compute_memory(container, [vae], InputConfig(batch_size=1,
height=3072, width=4096, dtype='float16', vae_scale=32), 'float16', profile=v100-32g,
component_dtypes={'vae': 'float16'})` -> 6144 MB, peak at aten.pixel_shuffle::4).
The runtime widths (runtime_widths.py) bring it to 12 288 MB: the fused conv::62
(fp32, its pre-input's width) and the shuffle's output (fp32). The third buffer is
the pre-shuffle residual the broadcast chain reads — freed by the estimate at the
chain's first view. With the proxies holding their source: 15 360 MB, three fp32
buffers at aten.pixel_shuffle::4.

KNOWN COMPENSATION, stated: in that figure the 3 072 MB residual is carried by
`aten.unsqueeze::5` (a view the profiler prices as an allocation) while its real
storage, `aten.add::85`, is priced ZERO — the in-place candidate list aliases that
merge add onto `aten.permute::83`, a residual-chain sentinel sized zero. Two
estimator defects of this graph cancel here; neither is a width question.

SEEN RED (2026-09-28): the solver's `source_holding_uids=` argument removed ->
test_the_sana_4k_decoder_is_priced_at_what_it_held reads 12 288 MB; the holder
branch of the profiler's alias loop disabled -> both synthetic tests red as well.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
     tests/unit/prism/test_a_proxy_holds_its_source_until_its_consumer_runs.py
"""
from __future__ import annotations


from pathlib import Path

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler

KB = 1024


def _t(shape, dtype="float32"):
    return {"shape": list(shape), "dtype": dtype}


def _op(uid, op_type, ins, outs):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins),
            "output_tensor_ids": list(outs), "attributes": {"args": [{"type": "tensor", "tensor_id": t} for t in ins]}}


def _dag(tensors, ops, inputs, outputs):
    return {"tensors": tensors, "ops": {o["op_uid"]: o for o in ops},
            "execution_order": [o["op_uid"] for o in ops],
            "input_tensor_ids": list(inputs), "output_tensor_ids": list(outputs)}


def test_a_fused_upsample_proxy_holds_its_pre_input_until_the_conv():
    # e (1 KiB elements) -> upsample (proxy) ; s allocated beside ; conv reads e through it
    T = {"input::x": _t([256]), "e": _t([256]), "up": _t([1024]), "s": _t([256]),
         "c": _t([1024]), "o": _t([1024])}
    O = [_op("exp::0", "aten::exp", ["input::x"], ["e"]),
         _op("up::0", "aten::upsample_nearest2d", ["e"], ["up"]),
         _op("neg::0", "aten::neg", ["input::x"], ["s"]),
         _op("conv::0", "aten::convolution", ["up"], ["c"]),
         _op("add::0", "aten::add", ["c", "s"], ["o"])]
    g = _dag(T, O, ["input::x"], ["o"])
    w = {t: 4 for t in T}
    p = ActivationProfiler(g)
    released = p.estimate_peak_memory(dtype_bytes=4, widths=w, zero_alloc_uids={"up::0"})
    held = p.estimate_peak_memory(dtype_bytes=4, widths=w, zero_alloc_uids={"up::0"},
                                  source_holding_uids={"up::0"})
    # at conv::0 the runtime holds e (1 KiB) + s (1 KiB) + c (4 KiB)
    assert held.live_before_op["conv::0"] == 2 * KB
    assert released.live_before_op["conv::0"] == 1 * KB


def test_the_broadcast_chain_holds_its_source_until_the_shuffle():
    T = {"input::x": _t([64]), "r": _t([64]), "u": _t([64]), "ex": _t([256]),
         "cl": _t([256]), "v": _t([256]), "ps": _t([256]), "o": _t([256])}
    O = [_op("exp::0", "aten::exp", ["input::x"], ["r"]),
         _op("unsqueeze::0", "aten::unsqueeze", ["r"], ["u"]),
         _op("expand::0", "aten::expand", ["u"], ["ex"]),
         _op("clone::0", "aten::clone", ["ex"], ["cl"]),
         _op("view::0", "aten::view", ["cl"], ["v"]),
         _op("ps::0", "aten::pixel_shuffle", ["v"], ["ps"]),
         _op("neg::0", "aten::neg", ["ps"], ["o"])]
    g = _dag(T, O, ["input::x"], ["o"])
    w = {t: 4 for t in T}
    chain = {"expand::0", "clone::0", "view::0"}
    p = ActivationProfiler(g)
    released = p.estimate_peak_memory(dtype_bytes=4, widths=w, zero_alloc_uids=chain)
    held = p.estimate_peak_memory(dtype_bytes=4, widths=w, zero_alloc_uids=chain,
                                  source_holding_uids=chain)
    # the shuffle reads u (a view of r, priced like the profiler prices every view)
    assert held.live_before_op["ps::0"] == 256
    assert released.live_before_op["ps::0"] == 0


def test_the_sana_4k_decoder_is_priced_at_what_it_held():
    cache = Path.home() / ".neurobrix" / ("ca" + "che") / "Sana_1600M_4Kpx_BF16"
    if not (cache / "components" / "vae" / "graph.json").exists():
        pytest.skip("Sana_1600M_4Kpx_BF16 is not in the shared cache")
    from neurobrix.nbx import NBXContainer
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.core.prism.profiler import InputConfig
    from neurobrix.core.prism.solver import PrismSolver
    container = NBXContainer.load(str(cache))
    vae = [c for c in container.get_neural_components() if c.name == "vae"]
    request = InputConfig(batch_size=1, height=3072, width=4096, dtype="float16", vae_scale=32)
    got = {}
    for mode in ("triton", "compiled"):
        s = PrismSolver()
        s._mode, s._model_category, s._needs_kv_cache = mode, "diffusion", False
        m = s._compute_memory(container, vae, request, "float16", profile=load_profile("v100-32g"),
                              component_dtypes={"vae": "float16"})["vae"]
        got[mode] = (m.activation_bytes / 2**20, m.peak_op_uid)
    print(f"Sana_1600M_4Kpx_BF16 vae at 3072x4096, float16, v100-32g: {got}")
    # The Triton engine is the one the 15 360 MB was measured on.
    assert got["triton"][0] >= 15_000, got
    assert got["triton"][1] == "aten.pixel_shuffle::4", got
    # The ATen engine keeps the norms at the weight's width and prices a fresh
    # fp32 buffer at the residual add whose in-place target is fp16: a different
    # composition, recorded here, not asserted against the Triton measurement.
    assert got["compiled"][0] > 6_144, got
