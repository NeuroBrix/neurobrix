"""An LSTM is ONE graph op: the calibration that grants the fp16 contract never observed its internal products, so the
recurrence runs under the conservative store whatever the flag (2026-10-26: Kokoro-82M's speech became gibberish when
its predictor's LSTM steps stored fp16). What this test would do if the code were wrong: under the flag the products
store fp16 and the outputs differ from the flag-off run — RED; and a flag left changed after the call — RED."""
import os
import numpy as np
import pytest
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import wrappers as W
from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)

pytestmark = pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="needs a card")


def _t(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a.astype(np.float32))).to("cuda").to(NBXDtype.float16)


def test_the_flag_does_not_reach_the_recurrence():
    rng = np.random.default_rng(0)
    B, T, I, H = 1, 64, 32, 32
    x, wi, wh = _t(rng.standard_normal((B, T, I))), _t(rng.standard_normal((I, 4 * H)) * 0.2), _t(rng.standard_normal((H, 4 * H)) * 0.2)
    h, c = _t(np.zeros((B, H))), _t(np.zeros((B, H)))
    outs = []
    for flag in (False, True):
        W.set_activations_fp16_safe(flag)
        o, _h, _c = _nbx_elementwise_wrappers._lstm_run_direction(x, wi, wh, None, None, h, c, H, False)
        assert W.get_activations_fp16_safe() is flag
        outs.append(o.to(NBXDtype.float32).numpy() if hasattr(o, "numpy") else o)
    W.set_activations_fp16_safe(False)
    assert np.array_equal(np.asarray(outs[0]), np.asarray(outs[1]))
