"""Layer norm and RMS norm run their certified configuration (workstream 1): the key their wrapper forms at the launch
(`triton/autotune_cache.key_of`, the launcher's own) IS the key `launch_keys` computes from shapes and dtypes — the
function the derived census calls. On the device: every shape, flag and dtype below, wrapper against key function.

Injection (seen red, then restored green, 2026-10-22): the layer norm's key built without the bias flag -> red."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from neurobrix.kernels import launch_keys as LK  # noqa: E402
from neurobrix.kernels import wrappers as W  # noqa: E402
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype  # noqa: E402
from neurobrix.triton.autotune_cache import key_of  # noqa: E402


class _Launched(Exception):
    """The launch reached: its key is read, the sweep never runs (the key is what is judged here)."""


def _capture(monkeypatch, tuner):
    seen = []

    def run(*args, **kwargs):
        seen.append(key_of(tuner, args, kwargs))
        raise _Launched
    monkeypatch.setattr(tuner, "run", run)
    return seen


def _t(shape, dt=np.float32):
    return NBXTensor.from_numpy(np.random.default_rng(0).standard_normal(shape).astype(dt)).to("cuda")


@pytest.mark.parametrize("rows,feat", [(7, 40), (2048, 576), (1, 4096), (300, 1280)])
@pytest.mark.parametrize("weight,bias", [(True, True), (True, False), (False, False)])
def test_the_layer_norm_launch_is_its_key(monkeypatch, rows, feat, weight, bias):
    from neurobrix.kernels.ops.layernorm import layer_norm_forward_kernel as K
    seen = _capture(monkeypatch, K)
    x = _t((rows, feat))
    w = _t((feat,)) if weight else None
    b = _t((feat,)) if bias else None
    with pytest.raises(_Launched):
        W.native_layer_norm(x, [feat], w, b)
    F = NBXDtype.float32
    want = LK.layer_norm_launches(rows, feat, weight, bias, F, F if weight else None, F if bias else None, F)
    assert seen == [want[0][1]], (seen, want)


@pytest.mark.parametrize("rows,feat", [(7, 40), (2048, 576), (1, 4096), (300, 2240)])
@pytest.mark.parametrize("weight", [True, False])
def test_the_rms_norm_launch_is_its_key(monkeypatch, rows, feat, weight):
    from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel as K
    seen = _capture(monkeypatch, K)
    x = _t((rows, feat))
    w = _t((feat,)) if weight else None
    with pytest.raises(_Launched):
        W.rms_norm(x, w)
    F = NBXDtype.float32
    want = LK.rms_norm_launches(rows, feat, weight, F, F if weight else None, F)
    assert seen == [want[0][1]], (seen, want)
