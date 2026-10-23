"""The looped softmax runs its certified configuration (chantier 1): a row longer than the single program holds
(`launch_keys.SOFTMAX_FUSED_MAX_FEAT`) launches `softmax_forward_looped_kernel` with the key `launch_keys.softmax_launches`
computes from shapes and dtypes — the function the derived census calls; at or under it, the single-program kernel's.

What this test would do if the code were wrong: the census still saying "no key" for a long row, or keying it on the
exact length instead of its class -> red at the key comparison; the wrapper launching a fixed tile -> no tuner run,
red at `_Launched`."""
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

F32, F16 = NBXDtype.float32, NBXDtype.float16


class _Launched(Exception):
    """The launch reached: its key is read, the sweep never runs."""


def _capture(monkeypatch, tuner):
    seen = []

    def run(*args, **kwargs):
        seen.append(key_of(tuner, args, kwargs))
        raise _Launched
    monkeypatch.setattr(tuner, "run", run)
    return seen


def _t(shape, dt):
    x = NBXTensor.from_numpy(np.random.default_rng(0).standard_normal(shape).astype(np.float32)).to("cuda")
    return x if dt == F32 else x.to(dt)


@pytest.mark.parametrize("rows,feat", [(1, 151936), (3, 16385), (2, 32000)])
@pytest.mark.parametrize("dt", [F32, F16])
@pytest.mark.parametrize("log", [False, True])
def test_a_long_row_launches_the_looped_kernel_with_its_key(monkeypatch, rows, feat, dt, log):
    from neurobrix.kernels.ops.softmax import softmax_forward_looped_kernel as K_
    seen = _capture(monkeypatch, K_)
    x = _t((rows, feat), dt)
    with pytest.raises(_Launched):
        (W.log_softmax if log else W.softmax)(x, -1)
    want = LK.softmax_launches(rows, feat, log, dt, dt)
    assert want[0][0] == LK.SOFTMAX_LOOPED and seen == [want[0][1]], (seen, want)


def test_a_row_the_single_program_holds_keeps_its_kernel():
    assert LK.softmax_launches(4, LK.SOFTMAX_FUSED_MAX_FEAT, False, F32, F32)[0][0] == LK.SOFTMAX
    assert LK.softmax_launches(4, LK.SOFTMAX_FUSED_MAX_FEAT + 1, False, F32, F32)[0][0] == LK.SOFTMAX_LOOPED
