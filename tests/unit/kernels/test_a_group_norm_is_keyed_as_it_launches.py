"""The group norm runs its certified configuration (chantier 1): `group_norm_wrapper` launches
`group_norm_forward_kernel` with the key `launch_keys.group_norm_launches` computes from shapes, flags and dtypes —
the function the derived census calls — and its result is the fp64 group norm's.

What this test would do if the code were wrong: the wrapper launching a fixed chunk never asks the tuner — red at
`_Launched`; a key formed on another extent than the group's (C x HxW / groups) or with the dtypes out of order —
red at the comparison; a chunk that drops part of a group — red at the values."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from nbxtensor.ops import launch_keys as LK  # noqa: E402
from nbxtensor.ops import wrappers as W  # noqa: E402
from nbxtensor.ops import norm_wrappers as _nbx_norm_wrappers  # the family the moved names live in (M2 split)
from nbxtensor.tensor import NBXTensor  # noqa: E402
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.autotune_cache import key_of  # noqa: E402

F32, F16 = NBXDtype.float32, NBXDtype.float16


class _Launched(Exception):
    pass


def _t(a, dt):
    x = NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("cuda")
    return x if dt == F32 else x.to(dt)


@pytest.mark.parametrize("shape,groups", [((1, 128, 64, 64), 32), ((2, 64, 9, 7), 8), ((1, 512, 16, 16, 4), 32)])
@pytest.mark.parametrize("dt", [F32, F16])
@pytest.mark.parametrize("affine", [True, False])
def test_the_launch_is_its_key(monkeypatch, shape, groups, dt, affine):
    from nbxtensor.ops.groupnorm import group_norm_forward_kernel as K_
    seen = []

    def run(*args, **kwargs):
        seen.append(key_of(K_, args, kwargs))
        raise _Launched
    monkeypatch.setattr(K_, "run", run)
    rng = np.random.default_rng(0)
    x = _t(rng.standard_normal(shape), dt)
    w = _t(rng.standard_normal(shape[1]), dt) if affine else None
    b = _t(rng.standard_normal(shape[1]), dt) if affine else None
    with pytest.raises(_Launched):
        _nbx_norm_wrappers.group_norm_wrapper(x, groups, w, b)
    hxw = int(np.prod(shape[2:]))
    want = LK.group_norm_launches(shape[0], shape[1], hxw, groups, affine, affine, dt, dt if affine else None,
                                  dt if affine else None)
    assert want[0][0] == LK.GROUP_NORM and seen == [want[0][1]], (seen, want)


@pytest.mark.parametrize("shape,groups", [((1, 64, 40, 40), 32), ((2, 32, 128, 129), 4)])
def test_the_values_are_the_group_norm(shape, groups):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(shape).astype(np.float32)
    w, b = rng.standard_normal(shape[1]).astype(np.float32), rng.standard_normal(shape[1]).astype(np.float32)
    y = _nbx_norm_wrappers.group_norm_wrapper(_t(a, F32), groups, _t(w, F32), _t(b, F32))[0].to("cpu").numpy()
    g = a.astype(np.float64).reshape(shape[0], groups, -1)
    z = (g - g.mean(-1, keepdims=True)) / np.sqrt(g.var(-1, keepdims=True) + 1e-5)
    z = z.reshape(shape) * w.reshape(1, -1, *([1] * (len(shape) - 2))) + b.reshape(1, -1, *([1] * (len(shape) - 2)))
    assert np.max(np.abs(y - z)) < 1e-4
