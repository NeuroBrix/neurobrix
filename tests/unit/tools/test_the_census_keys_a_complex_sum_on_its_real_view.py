"""A complex `aten::sum` is keyed on its real view: `wrappers.sum_wrapper` reduces `view_as_real(x)` over the same dim,
the appended [., 2] axis kept, so the real and imaginary parts are rows of their own, in the part's dtype.

Wan's rotary grid (`aten.sum::4`: [1, 30, 1, 21] complex64 summed over its last dim) launched 60 rows of 21 — the
key (64, 32, fp32, fp32) — where the census had keyed 30 rows; both Wan2.1 triton cells of the c1restlot2 gate stopped
on it (2026-10-24).

What this test would do if the code were wrong: keyed at the complex rows (30), the key's row class is 32, not 64.
"""
import collections
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as D  # noqa: E402
from neurobrix.kernels import census as _census  # noqa: E402
from nbxtensor.ops import launch_keys as LK  # noqa: E402
from nbxtensor.tensor import NBXDtype  # noqa: E402

S = lambda v: {"type": "scalar", "value": v}       # noqa: E731
T = lambda t: {"type": "tensor", "tensor_id": t}   # noqa: E731


@pytest.fixture(autouse=True)
def _the_bound_target_is_restored():
    from nbxtensor.backend import launcher
    from nbxtensor.ops import _configs
    target, active = getattr(launcher, "_TARGET", None), dict(_configs._ACTIVE_PROFILE)
    yield
    launcher._TARGET = target
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(active)


def _keys(dtype):
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)
    o = {"op_type": "aten::sum", "input_tensor_ids": ["x"], "attributes": {"args": [T("x"), S([-1]), S(True)]}}
    out = D._op_launches("aten::sum", "aten.sum::4", o, ["x"], lambda t: [1, 30, 1, 21], lambda t: dtype, LK, None,
                         "float16", False, 0, 128, 0, 1 << 30, collections.Counter())
    return [k for _q, k in out]


def test_a_complex_sum_reduces_its_real_and_imaginary_rows():
    assert _keys(NBXDtype.complex64) == [k for _q, k in LK.row_reduce_launches("sum", 60, 21, NBXDtype.float32)]


def test_a_real_sum_is_unchanged():
    assert _keys(NBXDtype.float32) == [k for _q, k in LK.row_reduce_launches("sum", 30, 21, NBXDtype.float32)]
