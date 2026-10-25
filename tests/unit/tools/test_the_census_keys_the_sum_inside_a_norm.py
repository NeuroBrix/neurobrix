"""`aten::norm` launches a sum: `wrappers.norm_wrapper` (p = 2) is sqrt(sum(x * x, dim)), so its key is the sum's
over `dim` — the op's THIRD argument (self, p, dim, keepdim) — in its input's dtype.

The census placed aten::norm keyless: s1-mini's codec norm (1 536 rows of 8 192, fp16 in) was absent from
the table and the c1restlot2 gate's triton cell stopped on it (2026-10-24); the gate before had stopped earlier in the
same cell, on another missing sum, and hid it.

What this test would do if the code were wrong: a norm placed keyless yields no launch — the first assertion fails; a
dim read at the second argument (the sum's position: here p = 2) reduces the wrong axis — the key assertion fails.
"""
import collections
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as D  # noqa: E402
from neurobrix.kernels import census as _census  # noqa: E402
from neurobrix.kernels import launch_keys as LK  # noqa: E402
from neurobrix.kernels.nbx_tensor import NBXDtype  # noqa: E402

S = lambda v: {"type": "scalar", "value": v}       # noqa: E731
T = lambda t: {"type": "tensor", "tensor_id": t}   # noqa: E731


@pytest.fixture(autouse=True)
def _the_bound_target_is_restored():
    from neurobrix.kernels import launcher
    from neurobrix.kernels.ops import _configs
    target, active = getattr(launcher, "_TARGET", None), dict(_configs._ACTIVE_PROFILE)
    yield
    launcher._TARGET = target
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(active)


def test_a_norm_is_the_sum_of_its_input_over_its_dims():
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)
    o = {"op_type": "aten::norm", "input_tensor_ids": ["v"],
         "attributes": {"args": [T("v"), S(2), S([1, 2]), S(True)]}}
    out = D._op_launches("aten::norm", "aten.norm::0", o, ["v"], lambda t: [1536, 512, 16],
                         lambda t: NBXDtype.float16, LK, None, "float16", False, 0, 128, 0, 1 << 30,
                         collections.Counter())
    assert out, "aten::norm launched nothing"
    assert [k for _q, k in out] == [k for _q, k in LK.row_reduce_launches("sum", 1536, 512 * 16, NBXDtype.float16)]
