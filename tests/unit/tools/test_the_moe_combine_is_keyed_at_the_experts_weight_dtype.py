"""The fused MoE's combine — a sum over top_k of [tokens, top_k, hidden] — is keyed at the EXPERT WEIGHTS' dtype.

`triton/moe.execute_moe_fused` casts the hidden states to the gate weights' dtype before its passes, and reads the
weights by name, so a calibration island that hands the op fp32 hidden states never widens them. deepseek-moe-16b-chat:
its island gave the hidden states fp32, the run summed fp16 (43 008 x 8, fp16 in, fp32 out) and the census, keyed at the
hidden states' dtype, had written fp32 — the c1restlot gate's triton cell stopped on the missing key (2026-10-23).

What this test would do if the code were wrong: keyed at the hidden states' dtype read through the island (fp32), the
combine's key reads fp32 in and the first assertion fails.
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


@pytest.fixture(autouse=True)
def _the_bound_target_is_restored():
    from nbxtensor.backend import launcher
    from nbxtensor.ops import _configs
    target, active = getattr(launcher, "_TARGET", None), dict(_configs._ACTIVE_PROFILE)
    yield
    launcher._TARGET = target
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(active)


def test_the_combine_reads_the_weights_dtype_not_the_islands():
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)
    o = {"op_type": "custom::moe_fused", "input_tensor_ids": ["h", "g"],
         "attributes": {"hidden_states_tid": "h", "gate_scores_tid": "g", "top_k": 6, "norm_topk_prob": False,
                        "expert_gate_weight_ids": ["wg0"], "expert_up_weight_ids": ["wu0"],
                        "expert_down_weight_ids": ["wd0"]}}
    island = lambda t: NBXDtype.float32                                        # noqa: E731 — every operand widened
    loaded = lambda t: NBXDtype.float16                                        # noqa: E731 — as the loader holds it
    out = D._op_launches("custom::moe_fused", "moe_fused::block.1", o, ["h", "g"],
                         lambda t: [1, 7168, 2048] if t == "h" else [1, 7168, 64], island, LK, None, "float16",
                         False, 0, 128, 0, 1 << 30, collections.Counter(), None, None, loaded_dt=loaded)
    sums = [k for q, k in out if q.endswith("sum_forward_kernel")]
    assert sums and all(k[2] == "fp16" for k in sums), sums
    want = [k for _q, k in LK.row_reduce_launches("sum", 7168 * 2048, 6, NBXDtype.float16)]
    assert sums == want, (sums, want)
