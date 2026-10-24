"""Point B, step 3 (docs/design/prism-time-score.md §4): the solver estimates every ranked candidate's seconds for the
request from its components' work, the flow's uses and the profile's MEASURED facts, and says it beside today's score;
the choice is unchanged. Without measured facts nothing is estimated; a flow whose loop has no step count is named.

What these tests would do if the code were wrong: an estimate invented on a profile without facts fails the first
case; zero3 priced like a resident plan (its weights not crossing the link every step) fails the second; a loop timed
without its step count passes where the third expects the refusal's name.
"""
import json
from types import SimpleNamespace as NS

from neurobrix.core.prism.profiler import InputConfig
from neurobrix.core.prism.solver import PrismSolver

DAG = {"tensors": {"x": {"shape": [1, 64, 32], "dtype": "float16"},
                   "w": {"shape": [32, 32], "dtype": "float16", "is_parameter": True},
                   "y": {"shape": [1, 64, 32], "dtype": "float16"}},
       "ops": {"mm": {"op_type": "aten::matmul", "input_tensor_ids": ["x", "w"], "output_tensor_ids": ["y"]}},
       "execution_order": ["mm"]}
MEASURED = {"matmul_tflops": {"float16": 100.0}, "memory_gbps": 900.0, "h2d_gbps": 12.0}
RANKED = [(1000.0, "single_gpu", {"transformer": "cuda:0"}, [0]),
          (300.0, "zero3", {"transformer": "zero3:cuda:0"}, [0])]


def _solver(tmp_path, flow):
    (tmp_path / "topology.json").write_text(json.dumps({"flow": flow}))
    s = PrismSolver.__new__(PrismSolver)
    s._neural_components = [NS(name="transformer", graph=DAG)]
    return s, NS(cache_path=str(tmp_path))


def _profile(measured):
    return NS(devices=[NS(index=0, measured=measured)], host_measured=None)


def test_nothing_is_estimated_on_a_profile_without_measured_facts(tmp_path):
    s, container = _solver(tmp_path, {"loop": {"components": ["transformer"]}})
    assert s._estimate_times(RANKED, container, InputConfig(steps=4), _profile(None), "float16") == {}


def test_a_plan_that_streams_its_weights_every_step_is_slower_than_the_resident_one(tmp_path):
    s, container = _solver(tmp_path, {"loop": {"components": ["transformer"]}})
    t = s._estimate_times(RANKED, container, InputConfig(steps=4), _profile(MEASURED), "float16")
    assert set(t) == {"single_gpu", "zero3"} and all(isinstance(v, float) for v in t.values())
    load = 32 * 32 * 2 / 12e9                                   # the weight over the host link, once
    assert t["zero3"] - t["single_gpu"] > 2.9 * load            # three more crossings over four steps


def test_a_loop_without_a_step_count_is_named_not_timed(tmp_path):
    s, container = _solver(tmp_path, {"loop": {"components": ["transformer"]}})
    t = s._estimate_times(RANKED, container, InputConfig(), _profile(MEASURED), "float16")
    assert set(t) == {"single_gpu", "zero3"}
    assert all(v.startswith("not estimated:") and "step count" in v for v in t.values())
