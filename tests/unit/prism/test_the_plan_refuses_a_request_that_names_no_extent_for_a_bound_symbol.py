"""The planner used to bind a spatial or temporal symbol the request left open to the
TRACE extent and estimate on it (Wan T2V: 1.74 GiB planned for a 35 GiB decode, 2026-09-21).
Now it refuses by name. What this test would do if the code were wrong: return a map with
the trace values in it."""
from __future__ import annotations

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig
from neurobrix.core.runtime_values import MissingRuntimeValue

DAG = {
    "symbolic_context": {"symbols": {
        "s1": {"name": "time", "trace_value": 9, "source": "input::z::dim_2"},
        "s2": {"name": "height", "trace_value": 14, "source": "input::z::dim_3"},
        "s3": {"name": "width", "trace_value": 22, "source": "input::z::dim_4"}}},
    "tensors": {"input::z": {"shape": [1, 16, 9, 14, 22], "is_input": True}},
    "ops": {}, "execution_order": [], "input_tensor_ids": ["input::z"],
}


def test_a_request_without_height_refuses_by_name():
    p = ActivationProfiler(DAG)
    with pytest.raises(MissingRuntimeValue, match="s2"):
        p.build_symbol_map(InputConfig(batch_size=1, height=None, width=None, vae_scale=8,
                                       num_frames=81, temporal_compression=4), placement_floor=True)


def test_a_complete_request_binds_every_symbol_from_the_request():
    p = ActivationProfiler(DAG)
    m = p.build_symbol_map(InputConfig(batch_size=1, height=480, width=832, vae_scale=8,
                                       num_frames=81, temporal_compression=4), placement_floor=True)
    assert m["s2"] == 60 and m["s3"] == 104 and m["s1"] == 21, m
