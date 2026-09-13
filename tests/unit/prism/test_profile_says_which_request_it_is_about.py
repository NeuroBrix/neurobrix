"""A peak in gigabytes is a property of a REQUEST, not of a model.

`estimate_peak_memory()` with no argument used to build `InputConfig()` — a
1024x1024 batch-2 image request — and apply it to whatever graph it held,
through the positional base that calls `s1` a latent height. For a video
component that binds the TIME axis to a spatial extent, and nothing in the
returned profile said which request the number was about.

Two things are pinned here: the default now binds to the extents the container
was traced at (the one configuration it is known to have witnessed) and says so;
and the ratio between the two bindings is an instrument in its own right.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_profile_says_which_request_it_is_about.py
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig


def _graph(trace_time: int, coefficient: int):
    """A minimal encoder whose temporal extent follows `k*s - (k-1)`.

    `k = 1` is the sound rule: the extent IS the axis. A large k is the
    compounded rule, and the OFFSET is the whole point — `2187*s - 2184` equals
    3 at s = 1, exactly like the sound `s + 2` it replaced, and 26247 at s = 13.
    A bare `2187*s` would be caught by anyone reading the trace shape; it is the
    compensating offset that makes the rule invisible at the trace point and
    steep one step away, which is why the instrument has to be a RATIO between
    two bindings and not a look at either one.
    """
    # Every node carries its own `trace` annotation: the resolver has a TRUST
    # GATE and uses an expression only when that annotation reproduces the
    # concrete dim at the same index. A fixture without it is silently resolved
    # from the concrete shape, so the profile does not move with the request and
    # the test measures the fallback instead of the feature.
    time = {"type": "symbol", "id": "s1", "trace": trace_time}
    if coefficient == 1:
        dim = time
    else:
        dim = {"type": "add",
               "left": {"type": "mul", "left": time, "right": coefficient,
                        "trace": trace_time * coefficient},
               "right": -(coefficient - 1),
               "trace": trace_time * coefficient - (coefficient - 1)}
    extent = trace_time * coefficient - (coefficient - 1)
    return {
        "version": "0.1",
        "symbolic_context": {"symbols": {
            "s0": {"name": "batch", "trace_value": 1},
            "s1": {"name": "time", "trace_value": trace_time},
        }},
        "tensors": {
            "input::x": {"shape": [1, 8, extent, 64, 64],
                         "dtype": "float16",
                         "symbolic_shape": {"dims": [{"type": "symbol", "id": "s0", "trace": 1},
                                                     8, dim, 64, 64]}},
            "aten.add::0::out_0": {"shape": [1, 8, extent, 64, 64],
                                   "dtype": "float16",
                                   "symbolic_shape": {"dims": [{"type": "symbol", "id": "s0", "trace": 1},
                                                               8, dim, 64, 64]}},
        },
        "ops": {"aten.add::0": {"op_type": "aten::add",
                                "input_tensor_ids": ["input::x"],
                                "output_tensor_ids": ["aten.add::0::out_0"]}},
        # The simulation walks `execution_order`. Without it the profiler runs
        # zero ops and reports a peak of zero -- which is a profile that
        # measured nothing and reads like a component that costs nothing.
        "execution_order": ["aten.add::0"],
    }


def _profiler(tmp_path, graph):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(graph))
    return ActivationProfiler.from_path(p)


def test_no_config_binds_to_the_trace_and_says_so(tmp_path):
    prof = _profiler(tmp_path, _graph(trace_time=9, coefficient=1))
    result = prof.estimate_peak_memory()
    assert result.binding == "trace"
    assert result.symbol_map == {"s0": 1, "s1": 9}


def test_a_supplied_config_is_marked_as_a_request(tmp_path):
    prof = _profiler(tmp_path, _graph(trace_time=9, coefficient=1))
    result = prof.estimate_peak_memory(InputConfig(num_frames=33))
    assert result.binding == "request"


def test_the_repr_carries_the_binding(tmp_path):
    """A number travels; it must not travel without its request."""
    prof = _profiler(tmp_path, _graph(trace_time=9, coefficient=1))
    assert "trace-bound" in repr(prof.estimate_peak_memory())


def test_a_graph_with_no_trace_value_is_refused_not_guessed(tmp_path):
    """A partial symbol map is completed by a positional guess downstream."""
    graph = _graph(trace_time=9, coefficient=1)
    graph["symbolic_context"]["symbols"]["s1"].pop("trace_value")
    prof = _profiler(tmp_path, graph)
    with pytest.raises(ValueError, match="no trace value"):
        prof.estimate_peak_memory()


def test_a_compounded_rule_is_an_anomaly_and_a_sound_one_is_not(tmp_path):
    """The positive and negative control, in the shape of the real case.

    Both graphs are traced at s = 1 and profiled at a request of 9. The sound
    one grows 9x, as its extents justify. The compounded one is 3 at the trace
    and 17499 at the request, and the ratio says so.
    """
    request = InputConfig(num_frames=33)          # latent (33-1)//4+1 = 9

    sound_dir = tmp_path / "sound"
    sound_dir.mkdir()
    clean = _profiler(sound_dir, _graph(trace_time=1, coefficient=1)).growth_anomaly(request)
    assert not clean["exceeds"], clean
    assert clean["measured_ratio"] == pytest.approx(clean["expected_linear"], rel=0.2)

    bad_dir = tmp_path / "compound"
    bad_dir.mkdir()
    bad = _profiler(bad_dir, _graph(trace_time=1, coefficient=2187)).growth_anomaly(request)
    assert bad["exceeds"], bad
    # 2187*9 - 2186 = 19497 against 3 at the trace: three orders of magnitude
    # above the bound of 2 * 9**2 = 162.
    assert bad["measured_ratio"] > 100 * bad["bound"], bad
