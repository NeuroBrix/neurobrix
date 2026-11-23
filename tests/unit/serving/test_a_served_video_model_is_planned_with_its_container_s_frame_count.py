"""A served video container is planned at load with the frame count and the temporal compression its container declares.

Before (regression 22/11/2026, Wan2.1-T2V-1.3B, `test_serve_warm.py::test_family_serve_warm[video-*]`): the load built
the plan's `InputConfig` with height, width and vae_scale from the container's defaults and no `num_frames`, while
`neurobrix run` resolves it from the container too. Every served video container then failed at the plan with
MissingRuntimeValue "the plan's request names no frame count for symbol s1 (time)".

What this stands on: `InferenceEngine.load` on a container directory holding a `runtime/defaults.json` and no weights.
The model lookup, the container object, the profile and the declared sub-components are stand-ins; `solve_served` is
replaced by a recorder that hands back the `InputConfig` the load built and stops the load there. The refusal test then
gives that same config to the profiler's own `build_symbol_map` on a graph with a `time` symbol.

What this test would do if the code were wrong: if the load leaves `num_frames` out, the recorded config carries None
and the first test is red; if the load invented a count for a container that declares none (a family default, a
literal), the second test is red, because the profiler would bind it and refuse nothing.
"""
import json
import pytest
from types import SimpleNamespace as NS


class _Stop(Exception):
    pass


def _load(tmp_path, monkeypatch, defaults):
    (tmp_path / "components").mkdir()
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps(defaults))
    import neurobrix.cli.utils as U
    import neurobrix.nbx as N
    import neurobrix.processors as PR
    import neurobrix.pytorch.prism.loader as L
    import neurobrix.pytorch.prism.served_prefill as SP
    monkeypatch.setattr(U, "find_model", lambda name: tmp_path)
    monkeypatch.setattr(N.NBXContainer, "load", staticmethod(lambda p: NS(
        _cache_path=tmp_path, get_manifest=lambda: {"family": "video", "model_name": "a video model"},
        get_neural_components=lambda: [])))
    monkeypatch.setattr(L, "launch_profile", lambda *a, **k: NS())
    monkeypatch.setattr(PR, "open_declared", lambda *a, **k: None)
    seen = []

    def record(container, hw, input_config, mode):
        seen.append(input_config)
        raise _Stop()
    monkeypatch.setattr(SP, "solve_served", record)
    from neurobrix.serving.engine import InferenceEngine
    with pytest.raises(_Stop):
        InferenceEngine("a video model", "any-profile", "compiled").load()
    return seen[0]


def test_a_served_video_model_is_planned_with_its_container_s_frame_count(tmp_path, monkeypatch):
    ic = _load(tmp_path, monkeypatch, {"height": 480, "width": 832, "vae_scale_factor": 8, "dtype": "float16",
                                       "num_frames": 81, "temporal_compression_ratio": 4})
    assert (ic.height, ic.width) == (480, 832)
    assert ic.num_frames == 81, f"the served plan names no frame count (got {ic.num_frames!r})"
    assert ic.temporal_compression == 4


def test_a_container_without_a_frame_count_is_refused_by_name_not_given_one(tmp_path, monkeypatch):
    from neurobrix.pytorch.prism.profiler import ActivationProfiler
    from neurobrix.core.runtime_values import MissingRuntimeValue
    ic = _load(tmp_path, monkeypatch, {"height": 480, "width": 832, "vae_scale_factor": 8, "dtype": "float16"})
    assert ic.num_frames is None
    profiler = object.__new__(ActivationProfiler)
    profiler.dag = {"symbolic_context": {"symbols": {"s1": {"name": "time", "trace_value": 10}}}}
    with pytest.raises(MissingRuntimeValue, match="no frame count"):
        profiler.build_symbol_map(ic)
