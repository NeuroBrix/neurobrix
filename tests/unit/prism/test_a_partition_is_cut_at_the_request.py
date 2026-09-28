"""The layer partitioner sizes activations at the REQUEST, with the profiler's own resolver.

It read each tensor's `shape` — the TRACE's — so every layer_streaming segment was cut for the trace, not
the run: SANA-Video's VAE (input [1,128,9,14,22] traced, 21x64x160 at a 1280x512 request) was declined as
"fits in ONE segment" when its activations at the request peak at 78 080 MB (measured with --explain-plan
on 16 GB, 2026-09-27). The profiler already resolved `symbolic_shape` at the request: the same bug
written twice. Before this branch the partitioner takes no symbol map: these fail.
"""
import pytest

from neurobrix.core.prism.layer_partition import LayerPartitioner


def _graph():
    sym = {"type": "symbol", "id": "s1", "trace": 10}
    t = lambda d: {"shape": [1, 10, 10], "dtype": "float32",
                   "symbolic_shape": {"dims": [1, sym, sym], "concrete": [1, 10, 10]}}
    return {"tensors": {"x": t(0), "a": t(1), "b": t(2)},
            "ops": {"op1": {"input_tensor_ids": ["x"], "output_tensor_ids": ["a"]},
                    "op2": {"input_tensor_ids": ["a"], "output_tensor_ids": ["b"]}},
            "execution_order": ["op1", "op2"]}


def test_the_trace_is_what_it_sizes_without_a_request():
    assert max(LayerPartitioner(_graph()).live_activation_curve()) == 10 * 10 * 4   # one tensor live at a time


def test_the_request_is_what_it_sizes_with_one():
    # 100 x 100 at the request, computed at the plan's width (2 bytes), as the profiler prices it.
    curve = LayerPartitioner(_graph(), symbol_map={"s1": 100}, compute_dtype_bytes=2).live_activation_curve()
    assert max(curve) == 100 * 100 * 2


def test_a_request_without_the_plans_width_is_refused():
    with pytest.raises(ValueError, match="compute dtype"):
        LayerPartitioner(_graph(), symbol_map={"s1": 100})
