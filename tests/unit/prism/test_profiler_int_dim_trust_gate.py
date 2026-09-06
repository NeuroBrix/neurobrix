"""A `symbolic_shape.dims` entry that is a bare integer must equal the trace's
concrete dim — an integer cannot vary with any symbol, so a different value
is a corrupted annotation, not information.

Found on 2026-09-06: Kokoro-82M's decoder carries `aten.convolution::19` with
`symbolic_shape.dims = [15361, 256, 2560]` against a witnessed
`[1, 256, 2560]` (the input's length written into the output's batch slot).
The trust gate only checked expression nodes, so the profiler sized that
activation at 19.2 GB, Prism planned 20 GB for an 82M model, every 16 GB card
placed the decoder on the host, and both ATen engines died on the host path.
"""

from neurobrix.core.prism.profiler import ActivationProfiler


def _profiler():
    dag = {"ops": [], "tensors": {}}
    return ActivationProfiler(dag)


def test_a_bare_integer_dim_that_contradicts_the_trace_is_refused():
    meta = {"shape": [1, 256, 2560], "symbolic_shape": {"dims": [15361, 256, 2560], "concrete": [1, 256, 2560]}}
    assert _profiler()._resolve_shape(meta, {}) == [1, 256, 2560]


def test_a_bare_integer_dim_that_matches_the_trace_passes():
    meta = {"shape": [1, 256, 2560], "symbolic_shape": {"dims": [1, 256, 2560], "concrete": [1, 256, 2560]}}
    assert _profiler()._resolve_shape(meta, {}) == [1, 256, 2560]


def test_a_symbol_node_still_evaluates_at_the_runtime_value():
    meta = {"shape": [1, 256, 23], "symbolic_shape": {"dims": [1, 256, {"type": "symbol", "id": "s1", "trace": 23}], "concrete": [1, 256, 23]}}
    assert _profiler()._resolve_shape(meta, {"s1": 448}) == [1, 256, 448]
