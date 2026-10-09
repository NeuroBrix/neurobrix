"""Prism's plan must depend on what was actually asked.

Measured 2026-09-17 on hat-s-x4 (M4 Pro, 26 GB unified), BEFORE this fix:

    --input-image apple_448.png      planned 278 MB
    --input-image apple_160x112.png  planned 278 MB

22.4x the pixels and not one byte of difference. Two causes, both general:

  1. the CLI never read the request's image size at all — height/width fell
     through to a cached or family default, or 1024;
  2. every graph on this rack records TRACE literals, not symbols. All four
     image models declare height/width in `symbolic_context` and ZERO of their
     tensors use them:
        real-esrgan-x2 0/1800   hat-s-x4 0/4660
        swin2SR        0/5163   swinir   0/4175
     so the profiler sizes every activation at the trace, whatever was asked.

Note what the second fact does NOT license: refusing a model because its graph
is frozen. real-esrgan-x2 has the same 0/1800 and renders 320x224 correctly from
a 112x80 trace — the runtime resolves real shapes regardless. A "refuse when
frozen" rule would have refused four working models. So the estimate is SCALED
by the pixel ratio the trace and the request imply, not refused.

Why it matters: the per-cell memory gate is fed this number. Fed 278 MB it can
refuse nothing — 278 x 1.25 fits any machine — and hat-s-x4 went on to hold
8408 MB live and take the machine to 127 MB before the OS killed it.
"""
from __future__ import annotations

import pytest


def test_scaling_is_derived_from_the_graph_and_the_request():
    from neurobrix.core.prism.solver import PrismSolver

    class _Comp:
        graph = {"symbolic_context": {"symbols": {
            "s0": {"name": "batch", "trace_value": 1},
            "s1": {"name": "height", "trace_value": 112},
            "s2": {"name": "width", "trace_value": 80},
        }}}

    class _Req:
        def __init__(self, h, w):
            self.height, self.width = h, w

    solver = PrismSolver()
    base = 100_000_000
    # 448x448 against a 112x80 trace is 200704/8960 = 22.4x
    scaled = solver._scale_activations_to_request(_Comp(), base, _Req(448, 448))
    assert scaled == int(base * (448 * 448) / (112 * 80)), scaled
    assert scaled > base * 22


def test_a_request_at_or_below_the_trace_is_not_shrunk():
    """A smaller request does not get a smaller promise than the profiler made."""
    from neurobrix.core.prism.solver import PrismSolver

    class _Comp:
        graph = {"symbolic_context": {"symbols": {
            "s1": {"name": "height", "trace_value": 112},
            "s2": {"name": "width", "trace_value": 80},
        }}}

    class _Req:
        height, width = 64, 64

    assert PrismSolver()._scale_activations_to_request(_Comp(), 5_000, _Req()) == 5_000


@pytest.mark.parametrize("graph", [
    {}, {"symbolic_context": {}}, {"symbolic_context": {"symbols": {}}},
    {"symbolic_context": {"symbols": {"s1": {"name": "height"}}}},   # no trace_value
])
def test_a_graph_that_declares_nothing_is_left_alone(graph):
    from neurobrix.core.prism.solver import PrismSolver

    class _Comp:
        pass

    class _Req:
        height, width = 448, 448

    c = _Comp()
    c.graph = graph
    assert PrismSolver()._scale_activations_to_request(c, 777, _Req()) == 777


def test_the_gate_carries_no_per_model_table():
    """A table keyed by model name only protects models that already crashed."""
    import pathlib
    tools = pathlib.Path(__file__).resolve().parents[3] / "tools"
    assert not (tools / "measured_peaks.json").exists()
    src = (tools / "apple_matrix_percell.py").read_text()
    assert "measured_peak" not in src
    assert "measured_peaks.json" not in src


def _batch_graph():
    return {"symbolic_context": {"symbols": {"s0": {"name": "batch", "trace_value": 1}}},
            "tensors": {"t": {"symbolic_shape": {"dims": [{"type": "symbol", "id": "s0", "trace": 1}]}}}}


def test_a_component_outside_the_loop_is_scaled_to_the_request_s_own_batch():
    """`batch_size` is the guidance batch (the request's batch times `guidance_passes`); only the
    loop runs it. mochi-1-preview's VAE, traced at batch 1, priced 7 560 MB for a 4 706 MB decode
    scaled to the guidance batch of 2 (campaigns/2026_10_09_transient_proof/run_A_mochi_vae.log)."""
    from neurobrix.core.prism.profiler import InputConfig
    from neurobrix.core.prism.solver import PrismSolver

    class _Comp:
        graph = _batch_graph()

    s, base = PrismSolver(), 100_000_000
    assert s._scale_activations_to_request(_Comp(), base, InputConfig(batch_size=2, guidance_passes=2)) == base
    # the request's OWN batch of two still scales: two latents decoded
    assert s._scale_activations_to_request(_Comp(), base, InputConfig(batch_size=4, guidance_passes=2)) == 2 * base
    assert s._scale_activations_to_request(_Comp(), base, InputConfig(batch_size=2)) == 2 * base


def test_a_symbol_the_flow_binds_is_not_scaled_again():
    """The flow binds the loop denoiser's CFG batch (`FlowBindings.overrides`); the profiler priced
    it there, so the scaler leaves it — even when it equals the trace value."""
    from neurobrix.core.prism.profiler import InputConfig
    from neurobrix.core.prism.solver import PrismSolver

    class _Flow:
        def overrides(self, dag, ic):
            return {"s0": 1}

    class _Comp:
        graph = _batch_graph()

    base = 100_000_000
    assert PrismSolver()._scale_activations_to_request(
        _Comp(), base, InputConfig(batch_size=2, flow=_Flow())) == base
