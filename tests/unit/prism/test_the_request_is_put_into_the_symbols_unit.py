"""Scaling a trace-sized activation estimate to the request actually made.

Three defects this file pins, all measured on CUDA on 2026-09-17 while proving
the Mac's `cc7d314b` on this rack:

1. **The scaler was orphan code.** `_scale_activations_to_request` existed and
   nothing called it, so `hat-s-x4` planned 164 MB for a 448x448 request and
   164 MB for a 160x112 one — 22.4x the pixels, not one byte of difference,
   which is the exact defect the commit exists to remove. A method with no call
   site is a fix that reads as applied.

2. **A symbol's NAME does not carry its UNIT.** Sana-1600M declares
   `height`/`width` with a trace value of 32 and a `trace_resolution` of 1024:
   the symbol counts LATENT rows, the request counts PIXELS, and 1024/32 is the
   VAE's compression, not a change of size. Taken raw it multiplied the estimate
   by 32 per axis and Prism refused, by name, a model that had rendered on a
   16 GB card that same morning: "This model cannot run on this machine."

3. **A dimension declared twice was counted twice.** PixArt-XL-1024's
   transformer declares `seq_len` at two symbol ids; the product over
   DECLARATIONS read a 2.5x request as 6.25x. Height and width are different
   dimensions and still contribute one factor each.

The shapes are not decorative. 448x448 against a 112x80 trace is deliberately
NON-square on both sides, so a bug that multiplies height by height, or that
transposes the two, cannot pass: the answer 22.4 is reachable only as
(448/112) x (448/80).
"""

import logging
import types

import pytest

from neurobrix.core.prism.profiler import InputConfig
from neurobrix.core.prism.solver import PrismSolver


def _graph(*symbols):
    """A graph declaring `symbols` as (name, trace_value), one symbol id each."""
    return {"symbolic_context": {"symbols": {
        f"s{i}": {"name": n, "trace_value": t} for i, (n, t) in enumerate(symbols)}}}


def _comp(*symbols, name="model"):
    return types.SimpleNamespace(name=name, graph=_graph(*symbols))


@pytest.fixture
def solver():
    return PrismSolver.__new__(PrismSolver)


def scale(solver, comp, ic, base=1000):
    return solver._scale_activations_to_request(comp, base, ic) / base


def test_a_model_without_a_vae_scales_on_the_raw_pixels(solver):
    # hat-s-x4's real trace: 112x80, asked for 448x448. Non-square both sides.
    got = scale(solver, _comp(("height", 112), ("width", 80)),
                InputConfig(height=448, width=448))
    assert got == pytest.approx(22.4), "(448/112) x (448/80) and nothing else"


def test_a_latent_model_at_its_trace_resolution_does_not_scale(solver):
    # Sana-1600M: symbols count latents (32), request counts pixels (1024),
    # vae_scale 32. 1024/32 == 32 == the trace, so the request IS the trace.
    got = scale(solver, _comp(("height", 32), ("width", 32)),
                InputConfig(height=1024, width=1024, vae_scale=32))
    assert got == 1.0, "the VAE's compression is not a change of size"


def test_a_latent_model_at_double_the_side_scales_by_four(solver):
    got = scale(solver, _comp(("height", 32), ("width", 32)),
                InputConfig(height=2048, width=2048, vae_scale=32))
    assert got == pytest.approx(4.0)


def test_the_unit_is_ignored_when_the_model_declares_none(solver):
    # An upscaler has no VAE, so vae_scale is None and the factor is 1 — the
    # request is already in the symbol's unit.
    got = scale(solver, _comp(("height", 112), ("width", 80)),
                InputConfig(height=448, width=448, vae_scale=None))
    assert got == pytest.approx(22.4)


def test_a_dimension_declared_twice_is_counted_once(solver):
    comp = types.SimpleNamespace(name="transformer", graph={"symbolic_context": {"symbols": {
        "a": {"name": "seq_len", "trace_value": 120},
        "b": {"name": "seq_len", "trace_value": 120}}}})
    assert scale(solver, comp, InputConfig(seq_len=300)) == pytest.approx(2.5), \
        "one dimension, however many times it is declared"


def test_height_and_width_are_different_dimensions_and_both_count(solver):
    # The other side of the dedup: it must not collapse two real dimensions.
    got = scale(solver, _comp(("height", 100), ("width", 100)),
                InputConfig(height=200, width=300))
    assert got == pytest.approx(6.0), "2 x 3, not 2 and not 3"


def test_temporal_frames_use_the_temporal_compression_not_the_spatial_one(solver):
    got = scale(solver, _comp(("num_frames", 10)),
                InputConfig(num_frames=80, temporal_compression=4, vae_scale=32))
    assert got == pytest.approx(2.0), "80/4 = 20 latent frames against a trace of 10"


def test_a_smaller_request_never_promises_less_than_was_measured(solver):
    got = scale(solver, _comp(("height", 448), ("width", 448)),
                InputConfig(height=112, width=112))
    assert got == 1.0, "the profiler measured that much; a smaller ask does not unmeasure it"


def test_a_symbol_the_request_cannot_answer_for_is_left_alone(solver):
    assert scale(solver, _comp(("mystery_dim", 7)), InputConfig(height=448)) == 1.0


def test_a_graph_with_no_symbols_is_returned_untouched(solver):
    comp = types.SimpleNamespace(name="m", graph={})
    assert solver._scale_activations_to_request(comp, 12345, InputConfig(height=448)) == 12345


def test_the_scaling_says_what_it_followed(solver, caplog):
    """It multiplies the number the per-cell memory gate consults. A decision
    that changes a plan and leaves no line behind is indistinguishable from not
    having run — `followed` was built and thrown away."""
    with caplog.at_level(logging.INFO, logger="neurobrix.core.prism.solver"):
        scale(solver, _comp(("height", 112), ("width", 80)), InputConfig(height=448, width=448))
    assert any("x22.4" in r.getMessage() for r in caplog.records), caplog.text
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "height 112->448" in said and "width 80->448" in said


def test_the_scaler_is_CALLED_by_the_planner_not_merely_defined():
    """The orphan guard. The method existed and nothing called it, so the plan
    did not scale at all while reading as though the fix had landed. This asserts
    the seam, which no test of the method itself can do (register 17)."""
    import inspect

    from neurobrix.core.prism import solver as solver_mod

    src = inspect.getsource(solver_mod)
    calls = src.count("self._scale_activations_to_request(")
    assert calls >= 1, "the scaler is defined but never called — the plan will not scale"


# ---------------------------------------------------------------------------
# The temporal axis, 2026-09-18. CogVideoX-2b on a 16 G card saves at 5, 6, 7
# and 8 frames and OOMs at 9, at `aten.convolution::90` asking 4.29 GiB. The
# container declares `temporal_compression_ratio: 4`, so 8 frames needs 2 latent
# frames and 9 is the FIRST count that needs 3 — the bisect lands exactly on that
# boundary. Prism planned the SAME 22 993 MB for both, because the video VAEs
# call their temporal axis `time` and the symbol map knew only `num_frames`.
# ---------------------------------------------------------------------------


def test_the_video_vaes_temporal_axis_is_followed(solver):
    """`time` is what CogVideoX-2b, mochi and Wan2.1 all declare, each with
    `source=input::z::dim_2`. Absent from the map, it was skipped in silence."""
    comp = _comp(("time", 9))
    got = scale(solver, comp, InputConfig(num_frames=33, temporal_compression=4))
    assert got == pytest.approx(1.0), "33 frames is (33-1)//4+1 = 9 latent, the trace value"


def test_a_temporal_axis_counts_LATENT_frames_by_the_engines_own_arithmetic(solver):
    """(n-1)//ratio + 1, not n/ratio. Plain division under-counts exactly where
    it matters: 9 frames at ratio 4 is 3 latent frames and not 2.25, and 9 is
    the first count needing a third."""
    comp = _comp(("time", 2))
    got = scale(solver, comp, InputConfig(num_frames=9, temporal_compression=4))
    assert got == pytest.approx(1.5), "3 latent against a trace of 2"
    assert got != pytest.approx(9 / 4 / 2), "plain division would read 1.125"


def test_the_step_between_eight_and_nine_frames_is_visible(solver):
    """The boundary the card dies on must appear in the estimate."""
    comp = _comp(("time", 2))
    eight = scale(solver, comp, InputConfig(num_frames=8, temporal_compression=4))
    nine = scale(solver, comp, InputConfig(num_frames=9, temporal_compression=4))
    assert eight == pytest.approx(1.0), "8 frames is 2 latent, the trace value"
    assert nine > eight, "9 frames needs a third latent frame and the plan must say so"


def test_a_symbol_no_request_field_answers_for_is_SAID_not_skipped(solver, caplog):
    """A dimension the estimate stops following in silence is how the plan read
    the same number at 8 and 9 frames while one ran and the other died."""
    import logging
    with caplog.at_level(logging.INFO, logger="neurobrix.core.prism.solver"):
        scale(solver, _comp(("a_dimension_nobody_maps", 7)), InputConfig(num_frames=9))
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "a_dimension_nobody_maps" in said
    assert "does not follow" in said
