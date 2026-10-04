"""A plan refusal names the component that BINDS, not one the same solve tiled.

SANA-Video_2B_720p_diffusers at 672x1344x81 under Triton on the 16 GB V100 class (default-ff6008b7,
the binned census container, 2026-10-04) was refused with two lines about its vae:

    largest component: vae at 303099MB
    vae: its weights and overhead alone are 16,449 MB against a 12,296 MB budget, and every band pays
    those in full — no number of bands reaches it

while the same solve had tiled that vae (tile extent 16, 5 200 MB of activations per tile), and with
"transformer: … it fits in 15 tiles" for a transformer the tiling engine had declined ("neither its
graph nor its profile states a spatial scale"). What bound was the transformer's own block: its
activations at 155 232 tokens peak at 18 774.5 MB (the fp32 linear attention's q, k, v and their ReLU
images, 1.3-2.7 GB each, alive at the rotary product) against a 15 072 MB segment. The same class as
Allegro at 720x1280 on that card: a DiT block's live set over the token axis, which no cut between ops
reaches and no tile exists for.

Under the Triton engine a host-placed component computes on the card, so on a discrete card the last
rung that can help is layer streaming, and its decline is the bound — not "the largest single
component in host RAM".

Injections that turn these red: drop `tiled=` / `untileable=` from the `what_would_have_fit` call in
`_fail_error`; make `_triton` False in `_fail_error`.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import ComponentMemory, PrismSolver

_MB = 1024 * 1024

STREAM_DECLINE = ("'transformer' cannot be cut into segments of 15072 MB: activations alone peak at "
                  "18774.5 MB, at or over the 15072.4 MB budget.")
NO_SCALE = "neither its graph nor its profile states a spatial scale"


class _Dev:
    def __init__(self, mb):
        self.capacity_mb = mb
        self.device_string = "cuda:0"


def _refusal(mode: str) -> str:
    """The SANA-Video solve's state at its refusal, its figures as measured."""
    comps = [
        ("vae", ComponentMemory(component_name="vae", weight_bytes=2016 * _MB,
                                activation_bytes=286650 * _MB, overhead_bytes=14433 * _MB)),
        ("transformer", ComponentMemory(component_name="transformer", weight_bytes=7848 * _MB,
                                        activation_bytes=34625 * _MB, overhead_bytes=2124 * _MB)),
        ("text_encoder", ComponentMemory(component_name="text_encoder", weight_bytes=9973 * _MB,
                                         activation_bytes=115 * _MB, overhead_bytes=504 * _MB)),
    ]
    s = PrismSolver.__new__(PrismSolver)
    s._mode = mode
    s._strategies_tried = ["layer_streaming", "cpu_streaming"]
    s._layer_streaming_declined = STREAM_DECLINE
    s._layer_stream_tilings = {"vae": {"tile_size": 16, "tiled_activation_bytes": 5200 * _MB}}
    s._tiling_declined = {"transformer": NO_SCALE}
    with pytest.raises(RuntimeError) as e:
        s._fail_error(comps, [_Dev(15565)])
    return str(e.value)


def test_a_component_the_solve_tiled_is_not_named_as_the_bound():
    text = _refusal("triton")
    assert "largest component: vae" not in text, text
    assert "largest component the plan did not tile: transformer" in text, text
    assert "vae: its weights and overhead alone" not in text, text
    assert "vae: the plan's tiling engine tiles it (tile extent 16, 5,200 MB of activations per tile)" in text, text


def test_a_component_the_tiling_engine_declined_is_offered_no_tile_count():
    text = _refusal("triton")
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("transformer: 34,625 MB"))
    assert "tiles of about" not in line, line
    assert f"no tile exists for it ({NO_SCALE})" in line, line
    assert text.count(NO_SCALE) == 1, "the reason is stated once"
    assert "each spatial dimension at" in line, "the input-side reduction is kept"


def test_a_reason_a_rung_line_already_prints_is_pointed_at_not_repeated():
    """The SANA refusal printed the transformer's decline on its rung lines; the advice points at it."""
    s = PrismSolver.__new__(PrismSolver)
    s._mode = "triton"
    s._strategies_tried = ["layer_streaming"]
    s._layer_streaming_declined = STREAM_DECLINE
    s._layer_stream_tilings = {}
    s._tiling_declined_by_rung = {"layer_streaming": {"transformer": NO_SCALE}}
    s._tiling_rung_figure = {"layer_streaming": ("cuda:0", 15073.0)}
    m = ComponentMemory(component_name="transformer", weight_bytes=7848 * _MB,
                        activation_bytes=34625 * _MB, overhead_bytes=2124 * _MB)
    with pytest.raises(RuntimeError) as e:
        s._fail_error([("transformer", m)], [_Dev(15565)])
    text = str(e.value)
    assert text.count(NO_SCALE) == 1, text
    assert "no tile exists for it (the tiling engine's reason is stated above)" in text, text


@pytest.mark.parametrize("mode", ["triton", "triton_sequential"])
def test_under_triton_on_a_discrete_card_the_bound_is_the_streaming_decline(mode):
    text = _refusal(mode)
    head = text.split("Components:")[0]
    assert "Under the Triton engine every rung computes on the card" in head, head
    assert STREAM_DECLINE in head.split("declined:\n", 2)[-1], head
    assert "needs only the largest single component to fit" not in head, head
    assert "More host RAM" not in text, text


def test_the_compiled_engine_keeps_the_host_ram_bound():
    """The compiled engine computes a host-placed component on the host: there the last rung IS the
    largest component in host RAM, and the refusal keeps saying so."""
    text = _refusal("compiled")
    assert "The last rung needs only the largest single component to fit" in text, text
    assert "More host RAM" in text, text
