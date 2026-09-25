"""No component falls between "held whole" and "streamed": one figure decides both.

`_place_component` held a component whole when `weight x cost + activation <= 0.92 x rung`.
`_try_layer_streaming` streamed a component when `total_bytes > rung`. Between the two figures
a component was too large to hold whole and too small to stream. Measured on main (69c98647),
PixArt-XL-2-1024-MS at 2048x1024 on the Mac's profile, the text_encoder costing 9 171.7 MB whole
and 9 630.3 MB in total:

    rung  9 700 MB  ->  cpu_streaming       (off the accelerator)
    rung  9 900 MB  ->  cpu_streaming
    rung 10 200 MB  ->  lazy_sequential

so every rung in [9 630, 9 969) MB sent a component the device holds to the host. Same class as
the Mac's refusal the streaming landing closed: two rungs reading two figures.

Now `PrismSolver._usable_mb` is the one figure: `_place_component` holds whole against it,
`_try_layer_streaming` streams exactly what it refuses and cuts segments against it.

THE SCENARIO IS CONSTRUCTED, NOT FOUND
--------------------------------------
The machine is pinned (the Mac's host reading injected, the rung imposed through the door). The
band is not hand-picked: it is computed from the solver's own estimate of the component and its
own fraction, so a retrace or a re-measured fraction moves the band and the cells follow it. A
precondition cell proves the band is non-empty before anything is judged in it.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, impose_rung, pin_host, profile

APPLE = APPLE_M4_PRO            # the machine is built here, never read off this one (register 102)
MODEL, COMP, H, W = "PixArt-XL-2-1024-MS", "text_encoder", 1024, 2048
HOST_FREE_MB = 11198            # the Mac's own reading (e904da83): capacity 10 638 >= every rung below
MODES = ["compiled", "triton"]
MB = 1024 * 1024


def _pin(monkeypatch, rung_mb):
    pin_host(monkeypatch, 24576, HOST_FREE_MB, "the Mac's reading")
    impose_rung(monkeypatch, rung_mb)


def _solve(rung_mb, monkeypatch, mode="compiled"):
    root = container_root(MODEL)
    _pin(monkeypatch, rung_mb)
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    c = NBXContainer.load(str(root))
    try:
        p = s.solve_smart(c, profile(APPLE), InputConfig(batch_size=1, height=H, width=W), mode=mode)
    except RuntimeError as e:          # a refusal is a verdict here, not an error of the cell
        p = e
    return p, s, seen, c


@pytest.fixture(scope="module")
def band():
    """[lo, hi): the rungs at which the component is too big for `total > rung` streaming and
    too big to place whole — derived from the solver's own figures, at a neutral rung."""
    mp = pytest.MonkeyPatch()
    try:
        _, s, seen, c = _solve(16384, mp)
        dev = s._prepare_devices(profile(APPLE))[0]
        whole = s._whole_component_mb(c, COMP, seen[COMP], dev)
        total = seen[COMP].total_mb
        return total, whole / s.whole_component_fraction
    finally:
        mp.undo()


def test_the_band_exists_for_this_component(band):
    lo, hi = band
    assert hi - lo > 50, (f"the band [{lo:.1f}, {hi:.1f}) MB is too thin to put three rungs in; "
                          f"the component or the fraction moved — re-derive the scenario")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("where", [0.1, 0.5, 0.9])
def test_a_rung_inside_the_band_streams_on_the_card(monkeypatch, band, where, mode):
    lo, hi = band
    rung = int(lo + where * (hi - lo)) + 1
    p, *_ = _solve(rung, monkeypatch, mode)
    strategy = getattr(p, "strategy", repr(p))
    assert strategy == "layer_streaming", (
        f"rung {rung} MB (band [{lo:.1f}, {hi:.1f})): {strategy!r}. A component too big to hold "
        f"whole on the accelerator is streamed on it, never sent to the host")


@pytest.mark.parametrize("mode", MODES)
def test_and_the_streamed_peak_stays_inside_the_usable_part_of_the_rung(monkeypatch, band, mode):
    lo, hi = band
    rung = int((lo + hi) / 2)
    p, s, seen, _ = _solve(rung, monkeypatch, mode)
    parts = getattr(s, "_layer_stream_partitions", None) or {}
    assert COMP in parts, f"{COMP} was not streamed at rung {rung}"
    dev = s._prepare_devices(profile(APPLE))[0]
    beside = sum(m.total_bytes for n, m in seen.items() if n not in parts)
    peak = (parts[COMP].peak_resident_bytes + beside) / MB
    assert peak <= s._usable_mb(dev), (
        f"streamed peak {peak:.1f} MB over the usable {s._usable_mb(dev):.1f} MB of the "
        f"{rung} MB rung: a streamed component is held to a looser standard than a whole one")


def test_above_the_band_the_component_is_held_whole(monkeypatch, band):
    """The control: past the band nothing is streamed that fits whole."""
    _, hi = band
    p, *_ = _solve(int(hi) + 64, monkeypatch)
    assert getattr(p, "strategy", repr(p)) == "lazy_sequential", p


def test_a_component_that_fits_whole_is_never_the_one_streamed(monkeypatch, band):
    """The other half of the law. Just above the band the text_encoder fits whole (its whole
    cost <= usable) while its TOTAL, which counts the estimator's overhead, does not. Streaming
    must classify by the same whole cost `_place_component` uses: classified by the total it
    would stream a component the rung above already holds whole."""
    _, hi = band
    _, s, seen, _ = _solve(int(hi) + 64, monkeypatch)
    dev = s._prepare_devices(profile(APPLE))[0]
    assert seen[COMP].total_mb > s._usable_mb(dev), (
        "precondition: at this rung the component's total must exceed the usable figure, or the "
        "cell cannot tell the whole cost from the total")
    parts = getattr(s, "_layer_stream_partitions", None) or {}
    assert COMP not in parts, f"{COMP} fits whole at rung {int(hi) + 64} MB and was streamed"
