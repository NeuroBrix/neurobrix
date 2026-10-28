"""The Triton engine's temporal stream is priced by the same rule as the compiled one (decision C1 extended, the
supervisor 2026-10-28 11:20): a chunk holds on the device its price times the engine's measured ratio
(`memory.triton_stream_driver_over_price`, the M4 Pro: mochi-1-preview's Triton VAE 1-frame chunk, 9 334 MB of device for
a 5 701 MB price), against the device or the rung it is held to, less the engine's base. Where no chunk fits so, the VAE
is sliced at its group norms exactly as the native one is. Before: at an imposed 8 GB rung the Triton plan streamed the
VAE in 1-frame chunks and the M4 Pro held 9 334 MB (10 034 MB before S4b's flush); the native plan sliced and held it.
RED on the base: the Triton plan streams. No card."""
import pytest

from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def _mochi(monkeypatch, mode, rung=None):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17582, "the calm Mac")
    if rung:
        monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(rung))
    p, _ = _plan("mochi-1-preview", mode)
    return p


@pytest.mark.parametrize("mode", ["triton", "compiled"])
def test_at_an_8_gb_rung_both_engines_slice_the_vae(monkeypatch, mode):
    p = _mochi(monkeypatch, mode, 8192)
    assert (p.layer_stream_chunks or {}).get("vae"), (mode, p.component_stream, p.stream_declined)
    assert "vae" not in (p.component_stream or {}), p.component_stream


def test_on_the_calm_mac_the_triton_vae_still_streams_in_time(monkeypatch):
    p = _mochi(monkeypatch, "triton")
    assert (p.component_stream or {}).get("vae", {}).get("schedule"), (p.component_stream, p.stream_declined)


def test_the_triton_ratio_is_a_vendor_fact_with_its_source_and_null_where_unmeasured():
    from neurobrix.core.config.loader import vendor_config_for_device
    apple = vendor_config_for_device("apple", "apple_silicon", "0.0", model="Apple M4 Pro").get("memory") or {}
    row = (apple.get("triton_stream_driver_over_price") or {}).get("vae")
    assert row and row.get("ratio") and row.get("source"), row
    nvidia = vendor_config_for_device("nvidia", "volta", "7.0").get("memory") or {}
    assert "triton_stream_driver_over_price" in nvidia and nvidia["triton_stream_driver_over_price"] is None
