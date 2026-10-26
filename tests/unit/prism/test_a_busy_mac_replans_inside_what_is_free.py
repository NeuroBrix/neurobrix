"""A busy Mac is re-planned inside what is free, or refused naming the shortfall — never a re-plan that places more than
the pool holds (the Mac's 98-cell table on 50a7993f, 2026-10-26 23:27). CogVideoX-5b-I2V triton with 13.5 GB free:
the re-plan took the one-guidance-branch-per-pass path, whose memory was priced WITHOUT the vendor's declared cuts —
its VAE whole, 69 705 MB — and the door refused "86 230 MB on this pool"; with the cuts it plans lazy_sequential within
the reading. What this test would do if the code were wrong: that refusal — RED. The other cells of the table: inside
the reading, or refused with the shortfall named (their floor at this reading is the price's: Wan2.1-T2V triton's
layer_streaming at its lowest rung holds 15.4 GB). No card."""
import pytest

from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def _need(p):
    fp = p.host_footprint or {}
    return (int(fp.get("total_bytes", 0)) - int(fp.get("resident_bytes", 0))) / 2**20


def test_the_guidance_split_re_plan_keeps_the_vendor_cuts(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 13500, "the Mac with ~4 GB taken by other applications")
    p, _ = _plan("CogVideoX-5b-I2V", "triton")
    assert not p.execution_door, p.execution_door
    assert _need(p) <= 13500, (p.strategy, int(_need(p)))
    assert p.component_memory["vae"].activation_bytes < 10 * 2**30, p.component_memory["vae"].activation_bytes


@pytest.mark.parametrize("model,mode", [("Wan2.1-I2V-14B-480P", "triton"), ("Wan2.1-T2V-1.3B", "triton"),
                                        ("mochi-1-preview", "compiled")])
def test_a_busy_mac_plans_inside_the_reading_or_names_the_shortfall(monkeypatch, model, mode):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 14500, "the Mac with ~3 GB taken by other applications")
    try:
        p, _ = _plan(model, mode)
    except RuntimeError as refusal:
        assert "free now" in str(refusal) and "does not fit" in str(refusal), str(refusal)[:300]
        return
    assert not p.execution_door, p.execution_door
    assert _need(p) <= 14500, (p.strategy, int(_need(p)))
