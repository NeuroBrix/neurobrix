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


def test_an_idle_v100_16gb_plans_cogvideox_i2v_with_the_vendor_cuts(monkeypatch):
    """The same defect on an IDLE V100-16GB (gate ebm 16g, 2026-10-27 00:59: CogVideoX-5b-I2V triton rc1 in 29 s): the
    16g plan takes the one-guidance-branch-per-pass path, its VAE was priced whole (109 641 MB with the door reading what
    a lazy plan holds) and the execution door refused a machine with nothing else on it. RED on 4651eedd."""
    from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, profile
    from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config
    from neurobrix.core.prism import PrismSolver
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "an idle V100-16GB")
    c, ic = _input_config("CogVideoX-5b-I2V", "triton")
    p = PrismSolver().solve_smart(c, profile(V100_16GB), ic, mode="triton")
    assert not p.execution_door, p.execution_door
    assert p.component_memory["vae"].activation_bytes < 10 * 2**30


def test_the_door_reads_a_tiled_component_at_its_tile(monkeypatch):
    """mochi-1-preview native on an IDLE V100-16GB (gate ebm 16g, 2026-10-27 01:35: rc1 in 35 s): the planner accepted
    lazy_sequential with the VAE tiled, and the execution door read the VAE whole (21 214 MB) and refused a machine with
    nothing else on it. The door now reads each component as the plan holds it (`_peak_loaded_bytes`'s figures).
    RED on 4651eedd."""
    from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, profile
    from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config
    from neurobrix.core.prism import PrismSolver
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "an idle V100-16GB")
    c, ic = _input_config("mochi-1-preview", "compiled")
    p = PrismSolver().solve_smart(c, profile(V100_16GB), ic, mode="compiled")
    assert not p.execution_door, p.execution_door


def test_an_idle_v100_32gb_runs_mochi_triton_past_the_door(monkeypatch):
    """Gate ebm 32g, 2026-10-27 02:05: mochi-1-preview triton rc1 in 27 s on an idle V100-32GB, the door refusing
    '36 441 MB on this pool' — green in tha3/tha4 before the busy-machine lot. The same door reading as the 16g native
    cell (a component read whole instead of as the plan holds it). RED on 4651eedd."""
    import copy
    from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, profile
    from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config
    from neurobrix.core.prism import PrismSolver
    v32 = copy.deepcopy(V100_16GB); v32["id"] = "scenario-v100-32gb"
    v32["devices"][0].update(model="Tesla V100-SXM2-32GB", memory_mb=32768)
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 32501, 306, "an idle V100-32GB")
    c, ic = _input_config("mochi-1-preview", "triton")
    p = PrismSolver().solve_smart(c, profile(v32), ic, mode="triton")
    assert not p.execution_door, p.execution_door


def test_a_lazy_plan_fits_its_dearest_phase_on_a_dedicated_card(monkeypatch):
    """MiniCPM-o-4_5 triton on an idle V100-16GB (gate ebm 16g, 2026-10-27 01:14: OOM loading the head): lazy_sequential
    placed each component alone against the usable part, but the flow holds the language model and its head together
    while it decodes — 14 218 + 1 259 MB over a 14 321 MB part, a phase nothing compared to the card. The lazy plan now
    declines and zero3 (its plan before 83b1c3d9) takes it; proven on card 1 through the entry point (rc 0, the apple
    described). RED on 4651eedd."""
    from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, profile
    from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config
    from neurobrix.core.prism import PrismSolver
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "an idle V100-16GB")
    c, ic = _input_config("MiniCPM-o-4_5", "triton")
    p = PrismSolver().solve_smart(c, profile(V100_16GB), ic, mode="triton")
    assert p.strategy != "lazy_sequential", p.strategy
