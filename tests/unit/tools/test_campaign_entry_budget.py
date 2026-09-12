"""A cell whose known cost exceeds the campaign's budget is refused at entry.

On 2026-09-10 the certified-directory proof spent about fifteen hours of a quiet
rig on `Allegro`: arm A killed at 28 800 s, arm B killed at 24 958 s, both
`rc < 0`, verdict `ran: false`. The cost was not a surprise — the same cell had
already been killed at the same wall on an earlier flight, and the projection
from its own log was ~31 h per arm against an 8 h timeout. The next flight
excluded Allegro by hand, so the lesson was learnt, but only after the spend.

There is already a guard INSIDE the cell: an arm that produces no output ends
the container, because the byte gate needs every arm and no remaining arm can
change the verdict. What was missing is the guard at the DOOR — the one that
never lets the first arm start.

The rule: a cell whose estimated cost exceeds the campaign's budget is refused
before it consumes a second of card, loudly, with its number.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_entry_budget.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "precision_zoo_campaign.py"


def _tool():
    spec = importlib.util.spec_from_file_location("precision_zoo_campaign", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cell(tmp_path: Path, name: str, payload) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (d / "result.json").write_text(json.dumps(payload))
    return d


def test_a_cell_never_measured_has_no_estimate(tmp_path):
    """No record, no number — the guard must not invent one and must not
    refuse a cell it knows nothing about."""
    m = _tool()
    assert m.cell_cost_estimate(_cell(tmp_path, "unknown", None), timeout=28800) is None


def test_a_killed_arm_estimates_at_least_the_timeout_times_the_arms(tmp_path):
    """The live Allegro record: both arms killed at the wall."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {
        "A": {"rc": -9, "wall_s": 28800.238490581512},
        "B": {"rc": -15, "wall_s": 24958.0},
        "gate": {"ran": False}})
    est, basis = m.cell_cost_estimate(d, timeout=28800)
    assert est >= 2 * 28800, f"a killed arm bounds the cost from below, got {est}"
    assert "killed" in basis.lower() or "tue" in basis.lower()
    assert "28800" in basis


def test_a_measured_cell_estimates_from_its_arms(tmp_path):
    m = _tool()
    d = _cell(tmp_path, "Qwen", {
        "A": {"rc": 0, "wall_s": 100.0},
        "B": {"rc": 0, "wall_s": 140.0},
        "gate": {"ran": True}})
    est, basis = m.cell_cost_estimate(d, timeout=28800)
    assert 200 <= est <= 300, f"expected roughly the two arms, got {est}"
    assert "140" in basis or "100" in basis


def test_the_guard_refuses_a_cell_that_cannot_fit(tmp_path):
    """Allegro against an eight-hour campaign budget."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {
        "A": {"rc": -9, "wall_s": 28800.0},
        "B": {"rc": -15, "wall_s": 24958.0}})
    verdict = m.budget_refusal(d, budget_s=8 * 3600, timeout=28800)
    assert verdict is not None, "a 16 h cell must be refused by an 8 h budget"
    assert "8" in verdict or "28800" in verdict or "57600" in verdict, (
        f"the refusal must carry its number: {verdict}")


def test_the_guard_admits_a_cell_that_fits(tmp_path):
    m = _tool()
    d = _cell(tmp_path, "Qwen", {
        "A": {"rc": 0, "wall_s": 100.0},
        "B": {"rc": 0, "wall_s": 140.0}})
    assert m.budget_refusal(d, budget_s=8 * 3600, timeout=28800) is None


def test_no_budget_declared_refuses_nothing(tmp_path):
    """The guard is opt-in: a campaign that declares no budget keeps today's
    behaviour exactly."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {"A": {"rc": -9, "wall_s": 28800.0}})
    assert m.budget_refusal(d, budget_s=None, timeout=28800) is None


# --------------------------------------------------------------------------
# 2026-09-12 — the ledger and the kill timer are two numbers about one model
#
# The MEET phase of the catalogue runs each model ONCE ("ONE run per model, not
# a paired A/B", its own docstring) and was pricing those runs with the paired
# CELL default, the sum of both arms. So every model with a record was charged
# to the budget at about twice what it would spend.
#
# Worse, nothing compared that price to the `--timeout` the run would be given.
# On 2026-09-11 the plan accepted Wan2.1-T2V-1.3B at 6 962 s and the runner
# killed it at 2 700 s; Allegro, a recorded eight-hour model, went the same way.
# Two kills, ninety minutes of rig, and not one shape collected between them.
# --------------------------------------------------------------------------

CATALOGUE = Path(__file__).resolve().parents[3] / "tools" / "certify_the_catalogue.py"


def _catalogue():
    sys.path.insert(0, str(CATALOGUE.parent))
    spec = importlib.util.spec_from_file_location("certify_the_catalogue", CATALOGUE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# The live 2026-09-11 record, transcribed from
# nbx/campaigns/2026_09_10_certified_reanchor_frozen/proof/Wan2.1-T2V-1.3B-Diffusers.
# Both arms exited rc=1 — failures, not kills, which is why they were counted as
# measurements.
WAN_T2V_0911 = {"A": {"rc": 1, "wall_s": 3468.2213735580444},
                "B": {"rc": 1, "wall_s": 3494.222447872162},
                "gate": {"ran": False}}
MEET_TIMEOUT_0911 = 2700           # what the campaign actually gave each run


def test_one_run_is_not_priced_as_a_paired_cell(tmp_path):
    """arms=1 prices ONE run; the default prices the whole cell."""
    m = _tool()
    d = _cell(tmp_path, "Wan2.1-T2V-1.3B-Diffusers", WAN_T2V_0911)
    cell, _ = m.cell_cost_estimate(d, timeout=28800)
    one, _ = m.cell_cost_estimate(d, timeout=28800, arms=1)
    assert round(cell) == 6962, f"the cell is both arms, got {cell}"
    assert round(one) == 3494, f"one run is the widest arm, got {one}"
    # Not the mean and not the sum: a guard that UNDER-states a known cost is the
    # one that lets a doomed run start.
    assert one == max(WAN_T2V_0911[a]["wall_s"] for a in ("A", "B"))


def test_a_failed_arm_is_named_as_the_cost_of_failing(tmp_path):
    """rc>0 is real time really spent, but it is not the price of finishing.

    cell_cost_estimate already treats a KILLED arm (rc<0) as a lower bound. A
    FAILED arm was counted as a measurement, and the resulting number read as
    the cost of a successful run.
    """
    m = _tool()
    d = _cell(tmp_path, "Wan2.1-T2V-1.3B-Diffusers", WAN_T2V_0911)
    _, basis = m.cell_cost_estimate(d, timeout=28800, arms=1)
    assert "NON-ZERO" in basis and "not of finishing" in basis, basis


def test_a_clean_cell_says_nothing_about_failure(tmp_path):
    """The note appears only when it is true — otherwise it is noise."""
    m = _tool()
    d = _cell(tmp_path, "Qwen", {"A": {"rc": 0, "wall_s": 100.0},
                                 "B": {"rc": 0, "wall_s": 140.0},
                                 "gate": {"ran": True}})
    _, basis = m.cell_cost_estimate(d, timeout=28800, arms=1)
    assert "NON-ZERO" not in basis
    est, _ = m.cell_cost_estimate(d, timeout=28800, arms=1)
    assert est == 140.0


def test_a_killed_arm_bounds_one_run_at_the_timeout(tmp_path):
    """Allegro: killed arms bound a CELL at timeout x arms, one run at timeout."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {"A": {"rc": -9, "wall_s": 28800.2},
                                    "B": {"rc": -15, "wall_s": 24958.0},
                                    "gate": {"ran": False}})
    cell, _ = m.cell_cost_estimate(d, timeout=28800)
    one, basis = m.cell_cost_estimate(d, timeout=28800, arms=1)
    assert cell >= 2 * 28800
    assert one == 28800, f"one run is bounded by one timeout, got {one}"
    assert "lower bound" in basis


def test_a_run_is_refused_when_its_own_cost_exceeds_its_own_clock():
    """The 2026-09-11 pairing, both halves of it."""
    c = _catalogue()
    # Wan2.1-T2V-1.3B: 3494 s of recorded cost against 2700 s of clock.
    why = c.timeout_refusal(3494.2, MEET_TIMEOUT_0911, basis="2 measured arm(s)")
    assert why and "3494" in why and str(MEET_TIMEOUT_0911) in why
    # It names the clock that WOULD let it run, so the refusal is actionable.
    assert "--timeout 3495" in why
    # Allegro: an eight-hour model against the same clock.
    assert c.timeout_refusal(28800.0, MEET_TIMEOUT_0911)


def test_a_run_that_fits_its_clock_is_not_refused():
    c = _catalogue()
    assert c.timeout_refusal(2699.0, MEET_TIMEOUT_0911) is None
    assert c.timeout_refusal(2700.0, MEET_TIMEOUT_0911) is None, "equal fits"


def test_an_unmeasured_model_is_never_refused_on_a_guess():
    """The guard's job is to stop a KNOWN cost, never to invent one.

    mochi-1-preview had no recorded cell on 2026-09-11 and must still run — its
    defect is the OOM-reclaim loop, and refusing it here would hide that.
    """
    c = _catalogue()
    assert c.timeout_refusal(None, MEET_TIMEOUT_0911) is None
    assert c.timeout_refusal(0, MEET_TIMEOUT_0911) is None


# The live 2026-09-10 cell for a model that MET in 365 s the next day. Arm B ran
# with the certified directory off and swept everything; arm A did not.
QWEN3_VL_0910 = {"A": {"rc": 0, "wall_s": 211.0},
                 "B": {"rc": 0, "wall_s": 3135.0},
                 "gate": {"ran": True}}


def test_the_doom_test_reads_the_narrowest_arm(tmp_path):
    """Reserving and doom-testing are two questions with two statistics.

    Priced on the WIDEST, Qwen3-VL is a 3135 s model and a 2700 s clock refuses
    it — but it met in 365 s on 2026-09-11. A model that has already finished
    once under the clock is not doomed by it, whatever a slower arm did.
    """
    m, c = _tool(), _catalogue()
    d = _cell(tmp_path, "Qwen3-VL-30B-A3B-Thinking", QWEN3_VL_0910)
    widest, _ = m.cell_cost_estimate(d, timeout=28800, arms=1)
    floor, basis = m.cell_cost_estimate(d, timeout=28800, arms=1, narrowest=True)
    assert widest == 3135.0 and floor == 211.0
    assert "narrowest" in basis
    # The false refusal this split exists to prevent, and the true one it keeps.
    assert c.timeout_refusal(widest, MEET_TIMEOUT_0911), "the widest would refuse it"
    assert c.timeout_refusal(floor, MEET_TIMEOUT_0911) is None, "and that is wrong"


def test_wan_is_still_refused_on_its_floor(tmp_path):
    """The split must not cost the true refusal: both Wan arms exceeded the clock."""
    m, c = _tool(), _catalogue()
    d = _cell(tmp_path, "Wan2.1-T2V-1.3B-Diffusers", WAN_T2V_0911)
    floor, _ = m.cell_cost_estimate(d, timeout=28800, arms=1, narrowest=True)
    assert floor == 3468.2213735580444
    assert c.timeout_refusal(floor, MEET_TIMEOUT_0911), "every arm was over the clock"


def test_a_killed_arm_is_not_a_cost(tmp_path):
    """A kill never reached its own end, so it cannot answer "how long"."""
    m = _tool()
    d = _cell(tmp_path, "Half", {"A": {"rc": -9, "wall_s": 28800.0},
                                 "B": {"rc": 0, "wall_s": 140.0},
                                 "gate": {"ran": False}})
    # One arm ended. The answer comes from it alone, not from the kill's wall.
    for narrow in (False, True):
        est, basis = m.cell_cost_estimate(d, timeout=28800, arms=1, narrowest=narrow)
        assert est == 140.0, f"narrowest={narrow} got {est}"
        assert "1 arm(s) that ended" in basis
