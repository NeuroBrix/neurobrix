"""A refusal names why `layer_streaming` declined, with its numbers.

`_try_layer_streaming` returned None at eight places and said nothing, and the refusal text listed
the strategies from a hand-kept list that never named it. 62 of the Mac's 242 refusals (4a3658d7)
had weights over the rung and still could not say why streaming did not serve them — Ming,
65 271 MB of weights and 16 MB of activations, refused at every rung up to 16 384. On this rack the
same silence hid a forced plan's refusal behind "cannot fit the model" (PixArt-XL-1024, 2048x1024,
rung 8 192, batch 2: the whole components beside the streamed encoder filled the usable budget).

Constructed (register 102): the Mac's profile, its reading, the rung imposed, the strategy forced.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, container_root, impose_rung, pin_host,
                                              profile)


#: The plan these cells refuse. It was batch 2, then batch 8, at 2048x1024 on PixArt (the Mac's own
#: 14 420 MB refusal, 2026-05-20, then the re-traced container). Since a component over the rung
#: streams inside itself (2026-10-04) PixArt plans layer_streaming at 2048x1024 at ANY batch and
#: any imposed rung down to 1 024 — its transformer, one piece, is cut in two. What still refuses is
#: a component whose ACTIVATIONS alone fill the usable rung and that no tile serves:
#: Wan2.1-I2V-14B-480P's `vae_encoder` at its derived request (17 738 MB of activations against
#: 15 073 usable at the 16 384 rung; its traced time axis is frozen, so no tile maps it — a Forge
#: re-trace, test_a_component_over_the_rung_streams_inside_itself). When that encoder is re-traced
#: these cells need another refusal, and say so by failing.
REFUSED_MODEL = "Wan2.1-I2V-14B-480P-Diffusers"
REFUSED_RUNG = 16384


def _refused_request():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import trace_request as TR
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    c = NBXContainer.load(str(container_root(REFUSED_MODEL)))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", REFUSED_MODEL,
                                       *TR.derived_request(REFUSED_MODEL), "--triton"])
    return c, request_input_config(args, man, man.get("family"), c.cache_path)


def _refusal(monkeypatch, rung, force):
    pin_host(monkeypatch, 24576, 18186, "the Mac's idle reading")
    impose_rung(monkeypatch, rung)
    if force:
        monkeypatch.setenv("NBX_FORCE_STRATEGY", "layer_streaming")
    else:
        monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    s = PrismSolver()
    c, ic = _refused_request()
    with pytest.raises(RuntimeError) as err:
        s.solve_smart(c, profile(APPLE_M4_PRO), ic, mode="triton")
    return str(err.value)


def test_the_retraced_container_plans_what_the_old_one_refused(monkeypatch):
    # A strict xfail until release-candidate-1 (2026-09-28): main's 3-segment plan sized its
    # activations at the trace; with the partition sized at the request (38751c12) it plans.
    """The Mac's 2048x1024 refusal (14 420 MB against the rung) was the OLD container's: its
    transformer declared `seq_len` at two symbol ids, and the reserve followed. The re-traced
    container plans batch 2 at 2048x1024 under layer_streaming at the 4096 MB rung — a claim
    the Mac can now check on its own card."""
    pin_host(monkeypatch, 24576, 11198, "the Mac's reading")
    impose_rung(monkeypatch, 4096)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    plan = PrismSolver().solve_smart(NBXContainer.load(str(container_root("PixArt-XL-2-1024-MS"))),
                                     profile(APPLE_M4_PRO), InputConfig(batch_size=2, height=2048, width=1024),
                                     mode="triton")
    assert "layer_streaming" in str(getattr(plan, "strategy", plan)), plan


def test_a_forced_streaming_refusal_says_why_it_declined(monkeypatch):
    text = _refusal(monkeypatch, REFUSED_RUNG, force=True)
    assert "layer_streaming declined:" in text, text[-600:]
    reason = text.split("layer_streaming declined:", 1)[1].splitlines()[0]
    assert "MB" in reason, f"the reason carries no figure: {reason!r}"


def test_the_cascade_refusal_names_every_strategy_it_tried_and_why_streaming_declined(monkeypatch):
    text = _refusal(monkeypatch, REFUSED_RUNG, force=False)
    assert "ALL FAILED" in text, text[:400]
    assert "layer_streaming" in text.split("ALL FAILED")[0], (
        "the refusal's list of strategies tried does not name layer_streaming")
    assert "layer_streaming declined:" in text, text[-800:]


def test_minicpm_on_the_idle_mac_is_streamed_not_refused(monkeypatch):
    """Every whole-component candidate is rejected by the KV check (the LM and head alone are over
    the rung, register 104), and every component fits the rung ALONE: the streaming rung used to be
    offered nothing ("the streamed []") and the solve refused. The owner's rule — the engine never
    refuses — holds since a-streamed-plan-states-its-window: the largest component is streamed."""
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    plan = PrismSolver().solve_smart(NBXContainer.load(str(container_root("MiniCPM-o-4_5"))), profile(APPLE_M4_PRO),
                                     InputConfig(batch_size=1), mode="triton")
    assert plan.strategy == "layer_streaming", plan.strategy
    assert plan.device_window_mb is not None and plan.device_window_mb < plan.total_memory_mb


def test_the_kv_refusal_names_every_rejection_after_scoring(monkeypatch):
    """The same machine, the streaming rung made to decline (a model it cannot cut): the refusal that
    follows the KV check must name every rejection it recorded and the decline — it once said only 'No
    strategy can fit', its rejections printed by a different refusal never reached after scoring."""
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)

    def declines(self, *a, **k):
        self._layer_streaming_declined = "stand-in: no cut of this model fits the rung"
        return None
    monkeypatch.setattr(PrismSolver, "_try_layer_streaming", declines)
    s = PrismSolver()
    with pytest.raises(RuntimeError) as err:
        s.solve_smart(NBXContainer.load(str(container_root("MiniCPM-o-4_5"))), profile(APPLE_M4_PRO),
                      InputConfig(batch_size=1), mode="triton")
    text = str(err.value)
    assert "No strategy can fit model + KV cache" in text, text[:300]
    assert len(s._rejected) >= 2, ("precondition: several candidates were rejected", s._rejected)
    for name, _score, _why in s._rejected:
        assert f"{name} rejected:" in text, (name, text[:1200])
    assert f"layer_streaming declined: {s._layer_streaming_declined}" in text, text[:1200]


def test_a_reused_solver_does_not_carry_a_previous_solves_reason(monkeypatch):
    """Solve once so streaming declines; then force ANOTHER strategy on the same instance, which
    filters streaming out so it never runs: the second refusal must not print the first's reason."""
    pin_host(monkeypatch, 24576, 18186, "the Mac's idle reading")
    impose_rung(monkeypatch, REFUSED_RUNG)
    c, ic = _refused_request()
    s = PrismSolver()
    monkeypatch.setenv("NBX_FORCE_STRATEGY", "layer_streaming")
    with pytest.raises(RuntimeError):
        s.solve_smart(c, profile(APPLE_M4_PRO), ic, mode="triton")
    assert s._layer_streaming_declined, "precondition: the first solve recorded a decline"
    monkeypatch.setenv("NBX_FORCE_STRATEGY", "single_gpu")
    with pytest.raises(RuntimeError) as err:
        s.solve_smart(c, profile(APPLE_M4_PRO), ic, mode="triton")
    assert "layer_streaming declined" not in str(err.value), str(err.value)[-400:]
