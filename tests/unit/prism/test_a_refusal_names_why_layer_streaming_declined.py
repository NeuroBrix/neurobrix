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


def _refusal(monkeypatch, rung, force):
    pin_host(monkeypatch, 24576, 11198, "the Mac's reading")
    impose_rung(monkeypatch, rung)
    if force:
        monkeypatch.setenv("NBX_FORCE_STRATEGY", "layer_streaming")
    else:
        monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    s = PrismSolver()
    with pytest.raises(RuntimeError) as err:
        s.solve_smart(NBXContainer.load(str(container_root("PixArt-XL-1024"))), profile(APPLE_M4_PRO),
                      InputConfig(batch_size=2, height=2048, width=1024), mode="triton")
    return str(err.value)


def test_a_forced_streaming_refusal_says_why_it_declined(monkeypatch):
    text = _refusal(monkeypatch, 4096, force=True)
    assert "layer_streaming declined:" in text, text[-600:]
    reason = text.split("layer_streaming declined:", 1)[1].splitlines()[0]
    assert "MB" in reason, f"the reason carries no figure: {reason!r}"


def test_the_cascade_refusal_names_every_strategy_it_tried_and_why_streaming_declined(monkeypatch):
    text = _refusal(monkeypatch, 4096, force=False)
    assert "ALL FAILED" in text, text[:400]
    assert "layer_streaming" in text.split("ALL FAILED")[0], (
        "the refusal's list of strategies tried does not name layer_streaming")
    assert "layer_streaming declined:" in text, text[-800:]


def test_the_kv_refusal_names_every_rejection_after_scoring(monkeypatch):
    """MiniCPM on the Mac: every candidate is rejected by the KV check (its LM and head alone are
    over the rung, register 104). That refusal said only 'No strategy can fit' — the rejections it
    had just recorded were printed by a different refusal that is never reached after scoring."""
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    s = PrismSolver()
    with pytest.raises(RuntimeError) as err:
        s.solve_smart(NBXContainer.load(str(container_root("MiniCPM-o-4_5"))), profile(APPLE_M4_PRO),
                      InputConfig(batch_size=1), mode="triton")
    text = str(err.value)
    assert "No strategy can fit model + KV cache" in text, text[:300]
    assert len(s._rejected) >= 2, ("precondition: several candidates were rejected", s._rejected)
    for name, _score, _why in s._rejected:
        assert f"{name} rejected:" in text, (name, text[:1200])
    assert s._layer_streaming_declined, "precondition: streaming declined MiniCPM on the Mac"
    assert f"layer_streaming declined: {s._layer_streaming_declined}" in text, text[:1200]


def test_a_reused_solver_does_not_carry_a_previous_solves_reason(monkeypatch):
    """Solve once so streaming declines; then force ANOTHER strategy on the same instance, which
    filters streaming out so it never runs: the second refusal must not print the first's reason."""
    pin_host(monkeypatch, 24576, 11198, "the Mac's reading")
    impose_rung(monkeypatch, 4096)
    c = NBXContainer.load(str(container_root("PixArt-XL-1024")))
    s = PrismSolver()
    monkeypatch.setenv("NBX_FORCE_STRATEGY", "layer_streaming")
    with pytest.raises(RuntimeError):
        s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=2, height=2048, width=1024),
                      mode="triton")
    assert s._layer_streaming_declined, "precondition: the first solve recorded a decline"
    monkeypatch.setenv("NBX_FORCE_STRATEGY", "single_gpu")
    with pytest.raises(RuntimeError) as err:
        s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=2, height=2048, width=1024),
                      mode="triton")
    assert "layer_streaming declined" not in str(err.value), str(err.value)[-400:]
