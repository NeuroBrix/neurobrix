"""The overrule is WIRED, not merely available.

`configs_agreeing_with_oracle` was written on 2026-09-10 with six passing tests
and could not be called from the screen: it takes ONE buffer, and a screened
result is a SNAPSHOT — a list of buffers with a dtype each. Nothing carried a
snapshot to it. Every test green, no seam, and the overrule the doctrine page
promised did not exist in the running engine.

So this file does not test the predicate again. It drives `screen_configs`
itself and pins the two failure modes consensus cannot see:

  * a MAJORITY WRONG IN THE SAME WAY — the vote seats it, the oracle must not,
    and the disagreement must be REPORTED as a finding rather than quietly
    corrected;
  * a UNANIMOUS WRONG SPACE — `len(clusters) == 1` returns the whole space and
    says nothing; with an oracle it must seat none and say why.

Plus the control that matters most for a guard installed on a hot path: with no
oracle provider, nothing changes at all.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_the_oracle_overrules_the_vote_in_the_screen.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import launcher as L


#: The screen refuses to invent a tolerance, so the fake profile carries the
#: real one's shape: a budget and a per-dtype rtol. The engine's own refusal
#: (`autotune_screen_rtol` absent -> RuntimeError) is what makes this necessary,
#: and it is the right refusal — a screen that guessed a tolerance could not
#: tell a reordered sum from a wrong answer.
_PROFILE = {
    "autotune_screen_max_bytes": 1 << 30,
    "autotune_screen_rtol": {"fp32": 1e-3, "fp16": 2e-2, "bf16": 5e-2},
}


def _f32(v):
    return np.asarray(v, dtype=np.float32).tobytes()


RIGHT = _f32([1.0, 2.0, 3.0, 4.0])
WRONG = _f32([1.0, 2.0, 0.0, 0.0])          # half the output unwritten
OTHER_WRONG = _f32([9.0, 9.0, 9.0, 9.0])


class _Config:
    def __init__(self, name):
        self.name = name

    def all_kwargs(self):
        return {}

    def __repr__(self):
        return self.name


class _Tuner:
    """The surface `screen_configs` actually touches: the bound arguments, the
    runnable, and a name for its own messages."""

    def __init__(self, state, produces):
        self.nargs = {"x": object()}
        self.arg_names = ["x"]
        self.fn = type("_Fn", (), {"run": staticmethod(lambda *a, **k: None)})()
        self.base_fn = lambda: None
        self.base_fn.__name__ = "fake_kernel"


@pytest.fixture
def screen(monkeypatch):
    """Drive the real decision path with the device memcpys stubbed out."""
    state = {"out": RIGHT}
    buffers = [(0, len(RIGHT), "float32")]

    monkeypatch.setattr(L, "_writable_buffers", lambda values: buffers)
    monkeypatch.setattr(L, "_snapshot", lambda b: [state["out"]])
    monkeypatch.setattr(L, "_restore", lambda b, shot: state.update(out=shot[0]))
    monkeypatch.setattr(L, "_record_screen_exclusions", lambda dropped: None)
    monkeypatch.setattr(L, "_SCREEN_CACHE", {})
    import neurobrix.kernels.ops._configs as C
    monkeypatch.setattr(C, "active_vendor_profile",
                        lambda: _PROFILE)

    def go(produces, oracle):
        state["out"] = RIGHT
        tuner = _Tuner(state, produces)
        L.set_screen_oracle((lambda t, k, b: oracle) if oracle else None)
        try:
            configs = [_Config(n) for n in produces]
            # The screen runs each candidate once, in order, from the same
            # restored starting state — so the fake kernel simply hands back
            # what that config is defined to produce.
            order = iter([c.name for c in configs])
            tuner.fn.run = lambda *a, **kw: state.update(out=produces[next(order)])
            return L.screen_configs(tuner, configs, key=(4,))
        finally:
            L.set_screen_oracle(None)
    return go


def test_a_majority_wrong_in_the_same_way_is_overruled(screen, capsys):
    produces = {"m0": WRONG, "m1": WRONG, "m2": WRONG, "ok0": RIGHT, "ok1": RIGHT}
    kept = screen(produces, [RIGHT])
    assert {c.name for c in kept} == {"ok0", "ok1"}, (
        "the vote's majority (3 wrong the same way) was seated over the oracle")
    out = capsys.readouterr().out
    assert "FINDING" in out and "wrong in the same way" in out, (
        "the overrule must SAY the vote was about to be wrong; silently "
        "correcting it hides the only evidence that this hardware needs looking at")


def test_a_unanimous_wrong_space_seats_nothing_and_says_why(screen):
    produces = {"a": WRONG, "b": WRONG, "c": WRONG}
    with pytest.raises(RuntimeError, match="contradicts EVERY candidate"):
        screen(produces, [RIGHT])


def test_a_healthy_space_is_untouched_by_the_overrule(screen, capsys):
    """The control: a guard that empties a correct space is worse than none."""
    produces = {"a": RIGHT, "b": RIGHT, "c": RIGHT}
    kept = screen(produces, [RIGHT])
    assert {c.name for c in kept} == {"a", "b", "c"}
    assert "FINDING" not in capsys.readouterr().out


def test_without_a_provider_the_consensus_decides_exactly_as_before(screen, capsys):
    """The hot path pays one `is None`. With no oracle the majority wins — the
    behaviour the engine has shipped since 2026-09-07, unchanged."""
    produces = {"m0": WRONG, "m1": WRONG, "m2": WRONG, "ok0": RIGHT, "ok1": RIGHT}
    kept = screen(produces, None)
    assert {c.name for c in kept} == {"m0", "m1", "m2"}, (
        "with no oracle the screen must still keep the largest cluster; this "
        "test is what proves the wiring did not change the default path")
    assert "AUTOTUNE_ORACLE" not in capsys.readouterr().out


def test_a_provider_that_raises_falls_back_loudly(screen, capsys, monkeypatch):
    produces = {"a": RIGHT, "b": RIGHT}
    state_tuner = _Tuner({"out": RIGHT}, produces)
    L.set_screen_oracle(lambda t, k, b: (_ for _ in ()).throw(ValueError("no oracle here")))
    try:
        monkeypatch.setattr(L, "_writable_buffers", lambda v: [(0, len(RIGHT), "float32")])
        monkeypatch.setattr(L, "_snapshot", lambda b: [RIGHT])
        monkeypatch.setattr(L, "_restore", lambda b, s: None)
        monkeypatch.setattr(L, "_SCREEN_CACHE", {})
        import neurobrix.kernels.ops._configs as C
        monkeypatch.setattr(C, "active_vendor_profile",
                            lambda: _PROFILE)
        cfgs = [_Config("a"), _Config("b")]
        state_tuner.fn.run = lambda *a, **kw: None
        L.screen_configs(state_tuner, cfgs, key=(7,))
    finally:
        L.set_screen_oracle(None)
    out = capsys.readouterr().out
    assert "oracle provider raised" in out, (
        "an oracle that fails must be announced, never swallowed into the "
        "consensus as if none had been asked for")
