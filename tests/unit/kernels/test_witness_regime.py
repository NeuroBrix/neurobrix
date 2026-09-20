"""The stability regime is a lock OR a witness — one contract, two proofs.

A certified entry claims two things: the configuration is CORRECT (the fp64
oracle, clock-independent) and it is the FASTEST among the correct (the sweep's
timer). The timer is only comparable if the machine held still across the sweep.
On an NVIDIA rack that is a clock LOCK, read at entry. On Apple the clock is
OS-managed and cannot be locked, so stability is PROVEN by a WITNESS: a fixed
reference kernel timed at the start and end of each sweep, refused if it drifts.

These tests pin the witness half of the contract without a GPU: the regime
dispatch, the entry condition, and that `proof_records_regime` accepts either a
recorded lock or a recorded witness (an entry with neither is not served).
"""
from __future__ import annotations

import json

import pytest

from neurobrix.kernels import autotune_certify as AC
from neurobrix.kernels import autotune_certified as ACD


_METAL_PROTO = {
    "_backend": "metal",
    "regime": "witness",
    "witness": {"kernel": "matmul", "M": 512, "N": 512, "K": 512,
                "dtype": "fp16", "reps": 50, "drift_tolerance": 0.08},
}


@pytest.fixture(autouse=True)
def _fresh_regime(monkeypatch):
    monkeypatch.setattr(AC, "_REGIME", AC._UNREAD)
    yield
    monkeypatch.setattr(AC, "_REGIME", AC._UNREAD)


def test_the_witness_regime_does_not_refuse_on_clocks(monkeypatch, tmp_path):
    """The whole point: a witness machine is NOT refused for having no readable
    clock. It dispatches to the witness, whose condition is that the reference
    kernel RUNS — not that a clock can be read."""
    p = tmp_path / "rig_protocol.metal.json"
    p.write_text(json.dumps(_METAL_PROTO))
    monkeypatch.setattr(AC, "_protocol_file", lambda backend=None: p)
    monkeypatch.setattr(AC, "_witness_time_ms", lambda proto: 0.5)   # the kernel runs
    said = []
    AC.rig_protocol_refusal(say=said.append)                          # must NOT raise
    assert "WITNESS" in " ".join(said)


def test_a_witness_that_cannot_be_timed_is_refused(monkeypatch, tmp_path):
    """A certification whose stability cannot be measured is not a measurement —
    the same rule the clock lock enforces, by the witness's own means. If the
    reference kernel does not run, the witness has no baseline and refuses."""
    p = tmp_path / "rig_protocol.metal.json"
    p.write_text(json.dumps(_METAL_PROTO))
    monkeypatch.setattr(AC, "_protocol_file", lambda backend=None: p)

    def _boom(proto):
        raise RuntimeError("codegen refused the witness kernel")
    monkeypatch.setattr(AC, "_witness_time_ms", _boom)
    with pytest.raises(RuntimeError, match="could not be timed"):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_a_witness_regime_with_no_witness_spec_is_refused(monkeypatch, tmp_path):
    """A witness regime that carries no witness is a door with no hinge."""
    p = tmp_path / "rig_protocol.metal.json"
    p.write_text(json.dumps({"_backend": "metal", "regime": "witness"}))
    monkeypatch.setattr(AC, "_protocol_file", lambda backend=None: p)
    with pytest.raises(RuntimeError, match="no 'witness' spec"):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_an_unknown_regime_is_refused(monkeypatch, tmp_path):
    p = tmp_path / "rig_protocol.metal.json"
    p.write_text(json.dumps({"_backend": "metal", "regime": "vibes"}))
    monkeypatch.setattr(AC, "_protocol_file", lambda backend=None: p)
    with pytest.raises(RuntimeError, match="unknown regime"):
        AC.rig_protocol_refusal(say=lambda *a: None)


def test_a_proof_records_the_regime_by_lock_or_by_witness():
    """`proof_records_regime` (the coverage gate) accepts EITHER — a witness and
    a lock are two proofs of the same fact, that the sweep was comparable."""
    # A recorded clock (NVIDIA) counts.
    assert ACD.proof_records_regime({"machine": {"clocks_mhz": {"0": {"graphics": 1290}}}})
    # A recorded witness (Apple) counts.
    assert ACD.proof_records_regime({"stability_witness": {"open_ms": 0.5, "close_ms": 0.51,
                                                           "drift": 0.02, "tolerance": 0.08}})
    # Neither: not served (the 2026-09-07 regime-less proofs).
    assert not ACD.proof_records_regime({"machine": {}})
    assert not ACD.proof_records_regime({})
    # The old name still answers (back-compat alias).
    assert ACD.proof_records_clock is ACD.proof_records_regime


def test_the_drift_tolerance_is_read_from_the_profile_not_a_constant():
    """The tolerance lives in the protocol file, so stability is proven by what
    was measured against a declared bar, not a number baked into the engine."""
    proto = json.loads((__import__("pathlib").Path(__file__).resolve().parents[3]
                        / "tools" / "rig_protocol.metal.json").read_text())
    assert proto["regime"] == "witness"
    assert 0.0 < float(proto["witness"]["drift_tolerance"]) < 1.0
