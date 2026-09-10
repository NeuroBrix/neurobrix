"""`--skip-done` skips a MEASURED model, never a model that only failed.

The flag tested the existence of `result.json` and nothing else. A run that
crashed still writes that file — with `gate.ran == False` and both arms
`rc=1` — so a transient or fixable failure became permanent: every later
`--skip-done` container printed "done, skipped" and the model was never
measured again.

Observed 2026-09-10 in the certified-directory proof, where three models
carried such a record and were skipped by the container that could have
re-run them:

  Wan2.1-VACE-1.3B-diffusers  malloc of -4860000 bytes at aten.convolution::60
                              (a NEGATIVE size — a defect, not an OOM)
  Wan2.1-T2V-1.3B-Diffusers   OOM at aten.convolution::31 (11.6 GiB requested)
  Allegro-TI2V                TilingEngine temporal-extent contract, in 6 s

That is the same shape as the precondition rule the queue gained the same
day: `done_when` must mean "the verdict is known", and a file recording that
the work did not happen carries no verdict. A model that is genuinely
impossible is the other predicate's business (`impossible_when`), with its
reason written — not a crash silently promoted to a result.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_skip_done_needs_a_verdict.py
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


def _write(tmp_path: Path, name: str, payload) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (d / "result.json").write_text(json.dumps(payload))
    return d


def test_a_measured_arm_pair_counts_as_done(tmp_path):
    """gate.ran is True — a verdict exists, whatever it says."""
    m = _tool()
    d = _write(tmp_path, "measured",
               {"gate": {"kind": "bytes", "identical": True, "pass": True, "ran": True}})
    assert m.has_verdict(d) is True


def test_a_verdict_that_disagrees_still_counts_as_done(tmp_path):
    """DIFFERENT is a measurement, not a failure — it must not be re-run."""
    m = _tool()
    d = _write(tmp_path, "different",
               {"gate": {"kind": "bytes", "identical": False, "pass": False, "ran": True}})
    assert m.has_verdict(d) is True


def test_a_crashed_run_is_not_done(tmp_path):
    """The live shape: both arms rc=1, gate never ran."""
    m = _tool()
    d = _write(tmp_path, "crashed",
               {"gate": {"kind": "bytes", "identical": False, "pass": False, "ran": False},
                "A": {"rc": 1}, "B": {"rc": 1}})
    assert m.has_verdict(d) is False


def test_a_record_without_a_gate_is_not_done(tmp_path):
    m = _tool()
    assert m.has_verdict(_write(tmp_path, "nogate", {"model": "x"})) is False


def test_no_record_is_not_done(tmp_path):
    m = _tool()
    assert m.has_verdict(_write(tmp_path, "absent", None)) is False


def test_an_unreadable_record_is_not_done(tmp_path):
    """Present-but-broken is not a verdict either — it re-runs rather than
    silently counting as measured."""
    m = _tool()
    d = tmp_path / "broken"
    d.mkdir()
    (d / "result.json").write_text("{ not json")
    assert m.has_verdict(d) is False
