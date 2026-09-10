"""A median is published with its population, or a comparison is refused.

An earlier campaign reported x13.9 over a population that carried `chatterbox`
(674 keys) and `openaudio` (693) — two of the highest key counts in the
catalogue. The campaign that followed does not run them. Writing "the median
fell" against that number would be exactly as false as the x13.9 was: what moved
is the population, and a statement about which models were run would be made in
the grammar of a statement about the engine.

So the comparison is a DOOR, not a footnote — a cross-population median cannot be
made honest by a caveat beside it.

Also pinned: the ordering. The gain is a function of the KEY COUNT, not of model
size — 54 GB at 203 keys gains x11.69 while 61 GB at 8 keys gains x1.54 — and a
table sorted by name invites a reader to invent the other law.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_table_refuses_a_moved_population.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "campaign_table.py"


def _cell(campaign: Path, model: str, keys: int, a: float, b: float) -> None:
    d = campaign / "proof" / model
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "model": model, "family": "llm", "weight_gb": 1.0,
        "A": {"rc": 0, "exec_s": a, "certified_served": keys,
              "reps": [{"exec_s": a, "rc": 0}]},
        "B": {"rc": 0, "exec_s": b, "swept": keys, "screen_excluded": 0,
              "contradictions": 0, "reps": [{"exec_s": b, "rc": 0}]},
        "gate": {"identical": True},
    }))


def _run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


def test_the_table_is_ordered_by_keys_not_by_name(tmp_path):
    c = tmp_path / "camp"
    _cell(c, "aaa_few_keys", 8, 100.0, 154.0)
    _cell(c, "zzz_many_keys", 203, 100.0, 1170.0)
    out = _run(c).stdout
    assert out.index("zzz_many_keys") < out.index("aaa_few_keys"), (
        "the many-key model must lead: the key count is the law this table "
        "exists to show")


def test_the_median_is_printed_with_every_model_of_its_population(tmp_path):
    c = tmp_path / "camp"
    _cell(c, "alpha", 8, 100.0, 154.0)
    _cell(c, "beta", 203, 100.0, 1170.0)
    out = _run(c).stdout
    assert "Median gain:" in out
    for name in ("alpha (8 keys)", "beta (203 keys)"):
        assert name in out, (
            f"{name} missing from the population listing — a median detached "
            f"from its population is the defect this tool was written after")


def test_a_comparison_across_different_populations_is_refused(tmp_path):
    """Seen failing is the point: this must refuse, and name the difference."""
    now, before = tmp_path / "now", tmp_path / "before"
    _cell(now, "shared", 8, 100.0, 154.0)
    _cell(before, "shared", 8, 100.0, 154.0)
    _cell(before, "chatterbox", 674, 100.0, 1390.0)   # the model that moved

    r = _run(now, "--compare", before)
    assert r.returncode == 2, "a moved population must be refused, not annotated"
    assert "COMPARISON REFUSED" in r.stderr
    assert "chatterbox" in r.stderr, (
        "the refusal must NAME what differs, or the reader cannot act on it")


def test_an_identical_population_compares(tmp_path):
    """The control. A door that never opens is a door nobody can use."""
    now, before = tmp_path / "now", tmp_path / "before"
    for c, b in ((now, 154.0), (before, 200.0)):
        _cell(c, "shared", 8, 100.0, b)
    r = _run(now, "--compare", before)
    assert r.returncode == 0, r.stderr
    assert "Same population." in r.stdout


def test_a_lever_that_measured_nothing_gets_no_ratio(tmp_path):
    """1.0 reads as 'no gain'; the truth is 'no measurement'."""
    c = tmp_path / "camp"
    d = c / "proof" / "empty"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "model": "empty", "family": "llm", "weight_gb": 1.0,
        "A": {"rc": 0, "exec_s": 100.0, "certified_served": 0,
              "reps": [{"exec_s": 100.0, "rc": 0}]},
        "B": {"rc": 0, "exec_s": 103.0, "swept": 0, "reps": [{"exec_s": 103.0, "rc": 0}]},
        "gate": {"identical": True},
    }))
    out = _run(c).stdout
    assert "no ratio" in out and "no measurement" in out
