"""No row carries a number measured on a different model, or in a void campaign.

Both failures were real, in the first version of this generator, in the very
document whose stated rule is that no cell lies:

  * a prefix match attributed `Qwen3-Coder-30B-A3B-Instruct`'s cell to its
    `-int4g128-ffnonly` variant, which has never been run. Two rows, identical
    numbers, one of them fiction.
  * the glob read `2026_09_10_certified_reanchor`, a campaign its own author had
    marked `INVALIDATED.md` after it was found measuring a live tree. Its
    deepseek-moe record (8 keys, x2.22) displaced the valid one (9 keys, x2.39).

Both are the same defect: a number whose basis is not what the cell claims. The
generator now matches EXACTLY (with a small explicit alias table) and skips any
campaign carrying an invalidation marker.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_catalogue_report_never_borrows_a_measurement.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "certified_catalogue_report.py"

SNAPSHOT = """======================================================================
NeuroBrix Hub
======================================================================
MODEL                          CATEGORY         SIZE          LICENSE     DL  STATUS
------------------------------------------------------------------------------------------
Qwen/Qwen3-Coder-30B-A3B-Instruct CODE          57.1 GB       apache-2.0      1  installed
Qwen/Qwen3-Coder-30B-A3B-Instruct-int4g128-ffnonly CODE          17.2 GB       apache-2.0      2  installed
Total: 2 model(s) on registry
"""


def _campaign(root: Path, name: str, model: str, keys: int, gain: float,
              invalid: str | None = None) -> None:
    d = root / name / "proof" / model
    d.mkdir(parents=True)
    if invalid:
        (root / name / invalid).write_text("void")
    (d / "result.json").write_text(json.dumps({
        "model": model, "family": "code", "weight_gb": 1.0,
        "A": {"rc": 0, "exec_s": 100.0, "wall_s": 101.0, "certified_served": keys,
              "reps": [{"exec_s": 100.0, "rc": 0}]},
        "B": {"rc": 0, "exec_s": 100.0 * gain, "wall_s": 100.0 * gain, "swept": keys,
              "screen_excluded": 0, "contradictions": 0,
              "reps": [{"exec_s": 100.0 * gain, "rc": 0}]},
        "gate": {"identical": True},
    }))


def _run(out: Path, campaigns: Path):
    (out).mkdir(parents=True, exist_ok=True)
    (out / "hub_snapshot.txt").write_text(SNAPSHOT)
    r = subprocess.run([sys.executable, str(TOOL), "--out", str(out),
                        "--campaigns", str(campaigns)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return (out / "CATALOGUE.md").read_text()


def test_a_variant_does_not_inherit_the_base_models_measurement(tmp_path):
    camps = tmp_path / "camps"
    _campaign(camps, "good", "Qwen3-Coder-30B-A3B-Instruct", keys=8, gain=1.54)
    doc = _run(tmp_path / "out", camps)

    base = next(l for l in doc.splitlines() if "int4g128" not in l and "Qwen3-Coder" in l and l.startswith("|"))
    variant = next(l for l in doc.splitlines() if "int4g128" in l and l.startswith("|"))
    assert "x1.54" in base, "the model that WAS measured must show its measurement"
    assert "not measured" in variant and "x1.54" not in variant, (
        "the variant has never been run; a row that borrows its base model's "
        "numbers is exactly the lying cell this document forbids")
    assert "1 of 2 models have been measured" in doc


def test_a_campaign_its_author_voided_is_not_read(tmp_path):
    camps = tmp_path / "camps"
    _campaign(camps, "void_one", "Qwen3-Coder-30B-A3B-Instruct", keys=99, gain=9.99,
              invalid="INVALIDATED.md")
    doc = _run(tmp_path / "out", camps)
    assert "x9.99" not in doc and "99" not in doc.split("| # |")[1][:400], (
        "a record from an invalidated campaign reached the table")
    assert "declared them void" in doc and "void_one" in doc, (
        "the exclusion must be VISIBLE in the document — a silent skip is a "
        "different kind of lie")
    assert "0 of 2 models have been measured" in doc


def test_the_valid_record_wins_when_both_exist(tmp_path):
    """The control: exclusion must not also throw away the good record."""
    camps = tmp_path / "camps"
    _campaign(camps, "void_one", "Qwen3-Coder-30B-A3B-Instruct", keys=99, gain=9.99,
              invalid="INVALIDATED.md")
    _campaign(camps, "good", "Qwen3-Coder-30B-A3B-Instruct", keys=8, gain=1.54)
    doc = _run(tmp_path / "out", camps)
    assert "x1.54" in doc and "x9.99" not in doc


def test_a_count_field_wins_over_a_capped_sample(tmp_path):
    """`choices` carries SAMPLES capped at twenty beside full `*_count` fields.

    Reading `len(near_tie)` reported 20 near-ties where the run had found 139 —
    a cell that lies, in the document whose rule is that no cell lies. The
    sample is for the findings list; the count is for the number.
    """
    camps = tmp_path / "camps"
    d = camps / "c" / "proof" / "M"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "model": "Qwen3-Coder-30B-A3B-Instruct", "family": "code", "weight_gb": 1.0,
        "A": {"rc": 0, "exec_s": 100.0, "wall_s": 100.0, "certified_served": 203,
              "reps": [{"exec_s": 100.0, "rc": 0}]},
        "B": {"rc": 0, "exec_s": 200.0, "wall_s": 200.0, "swept": 203,
              "screen_excluded": 0, "contradictions": 0,
              "reps": [{"exec_s": 200.0, "rc": 0}]},
        "gate": {"identical": True},
        "choices": {
            "keys": 203, "certified": 195, "differ": 151,
            "near_tie": [{"key": f"k{i}"} for i in range(20)],   # capped sample
            "near_tie_count": 139,                                # the truth
            "contradicted": [{"key": "kc", "margin": 0.11,
                              "best_ms": 0.026, "delta_ms": 0.003}],
            "contradicted_count": 4,
            "excluded_picked": [], "excluded_picked_count": 0,
            "differ_uncertified": [], "differ_uncertified_count": 8,
        },
    }))
    doc = _run(tmp_path / "out", camps)
    row = next(l for l in doc.splitlines()
               if l.startswith("|") and "Qwen3-Coder-30B-A3B-Instruct`" in l)
    assert "| 139 |" in row, f"the capped sample was reported instead of the count: {row}"
    assert "| **4** |" in row
    assert "the record's sample, not" in doc, (
        "when the findings list is shorter than the count, the document must "
        "say so rather than let four read as one")
    assert "decomposition closes" in doc, (
        "139 + 4 + 0 + 8 = 151 = differ; the check must run and say so")


def test_a_cell_that_ran_and_crashed_is_not_the_same_as_one_nobody_tried(tmp_path):
    """Collapsing both into "not measured" erases the attempt AND the defect.

    Two Wan video cells ran on 2026-09-11 and crashed — one on a broadcast that
    cannot happen, one on a 6.4 GB allocation with 1.6 GB free. Each produced a
    named debt. A row that reads "not measured" for them says the opposite of
    what happened.
    """
    camps = tmp_path / "camps"
    d = camps / "c" / "proof" / "Wan2.1-VACE-1.3B-diffusers"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "model": "Wan2.1-VACE-1.3B-diffusers", "family": "video", "weight_gb": 18.2,
        "A": {"rc": 1, "swept": 1, "certified_served": 36, "reps": [{"rc": 1}]},
        "B": {"rc": 1, "swept": 37, "reps": [{"rc": 1}]},
        "gate": {"identical": False, "ran": False},
    }))
    out = tmp_path / "out"
    out.mkdir()
    (out / "hub_snapshot.txt").write_text(
        "MODEL                          CATEGORY         SIZE          LICENSE     DL  STATUS\n"
        "----------\n"
        "Wan-AI/Wan2.1-VACE-1.3B        VIDEO         18.2 GB       apache-2.0      3  installed\n"
        "Total: 1 model(s) on registry\n")
    r = subprocess.run([sys.executable, str(TOOL), "--out", str(out),
                        "--campaigns", str(camps)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    doc = (out / "CATALOGUE.md").read_text()
    row = next(l for l in doc.splitlines() if l.startswith("|") and "Wan2.1-VACE" in l)
    assert "**failed**" in row and "D-WAN-VACE-BROADCAST-AT-DIV" in row, (
        "a crashed cell must name its debt, not vanish into 'not measured'")
    assert "were attempted and FAILED" in doc
