"""A byte difference at one run per arm is not attributable to the change.

Twice in one day a `DIFFERENT` verdict accused a change of what the model does
on its own:

* `CogVideoX-2b` carries `nondeterministic: ["A", "B"]` — three repetitions,
  three shas, in BOTH arms — and the video comparison agrees at 43.6 dB;
* `Kokoro-82M` was printed DIFFERENT by a `--paired 1` tree gate against
  `7de1560`, a commit touching one file, `core/prism/solver.py`. Two runs of the
  SAME tree then gave two shas. The change could not have moved it and did not.

With one run per arm there is no repetition, so nothing in the record separates
the two cases. The verdict must say which question it answered — and the way to
turn it into an attribution costs exactly one more run.

Run: PYTHONPATH=src:tools python -m pytest tests/unit/tools/test_a_difference_is_not_yet_an_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from precision_zoo_campaign import verdict


def _tree_cell(paired, identical):
    return {
        "lever": "tree",
        "paired": paired,
        "gate": {"kind": "bytes", "ran": True, "identical": identical,
                 "arms": {"after": {"identical": identical}}},
    }


def test_one_run_per_arm_says_the_difference_is_unadjudicated():
    v = verdict(_tree_cell(paired=1, identical=False))
    assert "UNADJUDICATED" in v and "run one side twice" in v, (
        "a --paired 1 difference was reported as if it were attributable to the "
        "change; that is how a Prism-only commit got blamed for a TTS model's "
        "own run-to-run variation")


def test_repetitions_make_it_an_attribution_again():
    """The control: with repetitions the record CAN show self-difference, so the
    verdict must not be watered down for every run forever."""
    v = verdict(_tree_cell(paired=3, identical=False))
    assert "DIFFERENT" in v and "UNADJUDICATED" not in v


def test_an_identical_verdict_is_untouched():
    assert verdict(_tree_cell(paired=1, identical=True)).startswith("IDENTICAL")
