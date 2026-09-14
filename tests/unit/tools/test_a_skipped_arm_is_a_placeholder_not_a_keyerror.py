"""When an earlier arm produced no output the later arms are skipped (the byte
gate needs every arm). The row builder then read `res["arms"]["after"]` and
raised KeyError (GLM-4.1V, 2026-09-14 09:54: its before arm could not load the
container on the older tree) — an unmeasurable pair reported as ERROR under a
family exit of 0. `arm_record` hands the row a placeholder that says why.

Injection: `arm_record` bypassed (direct indexing restored) made the first
test RED with KeyError; restored, green.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as Z  # noqa: E402


def test_a_skipped_arm_yields_a_placeholder_that_says_why():
    res = {"arms": {"before": {"rc": 1, "exec_s": None, "sha": None}},
           "arms_skipped": ["after"],
           "skip_reason": "arm before produced no output (rc=1); the byte gate needs every arm, so after could not change the verdict"}
    b = Z.arm_record(res, "after")
    assert b["exec_s"] is None and b["rc"] is None
    assert "produced no output" in b["skipped"]
    assert Z.arm_record(res, "before") is res["arms"]["before"]


def test_an_arm_that_ran_is_returned_as_is():
    res = {"arms": {"before": {"rc": 0, "exec_s": 3.0, "sha": "ab"}, "after": {"rc": 0, "exec_s": 2.0, "sha": "ab"}}}
    assert Z.arm_record(res, "after")["exec_s"] == 2.0
