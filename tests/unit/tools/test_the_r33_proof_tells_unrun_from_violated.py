"""The R33 execution proof must say "torch was here", "this did not run" and
"my detector is dead" as three different things.

It did not, and the cost was a standing false alarm: on 2026-09-17 the table
printed `*** R33 VIOLATION ***` on BOTH stacks — the candidate one and the old
one, one variable apart — while torch appeared in NO owned step on either. The
verdict was computed as `all(not torch and not error)`, so the two Metal rows,
which error on every CUDA box because the box has no Metal device, read as
sightings of torch. Every CUDA run of this tool since the Metal rows landed had
been red for a reason that had nothing to do with R33.

The second defect was the reverse: its negative control asserted that Triton's
C++ binder imports torch. Upstream made the CUDA driver probe native in 3.7
(triton#9578, #10935), the control read False on 3.8, and the table kept
printing verdicts with no ability to detect torch at all.

So the fix cannot simply be "an error is benign": that would make a step which
quietly stops being measured read exactly like a step that passed — the
`instrumentation that lies by construction` family, where silence and success
are indistinguishable. A case declares the platform it needs; an unrun case is
excused ONLY there, and is a broken harness anywhere else.
"""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "r33_execution_proof.py"
_spec = importlib.util.spec_from_file_location("r33_execution_proof", TOOL)
r33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r33)

OK = ("a NeuroBrix step", False, "", None)
METAL = ("COLD compile to MSL", False, "CASE DID NOT RUN: no Metal device", "darwin")
OBSERVATION = ("Triton's own binder", False, "", None)
CONTROL = ("import torch on purpose", True, "", None)


def report(*owned, control=CONTROL, platform="linux"):
    return r33.build_report([*owned, OBSERVATION, control], platform)


def test_a_clean_cuda_box_reads_torch_free_although_the_metal_rows_cannot_run():
    # The exact shape of every CUDA run: two Metal steps that cannot run here.
    # This is the reading that was printed as a VIOLATION for weeks.
    text, code = report(OK, METAL, OK, METAL)
    assert "TORCH-FREE" in text
    assert "VIOLATION" not in text
    assert code == 0, "a step that cannot run on this machine is not a violation"
    assert "not applicable on linux" in text, "and it is still SAID, never hidden"


def test_torch_in_an_owned_step_is_a_violation_and_names_the_step():
    text, code = report(OK, ("the launcher, cold", True, "", None))
    assert "*** R33 VIOLATION ***" in text
    assert "the launcher, cold" in text.split("Every step")[1]
    assert code == 1


def test_a_step_that_should_run_here_and_did_not_is_a_broken_harness():
    # NOT a pass. This is the door: a step that silently stops being measured
    # reads as success under any rule that merely excuses errors.
    text, code = report(OK, ("the launcher, cold", False, "CASE DID NOT RUN: boom", None))
    assert "*** BROKEN HARNESS ***" in text
    assert "the launcher, cold" in text
    assert code == 1


def test_the_platform_excuse_holds_only_on_the_platform_that_is_not_ours():
    # The same unrun Metal row, read on a Mac, is a broken harness there.
    text, code = report(OK, METAL, platform="darwin")
    assert "*** BROKEN HARNESS ***" in text
    assert code == 1


def test_a_silent_detector_control_makes_the_table_unproven_not_green():
    silent = ("import torch on purpose", False, "", None)
    text, code = report(OK, OK, control=silent)
    assert "UNPROVEN" in text
    assert "TORCH-FREE" not in text
    assert code == 1
    assert "NO — this table proves nothing" in text


def test_a_torch_sighting_outranks_an_unrun_step():
    text, code = report(("the launcher, cold", True, "", None),
                        ("something else", False, "CASE DID NOT RUN: boom", None))
    assert "*** R33 VIOLATION ***" in text, "a violation is never masked by a broken row"
    assert code == 1


def test_the_control_and_the_observation_are_not_counted_as_owned_steps():
    # Torch in the DETECTOR CONTROL is the control working, not a violation.
    text, code = report(OK)
    assert code == 0 and "TORCH-FREE" in text


@pytest.mark.parametrize("label", ["COLD compile a kernel to MSL (our driver)",
                                   "the FULL launcher contract checker, cold"])
def test_the_metal_cases_in_the_real_table_declare_the_platform_they_need(label):
    # Without this the tool is green here for the wrong reason.
    case = next(c for c in r33.CASES if c[0] == label)
    assert len(case) == 3 and case[2] == "darwin", f"{label} must declare its platform"


def test_the_last_case_is_a_control_that_cannot_go_inert():
    # The previous control depended on an upstream implementation detail and
    # died silently when upstream changed it. This one imports torch itself.
    label, body = r33.CASES[-1][0], r33.CASES[-1][1]
    assert "CONTROL" in label
    assert "import torch" in body
    assert "triton" not in body.lower(), "the control must not depend on upstream"
