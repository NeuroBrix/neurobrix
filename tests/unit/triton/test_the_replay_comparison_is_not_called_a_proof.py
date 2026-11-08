"""The replay's verify step compares the replayed outputs with the normal path's on one step; it is a divergence guard and
cannot see an error both passes share (on 2026-11-06 it said "plan VERIFIED byte-equal" 237 times on a wrong text). Its
line says what it observed and no more (the supervisor's decision 2026-11-06 21:30). No card."""
import inspect

from neurobrix.triton import replay


def test_the_matched_line_says_it_is_not_a_proof():
    assert replay.PLAN_MATCHED_LINE.format(launches=12) == (
        "[Replay] plan matches the normal path on this step (not a proof of the output) (12 launches) — replaying "
        "this bucket")


def test_the_verify_step_logs_that_line_and_never_verified():
    # the verify step says it through the engine's logger (the clean-terminal lot: a developer line, -v / NBX_LOG=debug)
    src = inspect.getsource(replay)
    assert "VERIFIED" not in src
    assert '_log.debug("%s", PLAN_MATCHED_LINE.format(launches=plan.launches))' in inspect.getsource(replay.maybe_run)
