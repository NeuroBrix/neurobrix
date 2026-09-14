"""The RNG guard's runner mirrors the suite's rule: a timeout the engine spent
autotuning is retried once warm; any other timeout fails with the sentence.
First form read two cold triton arms as failures (2026-09-14 02:03).

Injection: with the notice check removed, the third test went RED (a
timeout without the notice was retried instead of failing); restored, green.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.regression.test_a_container_with_an_rng_op_runs_twice_and_matches import _run_once, _TUNING_NOTICE  # noqa: E402


class _Runner:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes); self.calls = 0

    def __call__(self, cmd, **kw):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def _timeout(stderr=b""):
    return subprocess.TimeoutExpired(["x"], 900, output=b"", stderr=stderr)


def test_a_cold_timeout_is_retried_once_and_the_warm_result_returned():
    ok = subprocess.CompletedProcess(["x"], 0, stdout=b"", stderr=b"")
    r = _Runner([_timeout(_TUNING_NOTICE), ok])
    assert _run_once(["x"], 900, {}, ".", runner=r) is ok and r.calls == 2


def test_a_second_timeout_fails_as_warm():
    r = _Runner([_timeout(_TUNING_NOTICE), _timeout(_TUNING_NOTICE)])
    with pytest.raises(pytest.fail.Exception, match="WARM retry"):
        _run_once(["x"], 900, {}, ".", runner=r)


def test_a_timeout_without_the_notice_fails_at_once():
    r = _Runner([_timeout(b"nothing about tuning"), subprocess.CompletedProcess(["x"], 0)])
    with pytest.raises(pytest.fail.Exception, match="NOT autotuning"):
        _run_once(["x"], 900, {}, ".", runner=r)
    assert r.calls == 1
