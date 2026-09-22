"""A store that breathes is not a container that failed.

The reader in `tools/hub_cache_diff.py` already waits 5 + 15 + 45 + 120 + 120 s through a
refusal. The WRITER beside it waited not at all: one `subprocess.TimeoutExpired` abandoned a
21.9 GB upload at 15 %, and a single non-200 from the write probe DEFERRED the whole
container. Both read a temporary refusal as a failure.

The owner's decision (2026-09-22): the store is accepted as it is — nothing is flashed,
replaced or restarted. Its cause is known: the zvol behind 10.0.0.36 sits on four Micron 5200
drives with multi-second latency and takes its drive offline for 10 to 30 seconds at a time.
So the writes must be as patient as the reads: wait and retry through refusals, space the
writes, read back to verify, and never count a temporary refusal as a failure.

The read-back half already existed and is deliberately not re-implemented here: `publish` has
verified the hub's bytes member-by-member against the uploaded file, and hidden the object on
mismatch, since before this change. This file pins the three halves that did not exist.

Shapes: `AccessDenied` and `NoSuchBucket` are the permanent side — a retry would repeat them
forever and hide a real defect; `SlowDownWrite`, `RequestTimeout` and a lost write quorum are
the store breathing. The distinction is the whole point, so both sides are pinned.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "hub_cache_diff.py"


def _mod():
    spec = importlib.util.spec_from_file_location("hub_cache_diff_under_test", TOOL)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def hub(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m.time, "sleep", lambda *_a, **_k: None)   # the waits are real; the test's clock is not
    monkeypatch.setattr(m, "log", lambda *_a, **_k: None)
    return m


# ───────────────────────────── the refusal classifier ─────────────────────────────

@pytest.mark.parametrize("line", [
    "503 SlowDownWrite", "Retry-After: 60", "RequestTimeout", "write quorum lost",
    "drive is offline", "Connection reset by peer", "429 Too Many Requests",
    "subprocess.TimeoutExpired", "ServiceUnavailable", "EOF occurred in violation",
])
def test_the_store_breathing_is_read_as_TEMPORARY(hub, line):
    assert hub._TEMPORARY.search(line), line


@pytest.mark.parametrize("line", [
    "AccessDenied", "NoSuchBucket", "checksum mismatch", "InvalidArgument",
    "no such file or directory", "SignatureDoesNotMatch",
])
def test_a_real_defect_is_NOT_read_as_temporary(hub, line):
    assert not hub._TEMPORARY.search(line), line


# ───────────────────────────── the patient write ─────────────────────────────

def _spy_call(hub, monkeypatch, outcomes, logfile):
    """`outcomes`: per attempt, an rc or an exception to raise; writes the store's words to the log."""
    calls = {"n": 0}

    def fake(cmd, stdout=None, stderr=None, cwd=None, timeout=None):
        i = calls["n"]; calls["n"] += 1
        out = outcomes[min(i, len(outcomes) - 1)]
        if isinstance(out, BaseException):
            raise out
        rc, words = out
        if words and stdout is not None:
            stdout.write(words + "\n"); stdout.flush()
        return rc

    monkeypatch.setattr(hub.subprocess, "call", fake)
    return calls


def test_a_temporary_refusal_is_retried_and_the_publication_SUCCEEDS(hub, monkeypatch, tmp_path):
    log = tmp_path / "publish_x.log"
    calls = _spy_call(hub, monkeypatch, [(1, "503 SlowDownWrite"), (1, "503 SlowDownWrite"), (0, "")], log)
    assert hub._publish_patiently("x", ["py", "forge", "replace"], log) == 0
    assert calls["n"] == 3, "it did not retry through the refusals"


def test_a_timeout_is_a_refusal_not_an_abandonment(hub, monkeypatch, tmp_path):
    """The exact shape that abandoned a 21.9 GB upload at 15 %."""
    log = tmp_path / "publish_y.log"
    calls = _spy_call(hub, monkeypatch,
                      [subprocess.TimeoutExpired(cmd="forge", timeout=7200), (0, "")], log)
    assert hub._publish_patiently("y", ["py", "forge", "replace"], log) == 0
    assert calls["n"] == 2


def test_a_permanent_failure_is_reported_at_once_and_NOT_retried(hub, monkeypatch, tmp_path):
    log = tmp_path / "publish_z.log"
    calls = _spy_call(hub, monkeypatch, [(3, "AccessDenied: the key is not permitted")], log)
    assert hub._publish_patiently("z", ["py", "forge", "replace"], log) == 3
    assert calls["n"] == 1, "a permanent failure must not be retried — it would hide the defect"


def test_patience_is_BOUNDED(hub, monkeypatch, tmp_path):
    log = tmp_path / "publish_w.log"
    calls = _spy_call(hub, monkeypatch, [(1, "503 SlowDownWrite")], log)
    rc = hub._publish_patiently("w", ["py", "forge", "replace"], log)
    assert rc != 0
    assert calls["n"] == hub.PUBLISH_ATTEMPTS, "it must stop, and say the attempts are spent"


# ───────────────────────────── the patient probe ─────────────────────────────

def test_the_write_probe_retries_instead_of_deferring_the_container(hub):
    answers = iter([503, 503, 200])
    assert hub._probe_patiently(lambda: next(answers), "m") == 200


def test_a_probe_that_never_takes_a_write_still_gives_up(hub):
    assert hub._probe_patiently(lambda: 503, "m") == 503


def test_a_probe_that_RAISES_is_a_reading_not_a_crash(hub):
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise OSError("connection reset by peer")
        return 200

    assert hub._probe_patiently(flaky, "m") == 200


def test_the_constants_say_the_writer_is_at_least_as_patient_as_the_reader(hub):
    """The claim this whole file exists to make, as an executable comparison."""
    assert hub.PUBLISH_ATTEMPTS >= hub.RangeSource.ATTEMPTS
    assert hub.PUBLISH_MAX_BACKOFF_S >= hub.RangeSource.MAX_BACKOFF_S
    assert hub.PUBLISH_SPACING_S > 0
