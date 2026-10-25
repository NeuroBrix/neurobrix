"""The scope probe server forks a job per request and reads its answer from a pipe. A probed function that FORKS and
lets its child return (the serving daemon's `os.fork()`, reached by a constant probe on 2026-10-25) makes the job's
child run the same tail of the job and write the same answer: two JSON documents in the pipe, "Extra data" for one
`json.loads`, the whole scope of lot51 refused. Each answer now carries the pid that wrote it and the server takes the
forked job's by pid, naming the extra answer.

What this test would do if the server read the first document whatever its pid: the second case returns the child's
answer, not the job's — red. Injection seen red on 2026-10-25, restored green.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_scope_probe as P  # noqa: E402


def _doc(pid, out):
    return json.dumps({"pid": pid, "out": out}).encode()


def test_one_answer_is_the_job_s():
    assert P._answer_of(_doc(41, {"a": 1}), 41) == {"a": 1}
    assert "wrote nothing" in P._answer_of(b"", 41)["error"]
    assert "not an answer" in P._answer_of(b"{not json", 41)["error"]


def test_two_answers_the_job_s_is_taken_by_pid_and_the_fork_named():
    data = _doc(42, {"child": True}) + _doc(41, {"job": True})      # the child wrote first
    out = P._answer_of(data, 41)
    assert out["job"] is True and "child" not in out and "forked" in out["_forked"]
    missing = P._answer_of(_doc(42, {"child": True}), 41)
    assert "no answer from the job's process" in missing["error"]
