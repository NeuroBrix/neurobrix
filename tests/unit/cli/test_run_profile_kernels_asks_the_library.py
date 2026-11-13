"""`neurobrix run --profile-kernels [JSON]`: the engine only ASKS the library's kernel-time session for the request's
execution (nbxtensor.profile.ranked_call); without the flag the request runs exactly as before.

What this test does if the code is wrong: the off cell fails when a run without the flag opens a session or calls
anything but `executor.execute`; the on cell fails when the flag's path does not reach the library, or the request's
outputs do not come back; the parser cell fails when the flag is not `run`'s, or is given to calibrate or drift.
"""
from types import SimpleNamespace

import pytest

from neurobrix.cli import create_parser
from neurobrix.cli.commands import run as R


class _Exec:
    def __init__(self):
        self.calls = 0

    def execute(self, inputs):
        self.calls += 1
        return {"out": inputs}


def test_without_the_flag_the_request_runs_as_before(monkeypatch):
    import nbxtensor.profile as P
    monkeypatch.setattr(P, "ranked_call", lambda *a, **k: pytest.fail("a session opened without the flag"))
    ex = _Exec()
    assert R._execute(ex, 1, SimpleNamespace(profile_kernels=None)) == {"out": 1} and ex.calls == 1
    assert R._execute(ex, 2, SimpleNamespace()) == {"out": 2} and ex.calls == 2


@pytest.mark.parametrize("given, path", [("", None), ("t.json", "t.json")])
def test_with_the_flag_the_library_times_the_request(monkeypatch, given, path):
    import nbxtensor.profile as P
    seen = {}
    monkeypatch.setattr(P, "ranked_call", lambda fn, p=None: (seen.__setitem__("path", p), fn())[1])
    ex = _Exec()
    assert R._execute(ex, 3, SimpleNamespace(profile_kernels=given)) == {"out": 3}
    assert ex.calls == 1 and seen["path"] == path


def test_the_flag_is_run_s_alone():
    parser = create_parser(show_dev=True)
    assert parser.parse_args(["run", "--model", "m", "--profile-kernels"]).profile_kernels == ""
    assert parser.parse_args(["run", "--model", "m", "--profile-kernels", "x.json"]).profile_kernels == "x.json"
    assert parser.parse_args(["run", "--model", "m"]).profile_kernels is None
    for cmd in ("calibrate", "drift"):
        with pytest.raises(SystemExit):
            parser.parse_args([cmd, "--model", "m", "--profile-kernels"])
