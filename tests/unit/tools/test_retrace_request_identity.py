"""An output belongs to the request that produced it.

The campaign bounds a denoiser's step count (2026-09-08), so a video row's request changed under
outputs that were already in its directory: Wan2.1-T2V's sequential arm had rendered the vendor's
100 steps while its triton arm timed out. Reused as they are, the gate would have compared a
100-step arm with a 4-step one and called the container changed. An arm measured on another
request — or on one the state never recorded — is set aside and re-run, as one measured under
another precision policy already was.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402


def _args(tmp):
    return types.SimpleNamespace(out=str(tmp / "out"), backup=str(tmp / "backup"), models_root=str(tmp / "builds"),
                                 tmp=str(tmp / "tmp"), gpu=None, src=None, extra=[], timeout=10, trace_timeout=10,
                                 restore_mbps=10.0, upload_mbps=10.0)


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "family_of", lambda n: "video")
    monkeypatch.setattr(C, "request_args", lambda n, f, e: ["--prompt", "a red apple", "--seed", "42", "--steps", "4"])
    monkeypatch.setattr(C, "output_ext", lambda f, r: ".mp4")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    m = R.Model("video-test", _args(tmp_path))
    monkeypatch.setattr(m, "autotune_freeze", lambda: {"snapshot": "T", "directory": "d", "replay": "r"})
    monkeypatch.setattr(R, "run", lambda cmd, env, log, timeout: (Path(log).write_text("ran"), 0)[1])
    return m


def _arm_files(m, tag, blob=b"old-render"):
    (m.dir / f"{tag}_sequential.mp4").write_bytes(blob)
    (m.dir / f"{tag}_sequential.log").write_text("x")


def test_an_arm_measured_on_another_request_is_set_aside_and_re_run(model):
    m = model
    _arm_files(m, "old")
    m.state["steps"]["old_outputs"] = {"ok": False, "request": "--prompt a red apple --seed 42"}   # the unbounded request
    res = m.outputs("old")
    aside = list(m.dir.glob("superseded_*_old"))
    assert len(aside) == 1 and (aside[0] / "old_sequential.mp4").read_bytes() == b"old-render"
    assert "--steps 4" in (aside[0] / "WHY.txt").read_text()
    assert res["sequential"].get("cached") is not True                                             # re-run, not reused
    assert m.request_key.endswith("--steps 4")


def test_an_arm_whose_request_the_state_never_recorded_is_set_aside_too(model):
    m = model
    _arm_files(m, "new")
    m.state["steps"]["new_outputs"] = {"ok": False}                                                 # no request recorded
    m.outputs("new")
    aside = list(m.dir.glob("superseded_*_new"))
    assert len(aside) == 1 and "did not record" in (aside[0] / "WHY.txt").read_text()


def test_an_arm_of_the_same_request_is_reused(model, monkeypatch):
    m = model
    _arm_files(m, "old")
    monkeypatch.setattr(R, "sha", lambda p: "abc123")
    m.state["steps"]["old_outputs"] = {"ok": True, "request": "--prompt a red apple --seed 42 --steps 4"}
    res = m.outputs("old")
    assert not list(m.dir.glob("superseded_*"))
    assert res["sequential"]["cached"] is True and res["sequential"]["sha"] == "abc123"


def test_a_row_with_no_output_at_all_is_not_disturbed(model):
    m = model
    m.state["steps"]["old_outputs"] = {"ok": False, "request": "another"}
    m.outputs("old")
    assert not list(m.dir.glob("superseded_*"))


def test_a_verdict_belongs_to_its_request_too(model):
    """A gate recorded on the vendor's unbounded request is not this attempt's verdict: the row
    re-gates on the bounded one. A state that never recorded a request is tolerated — its arms
    are set aside by `outputs` when they run — so the 14 rows already passed are not re-rendered."""
    m = model
    m.state["autotune_freeze"] = {"snapshot": "T"}
    stamp = {"ok": True, "policy": R.POLICY, "autotune": {"snapshot": "T"}}
    m.state["steps"]["gate"] = {**stamp, "verdict": "PASS", "request": "--prompt a red apple --seed 42"}
    assert m.done("gate") is False
    m.state["steps"]["gate"]["request"] = m.current_request()
    assert m.done("gate") is True
    del m.state["steps"]["gate"]["request"]
    assert m.done("gate") is True                                    # unrecorded: tolerated


def test_an_output_step_of_another_request_is_not_done(model):
    m = model
    m.state["autotune_freeze"] = {"snapshot": "T"}
    m.state["steps"]["old_outputs"] = {"ok": True, "policy": R.POLICY, "autotune": "T",
                                       "request": "--prompt a red apple --seed 42"}
    assert m.done("old_outputs") is False
    m.state["steps"]["old_outputs"]["request"] = m.current_request()
    assert m.done("old_outputs") is True
