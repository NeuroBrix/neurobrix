"""The container judge names what it refuses.

An arm runs on the container the builder's record names or not at all: a cached container whose created_at differs
from previous.json's (old arm) or build.json's (new arm) is REFUSED BY NAME, as is an absent record or --gate on the
old arm. A gate between arms of other containers, other policies or other kernel-config states names the
difference in its verdict, and verdict.json copies `judged` from the build.json it read.
What this file does if the judge is wrong: an arm judged on another container, a silent refusal, or a verdict whose
`judged` is not build.json's — red.
"""
import json

from tests.unit.tools.container_judge_rig import NBX_SHA, NEW_AT, OLD_AT, J, Rig  # noqa: E402


def test_an_old_arm_on_another_container_is_refused_by_name(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    rig.install(NEW_AT)
    assert rig.main("--arm", "old") == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "old arm" in err and NEW_AT in err and OLD_AT in err and "previous.json" in err
    assert rig.calls == []


def test_a_new_arm_on_another_container_is_refused_by_name(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    rig.install(OLD_AT)                                # the cache went back to the old container
    assert rig.main("--arm", "new", "--gate") == 2
    err = capsys.readouterr().err
    assert "new arm" in err and "build.json" in err and len(rig.calls) == 2


def test_an_absent_record_and_a_gate_on_the_old_arm_are_refused_by_name(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "new") == 2
    assert "build.json: absent" in capsys.readouterr().err
    assert rig.main("--arm", "old", "--gate") == 2
    assert "--gate runs with --arm new" in capsys.readouterr().err


def test_a_gate_names_the_arms_that_differ(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new") == 0
    for field, value, words in (("policy", "default", "different precision policies"),
                                ("autotune", "2026-01-01T00:00:00", "different kernel-config states"),
                                ("created_at", "2026-01-01T00:00:00", "previous.json names")):
        state = rig.state()
        state["steps"]["old_outputs"] = {**state["steps"]["old_outputs"], field: value}
        state["steps"].pop("gate", None)
        (rig.judge_dir / "state.json").write_text(json.dumps(state))
        assert J.Judge(rig.args("--arm", "new", "--gate")).step_gate() is False
        v = rig.verdict()
        assert v["pass"] is False and words in v["verdict"], (field, v["verdict"])
        state["steps"]["old_outputs"][field] = rig.state()["steps"]["new_outputs"].get(field) if field != "created_at" else OLD_AT
        (rig.judge_dir / "state.json").write_text(json.dumps(state))


def test_the_verdict_names_the_containers_it_judged(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new", "--gate") == 0
    v = rig.verdict()
    assert v["pass"] is True
    assert v["judged"] == {"created_at": NEW_AT, "nbx_sha256": NBX_SHA}   # computed on the staged file
    assert v["old_container"] == OLD_AT
    assert set(v) >= {"verdict", "pass", "bytes", "vendor", "request", "policy", "autotune", "judged", "old_container"}
