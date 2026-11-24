"""The container judge's arms run under ONE output policy and write only into the judge's own directory.

Both arms of both containers run under the conservative precision policy and one frozen kernel-config state (the
autotune snapshot and one replay cache); an arm measured under another policy is set aside and re-run, never compared;
the judge writes judge/ only and leaves the builder's files byte for byte as it found them (RED_LINES line 28).
What this file does if the judge is wrong: an arm run without the policy, a policy-mismatched arm reused, a file
written under builder/ or a state.json left half-written — red.
"""
from tests.unit.tools.container_judge_rig import J, MODEL, Rig  # noqa: E402


def test_every_arm_runs_under_the_policy_and_the_frozen_kernel_state(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new") == 0
    assert [(c["tag"], c["arm"]) for c in rig.calls] == [("old", "sequential"), ("old", "triton"),
                                                         ("new", "sequential"), ("new", "triton")]
    freeze = rig.state()["autotune_freeze"]
    for c in rig.calls:
        assert all(c["env"].get(k) == v for k, v in J.POLICY_ENV.items())
        assert c["env"]["NEUROBRIX_AUTOTUNE_CERTIFIED_DIR"] == freeze["directory"]
        assert c["env"]["NEUROBRIX_REPLAY_CACHE"] == freeze["replay"]
        assert c["cmd"][c["cmd"].index("--model") + 1] == MODEL
        assert ("--sequential" in c["cmd"]) == (c["arm"] == "sequential")
    for step in ("old_outputs", "new_outputs"):
        st = rig.state()["steps"][step]
        assert st["ok"] and st["policy"] == J.POLICY and st["autotune"] == freeze["snapshot"]


def test_an_arm_measured_under_another_policy_is_set_aside_and_rerun(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    state = rig.state()
    state["steps"]["old_outputs"]["policy"] = "default"
    (rig.judge_dir / "state.json").write_text(J.json.dumps(state))
    j = J.Judge(rig.args("--arm", "old"))
    assert not j.done("old_outputs")
    n = len(rig.calls)
    assert j.step_old_outputs()
    assert len(rig.calls) == n + 2                     # both old arms ran again, none reused
    aside = list((rig.judge_dir / "set_aside").rglob("why.txt"))
    assert aside and "default" in aside[0].read_text()


def test_the_judge_never_writes_into_the_builders_directory(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    before = rig.builder_files()
    assert rig.main("--arm", "new", "--gate") == 0
    assert rig.builder_files() == before
    written = {p.name for p in rig.judge_dir.iterdir()}
    assert {"state.json", "verdict.json", "autotune_directory", "old_sequential.png", "new_triton.png",
            "old_sequential.log"} <= written
    assert not [p for p in rig.judge_dir.rglob("*.tmp")]   # every file written whole, then renamed
