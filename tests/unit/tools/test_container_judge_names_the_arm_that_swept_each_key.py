"""The gate names the arm that swept each key, and never reuses a verdict stored before the swept-key rule.

The replay cache judge/autotune_replay/ is shared by the old and the new arms. Each arm's run records the keys the
cache gained during that run (its key set read before and after the run), so every missing key is named with the arm
that swept it: verdict.json's autotune.missing is [{key, arm}], and the RED message says "swept by the old arm" or
"swept by the new arm". A key the OLD arm swept keeps the gate red: any key swept during the judging of this
container at its conventional size is a missing certificate. verdict.json and the gate's state carry rule_version;
a stored verdict with none (or an older one) is not reused: the gate re-judges, said in the log.
What this file does if the judge is wrong: an old-arm sweep unnamed or green, a new-arm sweep named as the old arm's,
or a verdict stored before the rule passed on as it is — red.
"""
import json

from tests.unit.tools.container_judge_rig import J, Rig  # noqa: E402

OLD_KEY = "matmul_kernel::(128, 64, 32, 'fp16')"
NEW_KEY = "softmax_forward_kernel::(4096, 'fp32')"
OLD_SWEEP = {"nbxtensor.ops.matmul.matmul_kernel::(128, 64, 32, 'fp16')": {"kwargs": {}}}
NEW_SWEEP = {"nbxtensor.ops.softmax.softmax_forward_kernel::(4096, 'fp32')": {"kwargs": {}}}


def _arms(rig):
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new") == 0


def test_a_key_the_old_arm_swept_turns_the_gate_red_naming_the_old_arm(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    rig.sweeps[("old", "triton")] = OLD_SWEEP
    _arms(rig)
    assert rig.state()["steps"]["old_outputs"]["runs"]["triton"]["swept"] == [OLD_KEY]
    assert rig.state()["steps"]["new_outputs"]["runs"]["triton"]["swept"] == []     # loaded, not swept again
    capsys.readouterr()
    assert rig.main("--arm", "new", "--gate") == 1
    v, gate = rig.verdict(), rig.state()["steps"]["gate"]
    assert v["pass"] is False and gate["ok"] is False
    for at in (v["autotune"], gate["autotune"]):
        assert at["missing"] == [{"key": OLD_KEY, "arm": "old"}] and at["missing_count"] == 1
    for text in (v["verdict"], gate["verdict"], capsys.readouterr().err):
        assert f"{OLD_KEY} (swept by the old arm)" in text
        assert "whichever arm swept a key, the gate is red" in text and "conventional size" in text


def test_a_key_the_new_arm_swept_names_the_new_arm(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    rig.sweeps[("new", "sequential")] = NEW_SWEEP
    rig.sweeps[("old", "sequential")] = OLD_SWEEP
    _arms(rig)
    capsys.readouterr()
    assert rig.main("--arm", "new", "--gate") == 1
    v = rig.verdict()
    assert v["pass"] is False
    assert v["autotune"]["missing"] == [{"key": OLD_KEY, "arm": "old"}, {"key": NEW_KEY, "arm": "new"}]
    assert f"{NEW_KEY} (swept by the new arm)" in v["verdict"] and f"{OLD_KEY} (swept by the old arm)" in v["verdict"]


def test_the_arm_record_survives_a_rerun_on_cached_outputs(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.sweeps[("old", "sequential")] = OLD_SWEEP
    rig.outputs[("old", "triton")] = None         # the old triton run fails: the old arm is re-run ...
    assert rig.main("--arm", "old") == 1
    rig.outputs[("old", "triton")] = b"A"
    assert rig.main("--arm", "old") == 0          # ... its sequential output reused, the run's record kept
    runs = rig.state()["steps"]["old_outputs"]["runs"]
    assert runs["sequential"]["cached"] is True and runs["sequential"]["swept"] == [OLD_KEY]
    rig.build()
    assert rig.main("--arm", "new") == 0
    assert rig.main("--arm", "new", "--gate") == 1
    assert rig.verdict()["autotune"]["missing"] == [{"key": OLD_KEY, "arm": "old"}]


def test_a_verdict_stored_without_the_rule_version_is_not_reused(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    rig.sweeps[("new", "triton")] = NEW_SWEEP
    _arms(rig)
    # A green verdict as a judge before the swept-key rule stored it: no rule_version anywhere.
    state = rig.state()
    snap = state["autotune_freeze"]["snapshot"]
    state["steps"]["gate"] = {"ok": True, "verdict": "PASS", "policy": J.POLICY, "autotune": {"snapshot": snap},
                              "request": state["steps"]["new_outputs"]["request"]}
    (rig.judge_dir / "state.json").write_text(json.dumps(state))
    (rig.judge_dir / "verdict.json").write_text(json.dumps({"verdict": "PASS", "pass": True}))
    capsys.readouterr()
    assert rig.main("--arm", "new", "--gate") == 1
    v = rig.verdict()
    assert v["pass"] is False and v["rule_version"] == J.RULE_VERSION
    assert rig.state()["steps"]["gate"]["rule_version"] == J.RULE_VERSION
    assert v["autotune"]["missing"] == [{"key": NEW_KEY, "arm": "new"}]
    err = capsys.readouterr().err
    assert "rule_version None" in err and "not reused" in err and "re-judges" in err


def test_an_older_rule_version_is_not_reused_and_the_current_one_is(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    _arms(rig)
    assert rig.main("--arm", "new", "--gate") == 0
    assert rig.verdict()["rule_version"] == J.RULE_VERSION
    capsys.readouterr()
    assert rig.main("--arm", "new", "--gate") == 0
    assert "not reused" not in capsys.readouterr().err        # the current version: reused
    v = rig.verdict()
    (rig.judge_dir / "verdict.json").write_text(json.dumps({**v, "rule_version": J.RULE_VERSION - 1}))
    assert rig.main("--arm", "new", "--gate") == 0
    err = capsys.readouterr().err
    assert f"rule_version {J.RULE_VERSION - 1}" in err and "not reused" in err
    assert rig.verdict()["rule_version"] == J.RULE_VERSION
