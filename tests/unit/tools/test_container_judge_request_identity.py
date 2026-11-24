"""The request is part of an arm's identity in the container judge.

An arm measured on another request is set aside and re-run, never compared with this one's; a gate between arms that
ran on different requests is refused by name; and verdict.json carries the request the arms ran.
What this file does if the judge is wrong: a cached arm of another request reused, or a gate across two requests
judged — red.
"""
from tests.unit.tools.container_judge_rig import J, Rig  # noqa: E402


def test_the_request_is_read_from_the_campaign_and_memoised(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    j = J.Judge(rig.args("--arm", "old"))
    assert j.current_request("old") == "--prompt a red apple --seed 42"
    rig.request = ["--prompt", "something else"]
    assert j.current_request("old") == "--prompt a red apple --seed 42"


def test_an_arm_of_another_request_is_set_aside_and_rerun(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    assert rig.main("--arm", "old") == 0              # done: nothing runs again
    assert len(rig.calls) == 2
    rig.request = ["--prompt", "a red apple", "--seed", "42", "--steps", "8"]
    assert rig.main("--arm", "old") == 0
    assert len(rig.calls) == 4                         # the new request ran both arms again
    assert rig.state()["steps"]["old_outputs"]["request"] == "--prompt a red apple --seed 42 --steps 8"
    why = next((rig.judge_dir / "set_aside").rglob("why.txt")).read_text()
    assert "--prompt a red apple --seed 42" in why and "--steps 8" in why


def test_a_gate_across_two_requests_is_refused_by_name(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.request = ["--prompt", "a green pear", "--seed", "42"]
    rig.build()
    assert rig.main("--arm", "new", "--gate") == 1
    v = rig.verdict()
    assert v["pass"] is False and "different requests" in v["verdict"] and "a green pear" in v["verdict"]


def test_the_verdict_carries_the_request_the_arms_ran(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new", "--gate") == 0
    v = rig.verdict()
    assert v["request"] == "--prompt a red apple --seed 42" and v["policy"] == J.POLICY
