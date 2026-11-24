"""The container judge's gate reads the builder's graph_diff.json, and the vendor decides when the sequential arms
differ.

The gate never compares graphs itself: it reads <out>/<model>/builder/graph_diff.json as the builder wrote it (absent:
refused by name). When the two sequential arms differ and --vendor-on-diff is given, the vendor's own render decides
(vendor_verdict); a corrupted dim left behind refuses whatever the vendor says.
What this file does if the gate is wrong: a verdict without graph_diff.json, a vendor verdict over a corrupted dim, or
no vendor render on a sequential difference — red.
"""
from tests.unit.tools.container_judge_rig import GRAPH_DIFF, J, Rig  # noqa: E402


def _differing(rig, vendor):
    rig.outputs[("new", "sequential")] = b"B"
    rig.outputs[("new", "triton")] = b"B"
    seen = []
    def fake(self, old, new):
        seen.append((old["sequential"]["output"], new["sequential"]["output"]))
        return vendor
    return seen, fake


def test_identical_bytes_and_a_closed_defect_pass_on_the_recorded_diff(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new", "--gate") == 0
    st = rig.state()["steps"]["gate"]
    assert st["graph"] == GRAPH_DIFF and rig.verdict()["verdict"] == "PASS"


def test_the_gate_reads_what_the_builder_recorded_not_its_own_comparison(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build(graph_diff={**GRAPH_DIFF, "beyond_annotation": 4})
    assert rig.main("--arm", "new", "--gate") == 1
    assert rig.verdict()["verdict"] == "FAIL"


def test_an_absent_graph_diff_is_refused_by_name(tmp_path, monkeypatch, capsys):
    rig = Rig(tmp_path, monkeypatch)
    assert rig.main("--arm", "old") == 0
    rig.build()
    (rig.builder / "graph_diff.json").unlink()
    assert rig.main("--arm", "new", "--gate") == 2
    assert "graph_diff.json: absent" in capsys.readouterr().err


def test_a_sequential_difference_is_decided_by_the_vendor(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    vendor = {"old": {"psnr_db": 20.6, "pass": False}, "new": {"psnr_db": 41.4, "pass": True}}
    seen, fake = _differing(rig, vendor)
    monkeypatch.setattr(J.Judge, "vendor_reproduction", fake)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new", "--gate", "--vendor-on-diff") == 0
    v = rig.verdict()
    assert len(seen) == 1 and v["vendor"] == vendor and v["verdict"].startswith("PASS (corrected")
    assert v["pass"] is True


def test_without_the_vendor_a_sequential_difference_needs_an_explanation(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    seen, fake = _differing(rig, {"old": {"pass": True}, "new": {"pass": True}})
    monkeypatch.setattr(J.Judge, "vendor_reproduction", fake)
    assert rig.main("--arm", "old") == 0
    rig.build()
    assert rig.main("--arm", "new", "--gate") == 1
    assert seen == [] and rig.verdict()["verdict"] == "NEEDS_EXPLANATION"


def test_a_corrupted_dim_refuses_whatever_the_vendor_says(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    _, fake = _differing(rig, {"old": {"psnr_db": 20.0, "pass": False}, "new": {"psnr_db": 41.0, "pass": True}})
    monkeypatch.setattr(J.Judge, "vendor_reproduction", fake)
    assert rig.main("--arm", "old") == 0
    rig.build(graph_diff={**GRAPH_DIFF, "corrupted_after": 1})
    assert rig.main("--arm", "new", "--gate", "--vendor-on-diff") == 1
    assert rig.verdict()["verdict"] == "FAIL"


def test_the_vendor_verdict_names_which_container_is_right():
    assert J.vendor_verdict({"old": {"pass": True, "psnr_db": 40}, "new": {"pass": False, "psnr_db": 20}}).startswith("FAIL")
    assert J.vendor_verdict({"old": {"pass": True, "psnr_db": 35}, "new": {"pass": True, "psnr_db": 41}}).startswith("PASS")
    assert J.vendor_verdict({"old": {"pass": True, "psnr_db": 41}, "new": {"pass": True, "psnr_db": 35}}).startswith(
        "NEEDS_EXPLANATION")


def test_an_unrunnable_hub_object_is_judged_on_the_new_engines_and_the_recorded_diff(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.write("previous.json", {"container": "hub previous object", "cache_path": str(rig.cache / "swin2SR-x4"),
                                "created_at": "2026-06-09T10:00:00"})
    rig.outputs[("old", "sequential")] = rig.outputs[("old", "triton")] = None
    assert rig.main("--arm", "old") == 0
    assert "no output" in rig.state()["steps"]["old_outputs"]["unrunnable"]
    layout = [{"path": "components.perception_encoder", "kind": "removed"}]
    rig.build(graph_diff={**GRAPH_DIFF, "topology": layout, "beyond_annotation": 1})
    assert rig.main("--arm", "new", "--gate") == 0
    st = rig.state()["steps"]["gate"]
    assert st["graph"]["topology_layout"] == layout and st["graph"]["beyond_annotation"] == 0
    assert rig.verdict()["verdict"].startswith("PASS (the hub's object does not run")
