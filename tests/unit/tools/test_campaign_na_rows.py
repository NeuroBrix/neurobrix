"""A lever row on an engine that refuses the container at its capability gate reads N/A, never FAILED."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def test_a_refused_engine_is_not_applicable():
    r = {"lever": "env:NBX_OPTIM_ALGEBRAIC", "gate": {"kind": "n/a", "reason": "component 'model' stores its weights with encoding 'int4-g128-asym', which the compiled engine does not execute", "pass": None, "ran": False}}
    assert C.verdict(r).startswith("N/A (component 'model' stores its weights with encoding 'int4-g128-asym'")
    assert C.verdict({"lever": "env:X", "gate": {"kind": "bytes", "ran": False}}) == "FAILED (an arm did not run)"


def test_the_tree_gate_proves_a_changed_output_against_the_oracle(tmp_path, monkeypatch):
    """Two trees differ on a model: the last tree's output is run against the sequential oracle
    from that tree, and the verdict says whether the corrected output is the oracle's."""
    calls = []

    def fake_run(cmd, env, log, timeout):
        outp = Path(cmd[cmd.index("--output") + 1])
        arm = outp.stem
        calls.append((arm, env.get("PYTHONPATH"), "--sequential" in cmd))
        outp.write_bytes(b"after-bytes" if arm in ("after", "oracle") else b"before-bytes")
        Path(log).write_text("[Timing] Total execution: 1.00s\n")
        return 0, 1.0
    monkeypatch.setattr(C, "run", fake_run)
    monkeypatch.setattr(C, "family_of", lambda m: "llm")
    monkeypatch.setattr(C, "weight_gb", lambda m: 1.0)
    monkeypatch.setattr(C, "request_args", lambda m, f, e: ["--prompt", "x"] + list(e))
    monkeypatch.setattr(C, "output_ext", lambda f, r: ".txt")
    monkeypatch.setattr(C, "exec_time", lambda log: 1.0)
    monkeypatch.setattr(C, "gate", lambda a, b: {"kind": "text", "identical": a.read_bytes() == b.read_bytes(), "pass": False})
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "/tree"})())
    trees = [("before", tmp_path / "t1" / "src"), ("after", tmp_path / "t2" / "src")]
    res = C.tree_ab("m", 0, tmp_path / "out", ["--triton"], 10, trees, oracle_on_diff=True)
    assert res["gate"]["identical"] is False
    assert res["oracle"]["corrected_identical"] is True and res["oracle"]["tree"] == "after"
    assert res["oracle"]["before_diff"]["identical"] is False                 # the output before the fix was not the oracle's
    assert "before the fix vs the oracle: DIFFERENT" in C.verdict(res)
    assert [c for c in calls if c[2]] == [("oracle", str((tmp_path / "t2" / "src").resolve()), True)]   # the oracle ran from the corrected tree, --sequential
    assert "corrected output IDENTICAL to the sequential oracle" in C.verdict(res)
