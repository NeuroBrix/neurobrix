"""An environment A/B whose bytes differ measures BOTH arms against the sequential oracle run
from the same tree without the lever (VibeVoice under the fusion lever, 2026-09-07); an
identical pair runs no oracle. The runs are faked: each command writes the output its arm
is told to, so the test reads the campaign's bookkeeping, not the engine."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def _fake_runs(monkeypatch, bytes_by_output: dict):
    """`run` writes the bytes registered for the output name (A / B / oracle) and exits 0."""
    calls = []
    def fake_run(cmd, env, log, timeout):
        outp = Path(cmd[cmd.index("--output") + 1])
        calls.append((outp.stem, "--sequential" in cmd, env.get("NBX_OPTIM_FUSION_VERTICAL")))
        outp.write_bytes(bytes_by_output[outp.stem])
        Path(log).write_text("ok\n")
        return 0, 1.0
    monkeypatch.setattr(C, "run", fake_run)
    monkeypatch.setattr(C, "family_of", lambda m: "tts")
    monkeypatch.setattr(C, "weight_gb", lambda m: 1.0)
    monkeypatch.setattr(C, "request_args", lambda m, fam, extra: ["--prompt", "x"])
    monkeypatch.setattr(C, "output_ext", lambda fam, req: ".bin")
    monkeypatch.setattr(C, "exec_time", lambda log: 1.0)
    monkeypatch.setattr(C, "gate", lambda a, b: {"kind": "audio", "snr_db": 39.3, "pass": True})
    return calls


def test_differing_arms_are_both_measured_against_the_oracle(tmp_path, monkeypatch):
    calls = _fake_runs(monkeypatch, {"A": b"plain", "B": b"fused", "oracle": b"plain"})
    res = C.env_ab("M", 0, tmp_path, [], 60, {"NBX_OPTIM_FUSION_VERTICAL": "1"}, "env:fusion",
                   src=tmp_path, oracle_on_diff=True)
    assert res["gate"]["identical"] is False
    assert [c[0] for c in calls] == ["A", "B", "oracle"]
    assert calls[2][1] is True and calls[2][2] is None, "the oracle is --sequential and carries no lever"
    assert res["oracle"]["A"] == {"identical": True}
    assert res["oracle"]["B"] == {"identical": False, "kind": "audio", "snr_db": 39.3, "pass": True}
    line = C.verdict(res)
    assert line.startswith("DIFFERENT") and "arms 39.3 dB apart" in line
    assert "vs the sequential oracle: A IDENTICAL, B PASS 39.3 dB" in line


def test_identical_arms_run_no_oracle(tmp_path, monkeypatch):
    calls = _fake_runs(monkeypatch, {"A": b"same", "B": b"same"})
    res = C.env_ab("M", 0, tmp_path, [], 60, {"NBX_OPTIM_FUSION_VERTICAL": "1"}, "env:fusion",
                   src=tmp_path, oracle_on_diff=True)
    assert res["gate"]["identical"] is True and "oracle" not in res
    assert [c[0] for c in calls] == ["A", "B"]
    assert C.verdict(res).startswith("IDENTICAL")
