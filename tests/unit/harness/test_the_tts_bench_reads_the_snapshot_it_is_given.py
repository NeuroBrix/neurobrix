"""The TTS bench's vendor arm (benchmarks/harness/bench_tts.py, kokoro) reads the snapshot it is given, never the hub:
a bare `KPipeline(lang_code="a")` downloads its config, weights and voice into the Hugging Face cache
(tests/unit/workshop/test_the_hugging_face_cache_stays_empty.py). Card-less and network-less: kokoro and soundfile are
stand-ins that record what they are handed.

Injections (each RED, then GREEN): the arm's script as it was (`KPipeline(lang_code="a")`, voice by name) — the script
test fails; the arm handing its script no snapshot and no HF_HUB_OFFLINE — the arm test fails.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "harness"))
import bench_tts  # noqa: E402


def _snapshot(tmp_path):
    snap = tmp_path / "snap"
    (snap / "voices").mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "weights.pth").write_bytes(b"")
    (snap / "voices" / "af_heart.pt").write_bytes(b"")
    return snap


def test_the_kokoro_script_builds_its_model_and_voice_from_the_snapshot(tmp_path, monkeypatch):
    snap, seen = _snapshot(tmp_path), {}

    class KModel:
        def __init__(self, **kw):
            seen["model"] = kw

        def to(self, device):
            return self

        def eval(self):
            return self

    class KPipeline:
        def __init__(self, lang_code, **kw):
            seen["pipeline"] = kw

        def __call__(self, text, voice):
            seen["voice"] = voice
            yield "g", "p", np.zeros(4, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=KModel, KPipeline=KPipeline))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(write=lambda *a, **k: None))
    monkeypatch.setattr(sys, "argv", ["-c", "a sentence", str(tmp_path / "out.wav"), str(snap)])
    exec(compile(bench_tts._KOKORO_SCRIPT, "<kokoro arm>", "exec"), {"__name__": "__main__"})
    assert seen["model"] == {"repo_id": str(snap), "config": str(snap / "config.json"),
                             "model": str(snap / "weights.pth")}, seen
    assert seen["pipeline"]["repo_id"] == str(snap) and isinstance(seen["pipeline"]["model"], KModel), seen
    assert seen["voice"] == str(snap / "voices" / "af_heart.pt"), seen


def test_the_kokoro_arm_hands_the_snapshot_and_refuses_the_hub(tmp_path, monkeypatch):
    snap, calls = _snapshot(tmp_path), []
    monkeypatch.setattr(bench_tts, "_locked", lambda args, fn: fn())
    monkeypatch.setattr(bench_tts, "gpu_state", lambda gpu: {})
    monkeypatch.setattr(bench_tts.subprocess, "run",
                        lambda cmd, env, **kw: calls.append((cmd, env)) or SimpleNamespace(returncode=0, stdout="",
                                                                                           stderr=""))
    args = SimpleNamespace(gpu="", sentence_text="a sentence", snapshot=str(snap))
    bench_tts.run_once_kokoro(args, {"KOKORO_PYTHON": "py"}, "k_1", tmp_path)
    (cmd, env), = calls
    assert cmd[-1] == str(snap) and env.get("HF_HUB_OFFLINE") == "1", (cmd[3:], env.get("HF_HUB_OFFLINE"))
    with pytest.raises(SystemExit, match="--snapshot"):
        bench_tts.run_once_kokoro(SimpleNamespace(**{**vars(args), "snapshot": None}), {"KOKORO_PYTHON": "py"}, "k_2",
                                  tmp_path)
