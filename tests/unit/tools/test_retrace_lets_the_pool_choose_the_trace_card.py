"""retrace_zoo never pins the trace to a card: Forge's `--device auto` loads on the VMM pool's
primary, the card with the most free memory. A pinned `cuda:0` contradicted a pool 0,3 whose
primary was the 32 GB card, and every host-loaded parakeet-tdt-1.1b component died in
`submodule.to('cuda:0')` with "CUDA error: invalid argument" (retrace5, 2026-10-12 01:07).
Seen RED on 4bf92142 (the command carried `--device cuda:0`).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402


def test_the_trace_command_leaves_the_card_to_the_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FORGE", tmp_path / "forge" / "forge.py")
    monkeypatch.setattr(R, "PY", sys.executable)
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(C, "family_of", lambda n: "stt")
    ran = []
    monkeypatch.setattr(R, "run", lambda cmd, *a, **k: ran.append([str(c) for c in cmd]) or 0)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"),
                                 models_root=str(tmp_path / "builds"), tmp=str(tmp_path / "tmp"), gpu="0,3",
                                 src=str(tmp_path / "src"), extra=[], timeout=10, trace_timeout=10)
    m = R.Model("stt-test", args)
    monkeypatch.setattr(m, "snapshot", lambda: tmp_path)
    monkeypatch.setattr(m, "fixed_point_door", lambda: None)
    assert m.step_trace() is True
    (cmd,) = ran
    assert cmd[2] == "trace" and "--device" not in cmd, cmd
