"""When the two sequential arms differ, the vendor's own render at the exact request decides:
the container that reproduces it passes (PixArt-XL-2, 2026-09-07: the old container's T5 graph
froze the "mask all ones" shortcut — 20.6 dB from the vendor; the retraced one 41.4 dB). A render
of the same request beside the arms is reused, never re-rendered."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402


def test_vendor_verdict_names_the_corrected_container():
    v = R.vendor_verdict({"old": {"psnr_db": 20.6, "pass": False}, "new": {"psnr_db": 41.4, "pass": True}})
    assert v.startswith("PASS (corrected: the old container was wrong") and "41.4" in v and "20.6" in v
    assert R.vendor_verdict({"old": {"psnr_db": 40.0, "pass": True}, "new": {"psnr_db": 20.0, "pass": False}}).startswith("FAIL")
    assert R.vendor_verdict({"old": {"psnr_db": 40.0, "pass": True}, "new": {"psnr_db": 42.0, "pass": True}}).startswith("PASS (both")
    assert R.vendor_verdict({"old": {"psnr_db": 42.0, "pass": True}, "new": {"psnr_db": 40.0, "pass": True}}).startswith("NEEDS_EXPLANATION")


def test_reproduction_reuses_a_render_of_the_same_request(tmp_path, monkeypatch):
    cache = tmp_path / "cache"; (cache / "m").mkdir(parents=True)
    (cache / "m" / "manifest.json").write_text(json.dumps({"created_at": "T"}))
    (cache / "m" / "topology.json").write_text(json.dumps({"flow": {"generation": {"num_inference_steps": 20, "guidance_scale": 4.5}}}))
    monkeypatch.setattr(R, "CACHE", cache)
    monkeypatch.setattr(C, "family_of", lambda n: "image")
    monkeypatch.setattr(C, "request_args", lambda n, fam, extra: ["--prompt", "a red apple on a wooden table", "--seed", "42"])
    snap = tmp_path / "hf_snapshots" / "m"; snap.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path)) if False else None
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"), models_root=str(tmp_path / "builds"),
                                 tmp=str(tmp_path / "tmp"), gpu=None, src=None, extra=[], timeout=10, trace_timeout=10,
                                 restore_mbps=10.0, upload_mbps=10.0, vendor_on_diff=True)
    (tmp_path / "out" / "m").mkdir(parents=True)
    m = R.Model("m", args)
    m.registry_name = "m"
    # a render of the same request already beside the arms
    (m.dir / "vendor_seed42.png").write_bytes(b"png")
    (m.dir / "vendor_seed42.json").write_text(json.dumps({"prompt": "a red apple on a wooden table", "seed": 42, "steps": 20, "guidance": 4.5}))
    calls = []
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(C, "gate", lambda a, b: {"psnr_db": 41.4 if "new" in str(b) else 20.6, "ssim": 0.99, "pass": "new" in str(b), "identical": False})
    # the snapshot lookup walks fixed roots; point the first at our tmp via a fake next()
    monkeypatch.setattr(R.Path, "is_dir", lambda self: str(self).endswith("/hf_snapshots/m") or Path.__dict__["is_dir"](self))
    old = {"sequential": {"output": str(m.dir / "old_sequential.png")}}; new = {"sequential": {"output": str(m.dir / "new_sequential.png")}}
    res = m.vendor_reproduction(old, new)
    assert calls == [], "the render of the same request is reused"
    assert res["old"]["pass"] is False and res["new"]["pass"] is True
    assert R.vendor_verdict(res).startswith("PASS (corrected")


def test_the_vendor_renders_the_length_the_request_pins(tmp_path, monkeypatch):
    """The arms render the request's step count, so the vendor must render it too — its own
    reading of the container's declaration would compare two different renders."""
    cache = tmp_path / "cache"; (cache / "m").mkdir(parents=True)
    (cache / "m" / "topology.json").write_text(json.dumps({"flow": {"generation": {"num_inference_steps": 25, "guidance_scale": 4.0}}}))
    monkeypatch.setattr(R, "CACHE", cache)
    monkeypatch.setattr(C, "family_of", lambda n: "image")
    monkeypatch.setattr(C, "request_args", lambda n, fam, extra: ["--prompt", "p", "--seed", "42", "--steps", "20"])
    snap = tmp_path / "hf_snapshots" / "m"; snap.mkdir(parents=True)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), backup=str(tmp_path / "backup"), models_root=str(tmp_path / "builds"),
                                 tmp=str(tmp_path / "tmp"), gpu=None, src=None, extra=[], timeout=10, trace_timeout=10,
                                 restore_mbps=10.0, upload_mbps=10.0, vendor_on_diff=True)
    (tmp_path / "out" / "m").mkdir(parents=True)
    m = R.Model("m", args); m.registry_name = "m"
    calls = []
    def _run(cmd, **k):
        calls.append(list(cmd)); (m.dir / "vendor_seed42.png").write_bytes(b"png")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(R.subprocess, "run", _run)
    monkeypatch.setattr(C, "gate", lambda a, b: {"psnr_db": 40.0, "ssim": 0.99, "pass": True, "identical": False})
    monkeypatch.setattr(R.Path, "is_dir", lambda self: str(self).endswith("/hf_snapshots/m") or Path.__dict__["is_dir"](self))
    old = {"sequential": {"output": str(m.dir / "old_sequential.png")}}; new = {"sequential": {"output": str(m.dir / "new_sequential.png")}}
    m.vendor_reproduction(old, new)
    assert calls and "--steps" in calls[0]
    assert calls[0][calls[0].index("--steps") + 1] == "20", "the request's pin, not the topology's 25"
