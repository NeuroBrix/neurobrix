"""The failures a user met on the published 0.6.2 path (the terminal-noise inventory, 2026-11-08), each now said in one
line with its remedy — or gone.

* A machine without a C compiler (a `python:3.10-slim` container): torch 2.14.1 compiles its own Triton kernel for
  `aten::bmm` at its first use and Triton builds its launcher with the machine's C compiler (`triton/runtime/build.py`,
  `_find_compiler`). Sana on the default engine ended in a 71-line traceback; now one line naming the missing compiler,
  how to install one, and `--triton`, which needs none.
* The first launch measured the reload rate of the weights of EVERY installed model, whatever was asked; it measures
  what the requested model's plan prices (its own stored dtypes) and says it in one line with the time it expects.
* `import` of a model installed under the container's own name (the hub slug `..._diffusers`, the container
  `Sana_1600M_1024px_MultiLing`) downloaded 9 GB before it said so; it is decided before any byte is fetched.
* `NEUROBRIX_HOME` naming a directory that does not exist ended every command, `--help` included, in a traceback; one
  line with the remedy now, the traceback under -vv.
* `doctor` listed every GPU on the host's PCI bus (4) where the process sees one; it lists the ones it sees.

What these tests would do if the code were wrong: a traceback or a multi-line error (red), an import that asks the
registry for a download URL (red), a first launch that probes a dtype the requested model does not store (red), a GPU
the process does not see listed as its own (red). Each was seen red on main caafe74c8 (2026-11-08)."""
from __future__ import annotations

import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("NBX_LOG", "NBX_DEBUG")}
    env.update(PYTHONNOUSERSITE="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "nbxtensor" / "src")]))
    env.update(extra)
    return env


def _one_card() -> bool:
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.device_count() == 1
    except Exception:
        return False


# ── (2) the PyTorch engine on a machine without a C compiler ─────────────────────────────────────────────────────

_NO_CC_RUN = r'''
import json, os, sys, types
from pathlib import Path
model, cache = sys.argv[1], Path(sys.argv[2])
import neurobrix.core.runtime.loader as L
import neurobrix.core.runtime.executor as E

class _Loader:                      # the container's manifest, no weight read: the failure under test is at execution
    def load(self, path):
        return types.SimpleNamespace(manifest=json.loads((cache / "manifest.json").read_text()), topology={},
                                     defaults={})

class _Executor:
    """What torch does at its first Triton kernel on such a machine: Triton's own build, with no compiler to find."""
    def __init__(self, *a, **k):
        pass
    def execute(self, inputs):
        from triton.runtime import build
        build._find_compiler.cache_clear()
        os.environ.pop("CC", None)
        os.environ["PATH"] = "/nonexistent"
        try:
            build._build("probe", "probe.c", "/nonexistent", [], [], [], [])
        except RuntimeError as e:   # the shape the compiled engine reports an op's failure in
            raise RuntimeError(f"Failed at op aten.bmm::0 (aten::bmm): {e}") from e

L.NBXRuntimeLoader, E.RuntimeExecutor = _Loader, _Executor
sys.argv = ["neurobrix", "run", "--model", model, "--prompt", "x", "--max-tokens", "1", "--compiled"]
from neurobrix.cli import main
main()
'''


@pytest.mark.skipif(not _one_card(), reason="needs exactly one visible card (CUDA_VISIBLE_DEVICES=<n>)")
def test_a_missing_c_compiler_is_one_line_with_its_remedy(tmp_path):
    pytest.importorskip("torch")
    from neurobrix.core.paths import cache_dir
    model = "TinyLlama-1.1B-Chat-v1.0"
    if not (cache_dir() / model / "manifest.json").exists():
        pytest.skip(f"{model} is not installed")
    driver = tmp_path / "no_cc_run.py"
    driver.write_text(_NO_CC_RUN)
    p = subprocess.run([PY, str(driver), model, str(cache_dir() / model)], capture_output=True, text=True,
                       env=_env(), timeout=900, cwd=str(tmp_path))
    err = p.stderr.splitlines()
    assert p.returncode == 1 and not p.stdout.strip(), (p.stdout[-2000:], p.stderr[-2000:])
    named = [l for l in err if "C compiler" in l]
    assert len(named) == 1 and "apt install gcc" in named[0] and "--triton" in named[0], err
    assert not any("Traceback" in l for l in err) and len(err) <= 2, err


def test_the_compiler_condition_is_read_where_triton_raises_it(monkeypatch):
    from triton.runtime import build
    from nbxtensor.backend.install_help import compiler_missing as compiler_missing_message
    build._find_compiler.cache_clear()
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    try:
        build._find_compiler("c")
    except RuntimeError as inner:
        try:
            raise RuntimeError("Failed at op aten.bmm::0") from inner
        except RuntimeError as outer:
            assert "C compiler" in compiler_missing_message(outer)
    finally:
        build._find_compiler.cache_clear()
    # the same words raised anywhere else are not the condition: read from the frame, never from the text
    assert compiler_missing_message(RuntimeError("Failed to find C compiler. Please specify via CC")) is None


# ── (3) a first launch measures what the requested model's plan prices ───────────────────────────────────────────

def test_a_first_launch_measures_the_requested_models_stored_dtypes_only(tmp_path, monkeypatch, caplog):
    import logging
    import yaml
    from neurobrix.core.prism import autodetect as A
    from neurobrix.core.prism import arena_probe

    def container(name, comps):
        root = tmp_path / "cache" / name
        for comp, dtype, size in comps:
            d = root / "components" / comp
            (d / "weights").mkdir(parents=True)
            (d / "weights" / "w.safetensors").write_bytes(b"\0" * size)
            (d / "weights_index.json").write_text(json.dumps({"tensors": {"w": {"dtype": dtype, "size_bytes": size}}}))
        (root / "manifest.json").write_text(json.dumps({"model_name": name}))
        return root
    mb = 1 << 20
    asked = container("asked", [("model", "bfloat16", 300 * mb)])
    container("other_f16", [("decoder", "float16", 300 * mb)])
    container("other_f32", [("vae", "float32", 300 * mb)])
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path / "cache"))
    hw = tmp_path / "hardware"
    hw.mkdir()
    measured = {"source": "test", "matmul_tflops": {"float16": 1.0}, "gemv_gbps": {"triton": {"float16": 1.0}},
                "expert_copy": {"x": 1}}                    # vmm_chunk_bytes unmeasured: its replays are priced too
    (hw / "default-t.yml").write_text(yaml.safe_dump({
        "devices": [{"memory_mb": 16000, "supports_dtypes": ["float16"], "measured": measured}],
        "host_measured": {"host_matmul_gflops": 1, "host_memory_gbps": 1}}))
    monkeypatch.setattr(A, "HARDWARE_DIR", hw)
    from neurobrix.core.prism import facts_probe as P
    handed = []
    # the chunk probe's first replay, measured ahead: 1.0 s replayed, 2.0 s for its whole child here
    monkeypatch.setattr(P, "vmm_first_replay", lambda dev: (1.0, 2.0))
    monkeypatch.setattr(P, "vmm_chunk_bytes", lambda dev, reps, ks, off_first=None: handed.append(off_first) or 0)
    probed = []
    monkeypatch.setattr(arena_probe, "child", lambda d, c, h, e: probed.append(Path(d).name) or
                        {"container": Path(d).name, "component": c, "draws": [(1, 1.0)]})
    A._FIRST_LAUNCH_SAID.clear()
    with caplog.at_level(logging.INFO, logger="neurobrix"):
        A.ensure_measured_facts("default-t", "triton", container_dir=asked)
    assert probed == ["asked"], f"probed {probed}: the reload rate of models nobody asked for"
    said = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(said) == 1 and said[0].startswith("First launch on this machine"), said
    # priced from what was measured: the 23 other replays at 2.0 s each (46 s) and the reload probe — at least 48 s
    m = re.search(r"up to about (\d+) (s|min)", said[0])
    assert m and (int(m.group(1)) * (60 if m.group(2) == "min" else 1)) >= 48, said[0]
    assert handed == [1.0], "the replay made ahead is one of the probe's own, never made twice"


# ── (4) an installed model is never downloaded again ─────────────────────────────────────────────────────────────

class _Hub(http.server.BaseHTTPRequestHandler):
    asked = []

    def do_GET(self):  # noqa: N802
        _Hub.asked.append(self.path)
        if self.path.endswith("/download"):
            self.send_response(500)
            self.end_headers()
            return
        body = json.dumps({"org": "vendor", "name": "Model_1B", "slug": "vendor/Model_1B_diffusers",
                           "fileUrl": "models/vendor/Model_1B.nbx", "fileSize": "9694994270", "category": "IMAGE",
                           "license": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_an_import_of_a_model_installed_under_its_own_name_downloads_nothing(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    home = tmp_path / "home"
    (home / "cache" / "Model_1B").mkdir(parents=True)
    (home / "cache" / "Model_1B" / "manifest.json").write_text(json.dumps({"model_name": "Model_1B"}))
    (home / "store").mkdir()
    _Hub.asked.clear()
    try:
        p = subprocess.run([PY, "-m", "neurobrix", "import", "vendor/Model_1B_diffusers", "--accept-license",
                            "--registry", f"http://127.0.0.1:{srv.server_address[1]}"], capture_output=True, text=True,
                           env=_env(NEUROBRIX_HOME=str(home), CUDA_VISIBLE_DEVICES=""), timeout=300)
    finally:
        srv.shutdown()
    assert not any(a.endswith("/download") for a in _Hub.asked), f"a download was asked: {_Hub.asked}"
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines() == ["Model_1B is already installed (--force re-downloads)."], p.stdout


# ── (5) a configured home that does not exist ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("argv", [["--help"], ["list"], ["run", "--model", "x", "--prompt", "y"]])
def test_a_home_that_does_not_exist_is_one_line_with_its_remedy(tmp_path, argv):
    missing = tmp_path / "not_created"
    p = subprocess.run([PY, "-m", "neurobrix", *argv], capture_output=True, text=True,
                       env=_env(NEUROBRIX_HOME=str(missing), CUDA_VISIBLE_DEVICES=""), timeout=300)
    err = p.stderr.splitlines()
    assert p.returncode == 1 and not p.stdout, (p.stdout, p.stderr)
    assert len(err) == 1 and str(missing) in err[0] and "Create or mount" in err[0], err
    p = subprocess.run([PY, "-m", "neurobrix", *argv[:1], "-vv", *argv[1:]], capture_output=True, text=True,
                       env=_env(NEUROBRIX_HOME=str(missing), CUDA_VISIBLE_DEVICES=""), timeout=300)
    assert "Traceback (most recent call last)" in p.stderr       # -vv: the traceback, for whoever debugs it
    assert not missing.exists()


# ── (6) doctor lists the GPUs this process sees ──────────────────────────────────────────────────────────────────

def test_doctor_lists_only_the_gpus_this_process_sees(tmp_path, monkeypatch):
    from nbxtensor.backend import install_help as H
    for i, bus in enumerate(("0000:1a:00.0", "0000:1b:00.0", "0000:3d:00.0", "0000:3e:00.0")):
        d = tmp_path / bus
        d.mkdir()
        (d / "class").write_text("0x030200\n")
        (d / "vendor").write_text("0x10de\n")
        (d / "device").write_text(("0x1db1" if i < 2 else "0x1db5") + "\n")
    monkeypatch.setattr(H, "_loads", lambda names: True)
    monkeypatch.setattr(H, "_triton_backends", lambda: ["amd", "nvidia"])
    # the runtime sees one card, the third on the bus (CUDA_VISIBLE_DEVICES=2, a container's --gpus device=2), and
    # names it with its own padding of the PCI address
    seen = H._visible_bus_ids(count=1, bus_of=lambda i: "0000:3D:00.0")
    assert seen == ("cuda", {(0, 0x3D, 0, 0)}) or seen[1] == {(0, 0x3D, 0, 0)}
    monkeypatch.setattr(H, "_visible_bus_ids", lambda: ("cuda", {(0, 0x3D, 0, 0)}))
    text = H.report(str(tmp_path))
    assert text.startswith("NVIDIA GPU (0x1db5; 3 more on the host, not visible here)"), text


def test_a_probe_child_reads_serially_when_its_launch_measured_no_read_rate(monkeypatch, capsys):
    """A first launch on a busy storage measures no stable read rate and reads serially (`workspace.
    declare_unmeasured_io_workers`); its reload probe child read the same profile, found no count and refused — the
    whole first run ended in `RuntimeError: ... no I/O worker count` (a loaded rack, 2026-11-08). The parent hands its
    word to the child, and the child reads serially too."""
    import runpy
    import types
    from neurobrix.core import workspace as W
    from neurobrix.core.prism import arena_probe as AP
    seen = {}
    monkeypatch.setattr(W, "_IO_UNMEASURED", "no stable read rate this time")
    monkeypatch.setattr(AP.subprocess, "run", lambda cmd, **k: seen.update(env=k["env"]) or
                        types.SimpleNamespace(returncode=0, stdout=AP.MARK + "{}", stderr=""))
    AP.child("/nowhere", "model", "default-x", "triton")
    assert seen["env"][W.UNMEASURED_ENV] == "no stable read rate this time"
    # the child's side: its entry declares it before anything reads the count
    monkeypatch.setattr(W, "_IO_UNMEASURED", None)
    monkeypatch.setattr(W, "_IO_WORKERS", None)
    monkeypatch.setenv(W.UNMEASURED_ENV, "no stable read rate this time")
    monkeypatch.setattr(sys, "argv", ["arena_probe", "/nowhere", "model", "default-x", "1"])
    with pytest.raises(SystemExit):
        runpy.run_module("neurobrix.core.prism.arena_probe", run_name="__main__")
    capsys.readouterr()
    assert W.io_workers() == W.UNMEASURED_IO_WORKERS
