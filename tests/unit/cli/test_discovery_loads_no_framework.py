"""Discovery loads no framework: `neurobrix info` (the record under `--json`
and the human listing) reads the hardware through the engine's own detection
(`core/prism/autodetect`, the OS-specific vendor-tool cascade) and the
installed compute stack through the package metadata — never by importing
torch (Studio request 1, item 4 of the engine-side queue).

The client calls `info --json` at start-up; importing torch cost 1.45 s of
its 2.0 s on the rack (2026-10-13, `-X importtime`), and tied the discovery
path to one branch (R33 keeps torch out of the Triton branch; the discovery
path belongs to neither).

What this test does if the code is wrong: each cell runs the command in a
child interpreter and reads `sys.modules` after it — `torch` or `triton`
present fails the cell (seen red on 74312cc4: both `info --json` and
`info --system` imported torch); the record cells fail if the GPUs are no
longer the detection's own list (index, name, memory) or if the stack
version is no longer the installed distribution's.

Run: python -m pytest tests/unit/cli/test_discovery_loads_no_framework.py
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys

import pytest

CHILD = r"""
import json, runpy, sys
sys.argv = ["neurobrix"] + sys.argv[1:]
try:
    runpy.run_module("neurobrix", run_name="__main__", alter_sys=True)
except SystemExit:
    pass
sys.stderr.write("\nLOADED " + json.dumps(sorted(m for m in ("torch", "triton") if m in sys.modules)) + "\n")
"""


def _run(*argv):
    p = subprocess.run([sys.executable, "-c", CHILD, *argv], capture_output=True, text=True, timeout=120)
    loaded = [l for l in p.stderr.splitlines() if l.startswith("LOADED ")]
    assert loaded, p.stderr[-800:]
    return p.stdout, json.loads(loaded[-1][len("LOADED "):]), p.returncode


@pytest.mark.parametrize("argv", [("info", "--json"), ("info", "--system"), ("info",)])
def test_info_loads_neither_torch_nor_triton(argv):
    out, loaded, rc = _run(*argv)
    assert rc == 0, out[-500:]
    assert loaded == [], f"{' '.join(argv)} imported {loaded}"


def test_the_record_reads_the_detection_and_the_installed_stack():
    from neurobrix.core.prism.autodetect import visible_devices
    out, loaded, rc = _run("info", "--json")
    rec = json.loads(out)
    devices, brand = visible_devices(platform.system())
    assert rec["gpus"] == [{"index": d["index"], "name": d["model"], "memory_bytes": d["memory_mb"] * 1024 * 1024}
                           for d in devices]
    assert rec["cuda_available"] == (brand == "nvidia" and bool(devices))
    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    assert rec["torch"] == installed
