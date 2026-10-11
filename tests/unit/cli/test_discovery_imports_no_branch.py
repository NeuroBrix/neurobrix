"""`neurobrix info` is the call NeuroBrix Studio makes at start-up (studio-engine-requests.md, request 4):
it answers from the engine's own torch-free detection (core/prism/autodetect.py), never from a branch.

Measured on the M4 Pro (2026-10-11): `info --json` took 0.62 s, 0.46 s of it importing torch at
cli/commands/info.py, and reported `gpus: []` on a machine with one Apple GPU — torch-CUDA was its only probe.

The cells run `info` in a child interpreter where importing torch RAISES a RuntimeError (an ImportError would be
swallowed by the old `except ImportError`), with the detection pinned to a fixed device so the cell does not
depend on this host's cards nor on autodetect's own last-resort probe.

SEEN RED (2026-10-11): on origin/main 4dd80924 both cells fail — the child dies on the torch import.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"

_CHILD = r"""
import sys, json, io, contextlib
class _NoBranch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("discovery imported torch")
        return None
sys.meta_path.insert(0, _NoBranch())
from neurobrix.core.prism import autodetect
DEVICE = {"index": 0, "brand": "acme", "model": "Acme X1", "memory_mb": 8192, "architecture": "acme_arch",
          "unified_memory": False}
autodetect.detect_devices = lambda: ([dict(DEVICE)], "acme")
from neurobrix.cli import main
sys.argv = ["neurobrix", "info"] + sys.argv[1:]
main()
"""


def _info(*argv):
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, "-c", _CHILD, *argv], capture_output=True, text=True, env=env, timeout=120)


def test_the_record_comes_from_the_engine_detection_without_a_branch():
    r = _info("--json")
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads(r.stdout)
    assert rec["schema"] == "neurobrix.info/2"
    assert rec["gpus"] == [{"index": 0, "name": "Acme X1", "memory_bytes": 8192 << 20, "brand": "acme",
                            "architecture": "acme_arch", "unified_memory": False}]
    assert "torch" not in rec and "cuda_available" not in rec


def test_the_human_listing_reads_the_same_detection_without_a_branch():
    r = _info("--system")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Acme X1" in r.stdout and "8.0 GB" in r.stdout
    assert "PyTorch" not in r.stdout
