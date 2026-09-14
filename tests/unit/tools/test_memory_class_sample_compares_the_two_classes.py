"""`tools/memory_class_sample.py`: the measurement register 56 does not contain.

A draft certified on 32 GB is read against the engine directory's 16 GB
entries: same config / different config / no counterpart, counted. Injection:
a draft config that differs → counted DIFFERENT (the tool cannot read
"identical" into a difference)."""
import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "memory_class_sample.py"


def _cert(kw, memory_mb, dev=1e-6, ms=0.1):
    return {"config": {"kwargs": kw, "num_warps": 4, "num_stages": 3},
            "proof": {"date": "2026-09-13T22:00:00+00:00", "engine_version": "0.5.3", "backend": {"name": "cuda"},
                      "shape": [10, 1536, 1536], "deviation": dev, "tolerance": 1e-4, "oracle": "fp64",
                      "machine": {"device": {"ordinal": 0, "name": "V100", "memory_mb": memory_mb}},
                      "best_ms": ms, "built": {"gpu": True}},
            "excluded": []}


def _file(root, entries):
    p = root / "nvidia" / "volta" / "matmul_kernel.fp32.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"format": "nbx-autotune-certified/2", "vendor": "nvidia", "profile": "volta",
                             "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "dtype": "fp32", "entries": entries}))


def test_same_and_different_configs_are_counted_not_reconciled(tmp_path, monkeypatch):
    engine = tmp_path / "engine"; draft = tmp_path / "draft"
    _file(engine, {"(1, 1, 1)": _cert({"BLOCK_M": 64}, 16384), "(2, 2, 2)": _cert({"BLOCK_M": 64}, 16384)})
    _file(draft, {"(1, 1, 1)": _cert({"BLOCK_M": 64}, 32768), "(2, 2, 2)": _cert({"BLOCK_M": 128}, 32768),
                  "(3, 3, 3)": _cert({"BLOCK_M": 64}, 32768)})
    r = subprocess.run([sys.executable, str(TOOL), "draft", str(draft)], capture_output=True, text=True,
                       env={**__import__("os").environ, "NEUROBRIX_AUTOTUNE_CERTIFIED_DIR": str(engine), "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr[-500:]
    assert "1 same config, 1 different, 1 without a counterpart (over 3 keys in the draft)" in r.stdout
    assert "DIFFERENT config" in r.stdout and "| 32 | 16 |" in r.stdout
