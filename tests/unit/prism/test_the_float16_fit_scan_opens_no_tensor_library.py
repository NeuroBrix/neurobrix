"""Prism's bfloat16 -> float16 fit scan (`PrismSolver._container_fits_float16`) reads the shards' bfloat16 patterns
itself and asks nbxtensor (one conversion module, one table) — it used to open every shard through
`safetensors.safe_open(framework="pt")`, i.e. torch, in a planning process (R33: Prism is shared by both branches and
its import and run of the Triton branch must load no torch)."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from nbxtensor.dtype.convert import f32_to_bf16_bits

SRC = Path(__file__).resolve().parents[3] / "src"


def _shard(path: Path, tensors: dict):
    header, blobs, off = {}, [], 0
    for name, (code, arr) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": code, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _container(tmp: Path, peak: float) -> Path:
    w = tmp / "components" / "denoiser" / "weights"
    w.mkdir(parents=True)
    bits = f32_to_bf16_bits(np.array([[0.5, -peak], [1.0, 2.0]], dtype=np.float32))
    _shard(w / "model.safetensors", {"a.weight": ("BF16", bits), "b.bias": ("F32", np.ones(3, np.float32))})
    return tmp


def _fits(root: Path) -> str:
    code = ("import sys; from types import SimpleNamespace as N; from pathlib import Path; "
            "from neurobrix.core.prism.solver import PrismSolver; "
            f"r = PrismSolver()._container_fits_float16(N(cache_path=Path({str(root)!r}))); "
            "print(r, 'torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.split()


def test_values_inside_float16_fit_and_beyond_do_not(tmp_path):
    assert _fits(_container(tmp_path / "in", 60000.0))[0] == "True"
    assert _fits(_container(tmp_path / "out", 70000.0))[0] == "False"


def test_the_scan_loads_no_torch(tmp_path):
    assert _fits(_container(tmp_path, 1.0))[1] == "False"
