"""Out-of-range index → LOUD failure on the three gather/scatter kernels
(D-GATHER-SCATTER-OOB-SILENT, promoted to correctness 2026-09-02).

torch raises on an out-of-range index in index_select / index_put /
embedding. The FlagGems-lineage Triton ports were silent: index_select
skipped the store (the output kept whatever the allocation held — the
pool gate caught it through a stale-memory DIFF), index_put wrote
outside the tensor, embedding read the row at the out-of-range address.
A byte gate whose two sides share the defect cannot see it; the only
parity is a trap where torch raises.

Mechanism: `tl.device_assert` inside each kernel, kept in the binary by
`@triton.jit(debug=True)` (TRITON_DEBUG-independent), no host-side
sync, no extra launch. The trap poisons the context; the next device
sync raises (DeviceAllocator.sync_device re-raises the sticky error).

Pins:
  CPU (no device): the three kernels carry debug=True, the source of
      each holds its device_assert, the bound parameters exist.
  GPU (skipped without a device): a subprocess per kernel — an
      in-range control exits 0 with the torch-equal values; an
      out-of-range index exits non-zero with the assert reported.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_gather_scatter_oob.py -v
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
import subprocess
import sys

import pytest

from neurobrix.kernels.ops.embedding import embedding_kernel
from neurobrix.kernels.ops.index_put_op import index_put_kernel
from neurobrix.kernels.ops.index_select import index_select_kernel

_KERNELS = {
    "index_select": (index_select_kernel, "N"),
    "index_put": (index_put_kernel, "R"),
    "embedding": (embedding_kernel, "V"),
}


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_kernel_keeps_its_device_assert(name):
    fn, bound = _KERNELS[name]
    assert getattr(fn, "debug", None) is True, f"{name}: debug=True dropped — the assert would compile out"
    src = inspect.getsource(fn.fn)
    assert "tl.device_assert(" in src, f"{name}: no device_assert in the kernel"
    assert bound in fn.arg_names, f"{name}: bound parameter {bound!r} missing from the signature"


_SUBPROCESS = r"""
import sys, numpy as np
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers as w
kind, oob = sys.argv[1], sys.argv[2] == "oob"
rows = np.arange(24, dtype=np.float32).reshape(6, 4)
if kind == "index_select":
    idx = np.array([1, 3, 6 + 5 if oob else 5], dtype=np.int64)
    out = w.index_select_wrapper(NBXTensor.from_numpy(rows), 0, NBXTensor.from_numpy(idx))
    DeviceAllocator.sync_device()
    got = out.numpy(); exp = rows[idx]
elif kind == "index_put":
    idx = np.array([0, 6 + 2 if oob else 2], dtype=np.int64)
    vals = np.full((2, 4), -1.0, dtype=np.float32)
    out = w.index_put_wrapper(NBXTensor.from_numpy(rows), [NBXTensor.from_numpy(idx)], NBXTensor.from_numpy(vals))
    DeviceAllocator.sync_device()
    got = out.numpy(); exp = rows.copy(); exp[idx] = vals
else:
    ids = np.array([[4, 6 + 1 if oob else 1]], dtype=np.int64)
    out = w.embedding(NBXTensor.from_numpy(rows), NBXTensor.from_numpy(ids))
    DeviceAllocator.sync_device()
    got = out.numpy(); exp = rows[ids]
assert np.array_equal(got, exp), (got, exp)
print("CONTROL OK", kind)
"""


def _run(kind: str, mode: str):
    env = dict(os.environ, PYTHONPATH=str(_REPO / "src"))   # absolute: runnable from any cwd
    return subprocess.run([sys.executable, "-c", _SUBPROCESS, kind, mode],
                          capture_output=True, text=True, env=env, timeout=300)


def _no_device() -> bool:
    try:
        from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor
        NBXTensor.empty((1,), NBXDtype.float32, "cuda:0")
        return False
    except Exception:
        return True


@pytest.mark.skipif(_no_device(), reason="needs a GPU")
@pytest.mark.parametrize("kind", sorted(_KERNELS))
def test_in_range_control_matches_torch_semantics(kind):
    r = _run(kind, "ok")
    assert r.returncode == 0, r.stderr[-2000:]
    assert f"CONTROL OK {kind}" in r.stdout


@pytest.mark.skipif(_no_device(), reason="needs a GPU")
@pytest.mark.parametrize("kind", sorted(_KERNELS))
def test_out_of_range_index_fails_loudly(kind):
    r = _run(kind, "oob")
    assert r.returncode != 0, f"{kind}: an out-of-range index passed silently\n{r.stdout}"
    blob = (r.stdout + r.stderr).lower()
    assert "assert" in blob or "index out of range" in blob, r.stderr[-2000:]
