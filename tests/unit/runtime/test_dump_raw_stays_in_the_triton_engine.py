"""NBX_DUMP_RAW in the Triton engine: one brick, no torch, exact values.

Reported by the Mac on 2026-09-26: with NBX_DUMP_RAW set, a triton-sequential
run died on the first matching op. The loop in graph_executor imported torch
and called `.detach().cpu()` on an NBXTensor, which has no `.cpu()`. That is an
R33 violation: a diagnostic path in the Triton branch reached the ATen branch.
The compiled Triton sequence already had an R33-clean mirror, so the same
capability was written twice and only one copy was right.

Now both Triton loops go through `TritonSequence.nbx_dump_raw`. These tests pin:

  * the brick runs in a child where torch is BLOCKED, on real NBXTensors, and
    writes each output's exact values (fp32 as is, bf16 widened exactly),
    named `<component>_<tid>.npy` so it pairs with the ATen `.pt`;
  * the triton-sequential loop calls the brick and imports no torch.

What these would do on the old code: the child fails (`nbx_dump_raw` does
not exist), and the source pin finds `import torch` in the loop.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
sys.path.insert(0, str(REPO / "tests" / "unit"))
from child_env import child_env  # noqa: E402


def _gpu():
    nbx = pytest.importorskip("neurobrix.kernels.nbx_tensor")
    try:
        nbx.DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


_CHILD = textwrap.dedent('''
    import importlib.abc, sys
    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                raise ImportError("R33: " + fullname + " is blocked")
            return None
    sys.meta_path.insert(0, _Blocker())

    import numpy as np
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
    from neurobrix.triton.sequence import TritonSequence
    DeviceAllocator.set_device(0)

    out_dir = sys.argv[1]
    a = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 7.25) / 3.0
    t32 = NBXTensor.from_numpy(a)
    t16 = t32.to(NBXDtype.bfloat16)
    store = {"attn::out": t32, "mlp:gate/out": t16, "other::x": t32}
    TritonSequence.nbx_dump_raw(out_dir + ":attn,gate", "transformer", "op::7",
                                list(store), store.get)
    got32 = np.load(out_dir + "/transformer_attn__out.npy")
    got16 = np.load(out_dir + "/transformer_mlp_gate_out.npy")
    assert got32.dtype == np.float32 and np.array_equal(got32, a), got32
    # bf16 of `a`, computed on the host: round-to-nearest-even on the top half.
    bits = a.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    want16 = rounded.astype(np.uint32).view(np.float32)
    assert got16.dtype == np.float32 and np.array_equal(got16, want16), (got16, want16)
    import os
    assert sorted(os.listdir(out_dir)) == ["transformer_attn__out.npy",
                                           "transformer_mlp_gate_out.npy"], os.listdir(out_dir)
    # An empty filter list matches nothing, as on the ATen side (R30).
    empty = out_dir + "/empty"
    os.mkdir(empty)
    TritonSequence.nbx_dump_raw(empty + ":", "transformer", "op::7", list(store), store.get)
    assert os.listdir(empty) == [], os.listdir(empty)
    assert "torch" not in sys.modules
    print("OK")
''')


@pytest.mark.skipif(not _gpu(), reason="needs a device the NBXTensor allocator can use")
def test_the_brick_dumps_exact_values_with_torch_blocked(tmp_path):
    out = tmp_path / "raw"
    out.mkdir()
    env = child_env(PYTHONPATH=str(SRC), PYTHONNOUSERSITE="1")
    import os
    for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME", "TRITON_CACHE_DIR"):
        if os.environ.get(name) is not None:
            env[name] = os.environ[name]
    r = subprocess.run([sys.executable, "-c", _CHILD, str(out)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stdout[-2000:] + r.stderr[-4000:]


def test_the_triton_sequential_loop_uses_the_brick_and_no_torch():
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    src = inspect.getsource(GraphExecutor._run_triton_sequential)
    assert "nbx_dump_raw(" in src
    assert "import torch" not in src
    assert ".cpu()" not in src
