"""A tensor still alive when the interpreter tears down frees cleanly.

`NBXTensor.__del__` runs at interpreter shutdown for anything still referenced,
and at shutdown `sys.meta_path` is None — so a function-level `import` anywhere on
that path raises `ImportError`. The three range helpers of `DeviceAllocator` each
carried one, and the failure landed AFTER `cudaFree` had already been called, so
it aborted the accounting that follows the free and printed a traceback of its own:

    Exception ignored in: <function NBXTensor.__del__>
      File "nbx_tensor.py", line 1461, in free_cuda
      File "nbx_tensor.py", line 1807, in _range_del
    ImportError: sys.meta_path is None, Python is likely shutting down

Harmless for the memory — the process is exiting and the driver reclaims — but not
harmless as a signal. `free_cuda` deliberately PRINTS rather than raises when
`cudaFree` returns non-zero, because that print is "the sticky-error surfacing
site": it is how an asynchronous fault from an earlier kernel becomes visible. Two
ignored tracebacks per teardown are exactly the noise that gets a real
`[NBX-CUDA-ERROR]` scrolled past or filtered out.

The test asserts on stderr — the artefact — rather than on the presence of the
import, because what must be true is that teardown is quiet, whatever the reason.

Injection that turns it red: remove the module-level `import bisect` from
`nbx_tensor.py` and put the three function-level ones back in `holds`, `_range_add`
and `_range_del` — the state before the fix. Restoring only ONE of them is NOT
enough, because the module-level import keeps `bisect` in `sys.modules` and the
local one then resolves from there; a faithful injection has to restore all four
edits. Seen red that way on 2026-09-18: four lines of `Exception ignored` /
`sys.meta_path is None` on stderr.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src"

# The shape matters, and choosing it was the work. A tensor merely left alive at
# MODULE scope does not reproduce this at all — those are torn down while the
# import machinery still stands, and the first version of this test was green
# against the un-fixed code, which is to say it was empty. What reproduces it is
# a tensor and its materialised transpose created and dropped INSIDE a function:
# the strided-copy path leaves finalizers that run late enough for `sys.meta_path`
# to be None. 256x256 is enough — the fault is in the import, not the size — so
# the cell costs no real memory.
_CHILD = """
from neurobrix.kernels.nbx_tensor import NBXTensor
def body():
    x = NBXTensor.ones((256, 256), dtype="float16")
    y = x.t().contiguous()
body()
print("ALLOCATED", flush=True)
"""


def _cuda_present() -> bool:
    """Asked of torch, deliberately, and not of `libcudart` through ctypes.

    This module sorts first in its directory, so whatever it does at import time
    happens before any other test module loads torch. A bare
    `ctypes.CDLL("libcudart.so")` here resolves the SONAME `libcudart.so.12` to
    the SYSTEM runtime; torch's `libc10_cuda.so` is then linked against that one
    instead of the newer copy torch ships, and every later module that imports
    torch dies with

        ImportError: .../torch/lib/libc10_cuda.so: undefined symbol:
        cudaGetDriverEntryPointByVersion, version libcudart.so.12

    Measured 2026-09-18: with this file present, `pytest tests/unit/kernels`
    collected 859 tests and 11 errors; without it, 1084 and none. Each of the
    eleven passes when run alone, which is what makes the shape worth naming —
    the damage is done by an earlier module and lands on later ones.

    The child process below still uses the real allocator; only this question,
    asked in the parent before anything else, goes through torch.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_present(), reason="needs a CUDA device to allocate on")
def test_a_live_tensor_at_teardown_leaves_stderr_clean():
    env = dict(os.environ, PYTHONPATH=str(_SRC), NBX_ALLOC_POOL="0")
    env.pop("NBX_DEBUG", None)
    p = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True,
                       text=True, env=env, timeout=300)
    assert "ALLOCATED" in p.stdout, f"the child never allocated:\n{p.stdout}\n{p.stderr}"
    assert "Exception ignored" not in p.stderr, (
        "a finalizer raised during interpreter teardown — the accounting after "
        f"cudaFree did not run, and this noise can bury a real [NBX-CUDA-ERROR]:\n{p.stderr}")
    assert "sys.meta_path is None" not in p.stderr, p.stderr
