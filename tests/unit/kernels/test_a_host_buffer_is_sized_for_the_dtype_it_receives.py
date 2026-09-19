"""Copying a complex tensor to the host must not write past the buffer.

`NBXTensor.empty_cpu` picks its numpy backing from a dict:

    np_dtype_name = {NBXDtype.float32: np.float32, ...}.get(nbx_dt, np.float32)
    arr = np.empty(shape, dtype=np_dtype_name)

There is no `complex64` entry, so the default answers instead, and the host
buffer is allocated at FOUR bytes an element. `to_cpu` then copies
`self._nbytes`, which is EIGHT bytes an element, into it:

    malloc(): corrupted top size

Double the buffer, every time, at every size. It does not always crash where it
happens -- the first small probe printed a correct-looking array and the next
allocation died -- which is what makes it worth a gate rather than a fix alone.

THE SHAPE OF THE DEFECT is `.get(key, default)`: a default makes "a dtype I do
not know" indistinguishable from "float32". Silence where a refusal belongs, the
family this project's register opens with. complex128 is the same bug at four
times the overrun.

RUN IN A SUBPROCESS, deliberately. A heap corruption in the test runner takes the
whole suite with it and reports as an unrelated failure somewhere later; the
child's exit code is the observation, and the parent survives to report it.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_a_host_buffer_is_sized_for_the_dtype_it_receives.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype

_TOTAL = DeviceAllocator.device_count()
needs_a_card = pytest.mark.skipif(_TOTAL == 0, reason="needs a CUDA device")


def _in_a_child(body: str):
    """Run `body` in a fresh interpreter; return (rc, stdout+stderr)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "src")
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                       capture_output=True, text=True, timeout=300, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_the_backing_map_has_an_entry_for_every_dtype_it_can_receive():
    """The cheap half: no card needed, and it names the missing entry.

    `empty_cpu` is reachable with any NBXDtype, so a dtype absent from its map
    is not a theoretical gap -- it is a wrong-sized allocation waiting for that
    dtype to arrive.
    """
    import inspect
    from neurobrix.kernels import nbx_tensor as nt

    src = inspect.getsource(nt.NBXTensor.empty_cpu)
    missing = [d.name for d in NBXDtype if f"NBXDtype.{d.name}" not in src]
    assert not missing, (
        f"empty_cpu's numpy backing map has no entry for {missing}; the "
        f"`.get(..., np.float32)` default will size the host buffer as float32 "
        f"and to_cpu will then copy this dtype's real width into it")


@needs_a_card
@pytest.mark.parametrize("dtype_name,pairs", [("complex64", 2), ("complex128", 2)])
def test_a_complex_tensor_reaches_the_host_without_corrupting_it(dtype_name, pairs):
    rc, out = _in_a_child(f"""
        import numpy as np
        from neurobrix.kernels.nbx_tensor import NBXTensor
        base = np.ascontiguousarray(
            np.random.default_rng(0).standard_normal((4, 8, {pairs}))
            .astype(np.float32 if "{dtype_name}" == "complex64" else np.float64))
        t = NBXTensor.from_numpy(base).view_as_complex()
        a = t.numpy()
        expected = base[..., 0] + 1j * base[..., 1]
        assert a.shape == expected.shape, (a.shape, expected.shape)
        assert np.allclose(a, expected), "values differ"
        # force several allocations afterwards: a corrupted heap dies HERE,
        # not at the copy, which is why the copy alone looked fine.
        for _ in range(64):
            _ = bytearray(1 << 16)
        print("OK")
    """)
    assert rc == 0 and "OK" in out, (
        f"{dtype_name} readback did not survive: rc={rc}\n{out[-800:]}")
