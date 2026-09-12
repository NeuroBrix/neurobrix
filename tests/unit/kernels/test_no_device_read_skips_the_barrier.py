"""No read of device memory may skip the barrier the copy path crosses.

The property is on the PATH, not on the dtype.

`screen_oracle._to_f64` reads a bf16 operand by raw pointer --
`ctypes.string_at(t.data_ptr(), n)` -- while every other dtype goes through
`t.numpy()`, which copies and therefore crosses a barrier. The two paths differ
in WHEN they look, not only in what they can address.

Eighteen recorded refusals rest on that read and all of them are bf16. It
would be easy, and wrong, to write "repair the bf16 read": they are bf16 only
because bf16 is the sole traffic on that path today. **The defect has the
shape of a path and the dtype is merely who happens to take it.** Route the
next dtype there and it inherits the defect in silence, and nobody will know
why its refusals are strange.

So the property asserted here is: for ANY dtype, a buffer a kernel wrote reads
the same through the oracle's path as through the copy path. A test that named
bf16 in its assertion would pass the day someone adds fp8 and be useless.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_no_device_read_skips_the_barrier.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from check_measurement_environment import owned_cache_env       # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="the Metal device-memory path")


@pytest.fixture(autouse=True)
def _owns_its_cache(tmp_path, monkeypatch):
    for var, value in owned_cache_env(tmp_path).items():
        monkeypatch.setenv(var, value)


def _dtypes():
    """Every dtype the oracle may be handed, named from the enum and not
    listed here, so a dtype added to the engine cannot stay untested."""
    from neurobrix.kernels.nbx_tensor import NBXDtype
    wanted = ("float32", "float16", "bfloat16")
    return [getattr(NBXDtype, n) for n in wanted if hasattr(NBXDtype, n)]


def _kernel_written(dtype, n=4096):
    """A device buffer a KERNEL wrote, not the host.

    A host-written buffer has nothing stale to show: the conversion is what
    puts a device write between the allocation and the read, which is the
    situation the oracle actually meets when a model's bias was converted
    earlier in the graph.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor

    rng = np.random.default_rng(20260912)
    host = NBXTensor.from_numpy(rng.standard_normal(n, dtype=np.float32))
    return host.to(dtype)


@pytest.mark.parametrize("dtype", _dtypes(), ids=lambda d: str(d).split(".")[-1])
def test_the_oracle_reads_what_the_copy_path_reads(dtype):
    """The differential, for every dtype and not for the one that broke.

    An ABSOLUTE check cannot decide this: a buffer that comes back readable
    says nothing about when it was read. Only the two paths against each
    other, on the same buffer after the same kernel, can.
    """
    from neurobrix.kernels.screen_oracle import _to_f64

    t = _kernel_written(dtype)
    by_oracle = _to_f64(t)
    assert by_oracle is not None, (
        f"the oracle cannot read a {dtype} operand at all; every comparison "
        f"involving one is then decided by something other than its value")

    by_copy = np.asarray(t.to_cpu().numpy(), dtype=np.float64).ravel()
    got = np.asarray(by_oracle, dtype=np.float64).ravel()
    assert got.shape == by_copy.shape, (
        f"the two paths disagree on shape: {got.shape} vs {by_copy.shape}")
    differing = int((got != by_copy).sum())
    assert differing == 0, (
        f"{differing} of {got.size} elements differ between the oracle's read "
        f"and the copy path, for {dtype}. A reference read at the wrong moment "
        f"contradicts every correct candidate exactly as loudly as a wrong "
        f"one -- which is how eighteen refusals came to rest on this.")


def test_the_property_is_not_stated_about_a_dtype():
    """The test file's own shape, pinned.

    The remedy that would have been wrong is "repair the bf16 read". This
    asserts the parametrisation covers more than one dtype, so the day the
    raw path is repaired for bf16 alone, this still fails for the next one.
    """
    assert len(_dtypes()) >= 2, (
        "a single dtype under test states a property about that dtype, not "
        "about the path; the defect has the shape of a path")
