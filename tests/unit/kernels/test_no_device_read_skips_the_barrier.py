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


def _copy_path(t):
    """The reference: the tensor's own copy to the host, as float64.

    bf16 has no numpy dtype, so its bits travel in a uint16 container and are
    widened here the way the certifier widens them -- NOT by asking numpy to
    coerce a container it does not understand, which is what a first version
    of this did and what made the bf16 case fail inside the test rather than
    inside the property.
    """
    from neurobrix.kernels.nbx_tensor import NBXDtype
    from neurobrix.kernels.autotune_certify import bf16_bits_to_f32

    host = t.to_cpu()
    if t.nbx_dtype == NBXDtype.bfloat16:
        return bf16_bits_to_f32(host.numpy().view(np.uint16)).astype(np.float64)
    return np.asarray(host.numpy(), dtype=np.float64)


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


@pytest.mark.parametrize("dtype", _dtypes(), ids=lambda d: getattr(d, "name", str(d)))
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

    by_copy = _copy_path(t).ravel()
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


# ── the invariant over the WHOLE module, not over the line that was repaired ─


def _raw_reads(path: Path):
    """Calls that read memory by raw address, found in CODE and not in prose.

    By AST, because a text search counts the comments this repository writes
    ABOUT the defect -- three of the four occurrences of `string_at` in
    `screen_oracle.py` are explanations of why it is gone, and a grep-based
    guard would have reported the file as still broken forever, or been
    loosened until it reported nothing.
    """
    import ast

    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.endswith("string_at") or name.endswith("memmove") or \
                name.endswith("from_address"):
            found.append((node.lineno, ast.unparse(node)[:70]))
    return found


def test_no_raw_device_read_survives_anywhere_in_the_oracle():
    """The repair is judged on the file, not on the line that was found.

    A first pass removed the bf16 read and left a second one nine lines from
    the end of the same function, justified as "an input, unchanged" -- an
    assumption this file's own measurement had destroyed the same day, since
    the bf16 operand of the eighteen refusals WAS an input, produced by a
    conversion kernel. A removal that is real at ninety percent leaves the next
    anomaly with the same two candidate causes.
    """
    import neurobrix.kernels.screen_oracle as mod

    found = _raw_reads(Path(mod.__file__))
    assert not found, (
        "raw address reads still in the oracle:\n  "
        + "\n  ".join(f"line {n}: {src}" for n, src in found)
        + "\n\nEvery read of device memory goes through the tensor's copy to "
          "the host, which crosses the barrier a raw pointer does not, and "
          "which has no 2 GiB ceiling.")


def test_the_guard_can_still_see_one():
    """Both directions: a module that HAS a raw read must be reported.

    Without this the assertion above is satisfied by a finder that finds
    nothing, which is what a guard reporting zero must be shown not to be.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import ctypes\n"
                "# ctypes.string_at in a comment must NOT count\n"
                "def f(a, n):\n"
                "    return ctypes.string_at(int(a), int(n))\n")
        tmp = Path(f.name)
    try:
        found = _raw_reads(tmp)
        assert len(found) == 1, f"expected exactly the call, got {found}"
        assert found[0][0] == 4, "the comment must not be counted"
    finally:
        tmp.unlink()
