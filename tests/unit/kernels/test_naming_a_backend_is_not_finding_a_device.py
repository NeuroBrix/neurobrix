"""`_detect_gpu_backend()` names a backend; it does not find a device.

The two questions read alike and are not the same, and the difference cost five tests on
2026-09-16: `test_autotune_correctness_screen.py` gated on `_detect_gpu_backend() is not None`,
got "cuda" on a host with no visible card — the vendor runtime LIBRARY loads there — and its
tests then failed at their first allocation with `cudaErrorNoDevice` instead of skipping.

On Metal the two coincide, because there is no library to dlopen and the probe opens the
device itself (`metal_device_available`: "it opens the real device rather than checking for
the import"). So the SAME call is an executing probe on one backend and a naming one on the
others, which is exactly the asymmetry a caller cannot see from the call site.

This pins it where it can be pinned: a child process with no device visible, on a host that
has one. `DeviceAllocator.device_count()` asks the driver and answers 0; the backend probe
still answers a name.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

_ASK_BOTH = """
import sys
sys.path.insert(0, %r)
from neurobrix.kernels.nbx_tensor import _detect_gpu_backend, DeviceAllocator
try:
    name = _detect_gpu_backend()
except Exception as exc:
    name = "raised:" + type(exc).__name__
print(name, DeviceAllocator.device_count())
""" % str(SRC)


def _ask(visible: str | None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    if visible is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible
    out = subprocess.run([sys.executable, "-c", _ASK_BOTH], capture_output=True, text=True,
                         env=env, timeout=180)
    if out.returncode != 0:
        pytest.skip(f"the probe could not run here: {out.stderr.strip()[:200]}")
    name, count = out.stdout.strip().split()
    return name, int(count)


def test_the_machine_has_a_device_at_all():
    """The premise. Without it the rest is a test about nothing."""
    name, count = _ask(None)
    if name.startswith("raised") or count == 0:
        pytest.skip("no GPU visible to this host — the asymmetry cannot be shown from here")
    assert count > 0


def test_naming_a_backend_survives_what_finding_a_device_does_not():
    """The door that makes the two answers separable: no device visible, runtime still loadable."""
    have, _ = _ask(None)
    if have.startswith("raised"):
        pytest.skip("no GPU visible to this host")
    name, count = _ask("")
    assert count == 0, "with no device visible the driver must report none"
    assert not name.startswith("raised"), (
        "the backend probe answered by raising, which would make it a device probe after all "
        "— if this ever becomes true the docstring and the callers must be revisited")
    assert name == have, (
        f"the backend probe must still NAME the backend with no device visible "
        f"(got {name!r} against {have!r}); that it does is the whole point — it reads the "
        f"install, not the machine")
