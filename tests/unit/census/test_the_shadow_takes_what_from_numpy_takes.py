"""The census shadow's `from_numpy` stand-in takes every parameter the real one takes.

`NBXTensor.from_numpy` grew a `device` argument (89a2cbb5c, 2026-11-12); the shadow kept the old signature and every
walked census died at its first host array (TinyLlama on the Mac: "_from_numpy_shadow() got an unexpected keyword
argument 'device'"). The shadow wraps the real function, so the two signatures are compared — the next parameter the
real one grows turns this red before a census does. Run in a subprocess: `census.install` patches the process for good.

What this test does if the code is wrong: it fails when the installed shadow lacks a parameter of the real function
(seen red on main d1d1c2e1d: ['arr', 'dtype'] against ['arr', 'dtype', 'device']).

Run: python -m pytest tests/unit/census/test_the_shadow_takes_what_from_numpy_takes.py
"""
from __future__ import annotations

import os
import subprocess
import sys

PROBE = """
import inspect
from nbxtensor.tensor import NBXTensor
real = list(inspect.signature(NBXTensor.from_numpy).parameters)
from neurobrix.kernels import census
census.install(hardware="default-ff6008b7")
shadow = list(inspect.signature(NBXTensor.from_numpy).parameters)
print(real, shadow)
assert shadow == real, (shadow, real)
"""


def test_the_installed_shadow_has_the_real_signature():
    # the shadow refuses a process that sees a card (`census.install`): the probe runs with none in sight, as a census does
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
