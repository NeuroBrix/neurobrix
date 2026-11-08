"""The derived census is cardless by construction: sizing its workers reads the cores from the profile it was GIVEN and
never runs the machine's detection, which opens a driver context on every visible card.

2026-11-08 16:37: two Qwen3-30B tables (`derived_census.py table --hardware default-...`) held 306 MiB on each of the
rack's four cards for over an hour — `_machine_cores` called `load_default_profile()` — and the device suite refused
every card as busy while all four sat at 0 %.

What this test would do if the code were wrong: the detection is called (red at the first test, the patched detection
raises), or the sizing process maps the CUDA driver (red at the second, on a Linux machine with the driver installed).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import derived_census as DC  # noqa: E402

#: a committed profile that declares its CPU
PROFILE = "v100-32g"


def test_the_workers_cores_come_from_the_given_profile_never_from_detection(monkeypatch):
    from neurobrix.core.prism import autodetect as AD
    from neurobrix.core.prism.loader import load_profile

    def detection(*_a, **_k):
        raise AssertionError("the census ran the machine's detection")
    monkeypatch.setattr(AD, "load_default_profile", detection)
    monkeypatch.setattr(AD, "get_or_create_default_profile", detection)
    monkeypatch.setattr(AD, "detect_hardware", detection)
    assert DC._machine_cores(PROFILE) == int(load_profile(PROFILE).cpu.compute_threads)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/maps")
def test_sizing_the_workers_maps_no_cuda_driver():
    code = ("import sys; sys.path.insert(0, 'tools'); import derived_census as DC\n"
            f"DC._workers_memory_holds(4, 1 << 20, {PROFILE!r})\n"
            "print(any('libcuda.so' in l for l in open('/proc/self/maps')))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300,
                         env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False", "sizing the census workers loaded the CUDA driver"
