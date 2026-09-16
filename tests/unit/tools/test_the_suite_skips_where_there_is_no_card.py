"""The suite-wide hook turns "there is no card" into a skip, and turns nothing else.

Measured 2026-09-16: the unit suite under the workshop's device door
(`CUDA_VISIBLE_DEVICES=`) returned 50 failures, **42 of them one thing** — a test
allocating on a device that is not there, raising `DeviceOOMError: GPU malloc failed
(error 100)`, error 100 being `cudaErrorNoDevice`. A suite that answers fifty reds
meaning "no card here" is a suite nobody runs beside a campaign, which is the only
place the door exists for.

`tests/conftest.py` converts that ONE failure into a skip. This file is what keeps the
conversion from becoming a swallow, and it pins both directions:

* a failure whose text carries a no-device mark becomes a skip — **only** when an
  executing probe says this process can see no device;
* every other failure stays a failure, at the same moment, in the same run;
* on a machine that HAS a device the hook is disarmed entirely, so a real allocation
  failure on a real card is untouched.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REAL_CONFTEST = ROOT / "tests" / "conftest.py"

_TWO_TESTS = '''
def test_it_cannot_reach_a_card():
    raise RuntimeError("GPU malloc failed (error 100) for 64 bytes [driver_total=0MB]")

def test_it_is_simply_wrong():
    assert 2 + 2 == 5, "this is arithmetic, not a missing card"
'''


def _run(tmp_path: Path, visible: str | None):
    (tmp_path / "conftest.py").write_text(REAL_CONFTEST.read_text())
    (tmp_path / "test_two.py").write_text(_TWO_TESTS)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    if visible is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          capture_output=True, text=True, env=env, timeout=300, cwd=str(tmp_path))


def test_with_no_card_visible_only_the_no_card_failure_becomes_a_skip(tmp_path):
    out = _run(tmp_path, "")
    tail = out.stdout.strip().splitlines()[-1]
    assert "1 failed" in tail and "1 skipped" in tail, (
        f"expected exactly one skip (the missing card) and one failure (the arithmetic); got: {tail}")
    assert "this is arithmetic" in out.stdout, "the real failure must still be reported in full"


def test_with_a_card_visible_the_hook_is_disarmed(tmp_path):
    """The half that keeps it from being a swallow: on a real machine, nothing is converted."""
    env2 = dict(os.environ); env2.pop("CUDA_VISIBLE_DEVICES", None); env2["PYTHONPATH"] = str(ROOT / "src")
    have = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "from neurobrix.kernels.nbx_tensor import DeviceAllocator; print(DeviceAllocator.device_count())"
         % str(ROOT / "src")],
        capture_output=True, text=True, timeout=180, env=env2)
    if have.returncode != 0 or int(have.stdout.strip() or 0) == 0:
        pytest.skip("no GPU visible to this host — the disarmed half cannot be shown from here")
    out = _run(tmp_path, None)
    tail = out.stdout.strip().splitlines()[-1]
    assert "2 failed" in tail, (
        f"with a device visible BOTH failures must stand — the hook must not convert "
        f"anything on a real machine; got: {tail}")
