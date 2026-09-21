"""The regression harness spawns the engine under NEUROBRIX_PYTHON. It used to fill
that from $VIRTUAL_ENV in silence and put a battery on the wrong stack twice
(register 78; 2026-09-21). What this test would do if the code were wrong: the
collection below succeeds without the variable (that is what it did before)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _collect(env_extra: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("NEUROBRIX_PYTHON",)}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-m", "pytest", "tests/regression/test_all_models.py",
                           "--collect-only", "-q", "-p", "no:cacheprovider"],
                          cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)


def test_without_neurobrix_python_the_harness_refuses_at_entry():
    r = _collect({"VIRTUAL_ENV": sys.prefix})   # the very guess it used to take
    assert r.returncode != 0
    assert "NEUROBRIX_PYTHON is not set" in r.stdout + r.stderr, (r.stdout + r.stderr)[-1500:]


def test_with_neurobrix_python_named_the_harness_collects():
    r = _collect({"NEUROBRIX_PYTHON": sys.executable})
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
