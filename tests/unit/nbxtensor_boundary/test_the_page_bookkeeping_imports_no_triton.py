"""The page bookkeeping the engine's two caches share is pure bookkeeping (the coordinator's condition, 2026-11-08, for
the compiled torch cache to import it from nbxtensor): `nbxtensor.memory.pages` and `nbxtensor.memory.kv_session`
load no triton and no torch, in a fresh interpreter.

What this test does if the code is wrong: a triton or torch import at the top of either module (or of a package
`__init__` on their path) puts it in `sys.modules` — red (seen red on an `import triton` injected into pages.py)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("module", ["nbxtensor.memory.pages", "nbxtensor.memory.kv_session"])
def test_the_bookkeeping_loads_no_triton_and_no_torch(module):
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}")
    code = (f"import sys, json; import {module}; "
            "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('triton', 'torch'))))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [], f"{module} loads {out.stdout.strip()}"
