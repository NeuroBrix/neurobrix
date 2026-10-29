"""A traced graph's ATen aliases are the engine's knowledge, not the library's (Hocine 2026-10-29 16:28: nbxtensor holds
nothing of NeuroBrix; `canonical_aten` returns to the engine). The library's `dispatch` maps its own op names as given
and asks the application for nothing; the engine resolves an alias first (`neurobrix.kernels.classification.kernel_for`).

What this test would do if the code were wrong: a library `dispatch` that still needs the host's `canonical_aten` raises
"not provided" in a process where NeuroBrix provided nothing (red); a library that resolves `multiply` itself carries
the engine's alias table (red); an engine lookup of `aten::multiply` that misses `mul` (red); the service still listed
(red)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.unit.child_env import child_env

REPO = Path(__file__).resolve().parents[3]

_LIBRARY_ALONE = r"""
import sys
from nbxtensor.ops.dispatch import dispatch
assert "neurobrix" not in sys.modules, sorted(m for m in sys.modules if m.startswith("neurobrix"))
mul = dispatch("mul")
print("MUL", mul is not None, "ATEN_MUL", dispatch("aten::mul") is mul, "ALIAS", dispatch("aten::multiply"))
"""


def test_the_library_dispatches_its_own_names_with_nothing_provided():
    out = subprocess.run([sys.executable, "-c", _LIBRARY_ALONE], cwd=REPO, capture_output=True, text=True, timeout=300,
                         env=child_env(PYTHONPATH=str(REPO / "nbxtensor" / "src"), PYTHONNOUSERSITE="1",
                                       CUDA_VISIBLE_DEVICES=""))
    assert "MUL True ATEN_MUL True ALIAS None" in out.stdout, out.stdout[-2000:] + out.stderr[-3000:]


def test_the_engine_resolves_an_alias_before_it_asks():
    from nbxtensor.ops.dispatch import dispatch
    from neurobrix.kernels.classification import kernel_for
    assert kernel_for("aten::multiply") is dispatch("mul") is not None
    assert kernel_for("aten::greater_equal") is dispatch("ge") is not None
    assert kernel_for("aten::mul") is dispatch("mul")


def test_the_host_lists_no_alias_service():
    from nbxtensor import host
    assert "canonical_aten" not in host.SERVICES
