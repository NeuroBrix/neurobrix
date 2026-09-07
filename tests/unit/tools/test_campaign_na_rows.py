"""A lever row on an engine that refuses the container at its capability gate reads N/A, never FAILED."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def test_a_refused_engine_is_not_applicable():
    r = {"lever": "env:NBX_OPTIM_ALGEBRAIC", "gate": {"kind": "n/a", "reason": "component 'model' stores its weights with encoding 'int4-g128-asym', which the compiled engine does not execute", "pass": None, "ran": False}}
    assert C.verdict(r).startswith("N/A (component 'model' stores its weights with encoding 'int4-g128-asym'")
    assert C.verdict({"lever": "env:X", "gate": {"kind": "bytes", "ran": False}}) == "FAILED (an arm did not run)"
