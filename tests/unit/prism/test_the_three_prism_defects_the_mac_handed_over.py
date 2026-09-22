"""Three defects in `core/prism/solver.py` the Mac found and could not fix (`9675411a`).

`core/prism` is this rack's. The Mac read them in the census shadow — no card, no weights —
and wrote them up rather than reaching into another machine's subtree. Each is closed here
with the injection that turns its cell red.

1. **The global `_try_zero3` had no unified-device guard**, where the component path has
   carried one since 2026-09-09. On a unified device host and device are the same bytes, so
   an "offload" frees nothing the budget has not already counted — and selecting zero3 there
   hands the executor a plan accepted under one memory model and run under another. It then
   dies inside zero3's CUDA machinery before a single op runs.

2. **`_host_budget_mb` never consulted `NBX_PRISM_BUDGET_MB`**, the door the DEVICE reading
   consults. A census enumerating rungs therefore imposed its rung on every card and left the
   HOST budget at the machine's real free RAM — so `layer_streaming`, whose whole reason to
   exist is the small rungs, was never censused at the rungs where it must fire.

3. **Three comments claimed a `0.7 × ram_mb` budget the code no longer computes.** This is the
   prose half of the vacuous-gate rule: a sentence stating a FORMULA is an assertion, and
   nothing re-checks it when the formula moves. The Mac read its own measurement off one of
   them, which is the cost being paid for here.

Shapes: 8192 MB is a real rung of the ladder and 200 000 MB of host RAM sits between the 192
and 256 GB rungs — chosen so the door's answer (8192) cannot be confused with the undoored
answer (196 608), which is what makes the cell able to fail.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from neurobrix.core.prism.solver import PrismSolver, _device_is_unified

SOLVER_SRC = Path(__file__).resolve().parents[3] / "src/neurobrix/core/prism/solver.py"


# ─────────────────────────── 1. the unified guard on zero3 ───────────────────────────

class _Dev:
    def __init__(self, s):
        self.device_string = s
        self.capacity_mb = 24576.0


class _Solver(PrismSolver):
    def __init__(self):            # no __init__ chain: this exercises one method
        pass

    def _fresh_devices(self, devices):
        return list(devices)


def _profile(unified: bool):
    from neurobrix.core.prism.loader import load_profile
    p = load_profile("default")
    p.devices[0].unified_memory = unified
    return p


def _zero3(device_string: str, unified: bool):
    s = _Solver()
    return s._try_zero3([], {}, [_Dev(device_string)], {}, _profile(unified), None)


def test_zero3_is_refused_on_a_unified_device():
    """The harmful STATE: a zero3 plan selected where offload frees nothing."""
    assert _profile(True).devices[0].unified_memory is True     # the scene is what it claims
    assert _device_is_unified("mps:0", _profile(True)) is True  # and the seam agrees
    assert _zero3("mps:0", unified=True) is None


def test_zero3_still_reaches_its_own_body_on_a_discrete_card():
    """The guard must not refuse everything — that would pass the cell above for free."""
    assert _device_is_unified("cuda:0", _profile(False)) is False
    # It gets past the guard; with no components it returns an empty allocation, not None.
    out = _zero3("cuda:0", unified=False)
    assert out is not None, "the discrete path was refused too — the guard is too wide"


def test_the_guard_sits_in_try_zero3_and_not_only_in_the_component_path():
    """Pins WHERE the fix is. Deleting the line from _try_zero3 must fail a cell."""
    src = SOLVER_SRC.read_text()
    body = src[src.index("    def _try_zero3("):]
    body = body[:body.index("\n    def ", 10)]
    assert "_device_is_unified(" in body, "_try_zero3 carries no unified-device guard"


# ─────────────────────────── 2. the host budget's door ───────────────────────────

class _Cpu:
    ram_mb = 257000.0


class _P:
    cpu = _Cpu()


def test_the_host_budget_honours_the_prism_budget_door(monkeypatch):
    s = _Solver()
    undoored = s._host_budget_mb(_P())
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "8192")
    doored = s._host_budget_mb(_P())
    assert doored == 8192.0, f"the door was ignored: host budget stayed {doored}"
    assert undoored != doored, (
        "the cell cannot fail: the undoored budget already equals the door's value on this "
        f"machine ({undoored}). Pick a rung this host does not land on."
    )


def test_without_the_door_the_host_budget_is_still_a_rung(monkeypatch):
    from neurobrix.core.prism.memory_budget import memory_ladder_mb
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    assert _Solver()._host_budget_mb(_P()) in set(memory_ladder_mb()) | {0.0}


# ─────────────────────────── 3. the prose nothing checked ───────────────────────────

def test_no_comment_in_the_solver_claims_a_fraction_of_installed_ram():
    """A comment stating a FORMULA is an assertion, and this one outlived its code.

    `2585: raw_scale ** 0.7` is an EXPONENT in a resolution scale, not a RAM fraction, and is
    deliberately not matched: the pattern requires 0.7 to multiply a memory quantity.
    """
    offenders = []
    for n, line in enumerate(SOLVER_SRC.read_text().splitlines(), 1):
        if not line.lstrip().startswith("#") and '"""' not in line and not line.lstrip().startswith("Activation budget"):
            stripped = line.split("#", 1)[1] if "#" in line else ""
        else:
            stripped = line
        if re.search(r"(0\.7\s*[*x×]\s*(ram|cpu\.ram|ram_mb)|(ram_mb|RAM)\s*\*\s*0\.7|70\s*%\s*of\s*RAM)",
                     stripped, re.I):
            offenders.append(f"{n}: {line.strip()[:100]}")
    assert not offenders, (
        "a comment still claims a 0.7 fraction of installed RAM as the host budget, which the "
        "code does not compute — it uses _host_budget_mb (the free reading on the ladder):\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("line", [
    "# Use 0.7 x ram_mb (matches the old formula)",
    "# weights must fit in 70% of RAM",
    "budget = cpu.ram_mb * 0.7   # reserve 30%",
])
def test_the_prose_gate_would_catch_each_form_it_is_meant_to_catch(line):
    """The gate above is only worth something if it matches the sentences that existed."""
    assert re.search(r"(0\.7\s*[*x×]\s*(ram|cpu\.ram|ram_mb)|(ram_mb|RAM)\s*\*\s*0\.7|70\s*%\s*of\s*RAM)",
                     line, re.I), line
