"""The shared files ask the NeuroTax twin of the branch in force AT THE CALL, never at import (qa 1286-2): the branch
is chosen after the imports, so a module-level ask bound whichever twin the import order happened to give.

What it would do if the claim were wrong: importing core/flow/shared_steps.py or core/runtime/graph/moe_fusion.py
asks the door (red: the ask below raises at import); a call after the branch is chosen reads another branch's twin (red:
the twin the call used is recorded and compared).

moe_fusion.py asks no twin at all since red line 28 (it is self-contained, its tokens given by the caller): it stays
in the import check; the call check is its shared callers' door, `neurotax.moe_tokens` asked at the call."""
import importlib
import subprocess
import sys

import pytest

from neurobrix import branch_door

SHARED = ("neurobrix.core.flow.shared_steps", "neurobrix.core.runtime.graph.moe_fusion")

_IMPORT_ASKING_NOTHING = """
import sys
from neurobrix import branch_door
def refuse(*a):
    raise AssertionError(f"asked the door at import: {a}")
branch_door.names = branch_door.module = refuse
import importlib
importlib.import_module(sys.argv[1])
"""


@pytest.mark.parametrize("name", SHARED)
def test_importing_a_shared_file_asks_no_twin(name):
    """In a fresh interpreter (this one has imported them already), with the door refusing every ask."""
    done = subprocess.run([sys.executable, "-c", _IMPORT_ASKING_NOTHING, name], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("side", ["pytorch", "triton"])
def test_a_call_reads_the_twin_of_the_branch_in_force(side, monkeypatch):
    shared_steps, moe_fusion = (importlib.import_module(name) for name in SHARED)
    asked = []
    real = branch_door.names

    def spy(brick, sub, *members):
        asked.append((branch_door.side(), brick))
        return real(brick, sub, *members)
    monkeypatch.setattr(branch_door, "_CHOSEN", [side])
    monkeypatch.setattr(branch_door, "names", spy)
    assert shared_steps._joint_keys() and moe_fusion._Tokens.of(branch_door.names("neurotax", "", "moe_tokens")[0]())
    assert asked and set(asked) == {(side, "neurotax")}, asked
