"""Prism is doubled (owner 23/11/2026, orders row 248; RED_LINES.md lines 16-22): the PyTorch branch's Prism is
`neurobrix.pytorch.prism` (the former core/prism, as it was), the Triton branch's is `neurobrix.triton.prism`, its twin
with every Torch line removed; code common to both branches reaches the one of the branch it runs through the door
(`neurobrix.branch_door`).

What this test does if the code is wrong:
1. a module of one Prism has no twin in the other, or the twin lacks a function or class of its original (the only
   members the Triton twin drops are the compiled engine's own measure, `facts_probe._compiled_loader`);
2. a module of the Triton Prism, imported in a process where torch and the PyTorch directory cannot be imported,
   fails or loads either;
3. the door hands a branch the other branch's Prism, or lets one process choose both branches.

Run: python -m pytest tests/unit/prism/test_the_triton_prism_is_the_pytorch_prism_without_torch.py
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
TORCH_PRISM, TRITON_PRISM = SRC / "pytorch" / "prism", SRC / "triton" / "prism"
#: what the Triton twin drops: the compiled engine's own measures, which the PyTorch Prism keeps
DROPPED = {"facts_probe.py": {"_compiled_loader"}}


def _modules(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _members(path: Path):
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def test_every_module_has_its_twin_with_the_same_members():
    assert _modules(TORCH_PRISM) == _modules(TRITON_PRISM)
    for rel in _modules(TORCH_PRISM):
        torch_side, triton_side = _members(TORCH_PRISM / rel), _members(TRITON_PRISM / rel)
        assert torch_side - triton_side == DROPPED.get(rel, set()), rel
        assert triton_side - torch_side == set(), rel


_CHILD = r"""
import importlib, pkgutil, sys
class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch.") or name == "neurobrix.pytorch" or name.startswith("neurobrix.pytorch."):
            raise ImportError("refused: " + name)
sys.meta_path.insert(0, Refuse())
import neurobrix.triton.prism as P
for m in pkgutil.walk_packages(P.__path__, "neurobrix.triton.prism."):
    importlib.import_module(m.name)
loaded = sorted(k for k in sys.modules if k == "torch" or k.startswith("torch.") or k.startswith("neurobrix.pytorch"))
print("LOADED", loaded)
"""


def test_the_triton_prism_imports_with_torch_and_the_pytorch_side_absent():
    r = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "LOADED []" in r.stdout, r.stdout[-2000:]


def test_the_door_hands_each_branch_its_own_prism():
    from neurobrix import branch_door
    assert branch_door.side_of("compiled") == branch_door.side_of("sequential") == "pytorch"
    branch_door.choose("compiled")
    assert branch_door.module("prism").__name__ == "neurobrix.pytorch.prism"
    branch_door._CHOSEN[0] = None                       # a new process
    branch_door.choose("triton_sequential")
    assert branch_door.module("prism").__name__ == "neurobrix.triton.prism"
    assert branch_door.side() == "triton"
    solver, = branch_door.names("prism", "solver", "PrismSolver")
    assert solver.__module__ == "neurobrix.triton.prism.solver"
    with pytest.raises(RuntimeError, match="BRANCH ALREADY CHOSEN.*one process runs one branch"):
        branch_door.choose("compiled")
