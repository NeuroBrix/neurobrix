"""The NeuroTax engine exists once per branch (RED_LINES line 34): src/neurobrix/pytorch/neurotax and
src/neurobrix/triton/neurotax, twins of one file, the former nbx/neurotax.py moved as it was (the lift, step 1 of line
27). No common copy is left in the container reader.

What this test does if the code is wrong:
1. the two twins differ by one byte (they must be identical until a reason the door checks makes them differ);
2. a NeuroTax copy is left in the common part (nbx/), or the common reader re-exports one;
3. the Triton twin imports Torch or reaches the PyTorch side; the door hands a branch the other branch's twin.
Run: python -m pytest tests/unit/nbx/test_the_two_neurotax_engines_are_identical_twins.py
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_the_twins_are_byte_identical():
    pt = (SRC / "pytorch" / "neurotax" / "__init__.py").read_bytes()
    tr = (SRC / "triton" / "neurotax" / "__init__.py").read_bytes()
    assert pt == tr
    assert sorted(p.name for p in (SRC / "pytorch" / "neurotax").glob("*.py")) == \
        sorted(p.name for p in (SRC / "triton" / "neurotax").glob("*.py"))


def test_no_neurotax_is_left_in_the_common_reader():
    assert not (SRC / "nbx" / "neurotax.py").exists()
    for path in (SRC / "nbx").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "neurotax" not in (node.module or ""), f"{path}: imports {node.module}"


def test_the_triton_twin_imports_only_the_standard_library():
    tree = ast.parse((SRC / "triton" / "neurotax" / "__init__.py").read_text())
    imported = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__", "re", "typing", "dataclasses"}, imported


def test_the_door_hands_each_branch_its_own_twin():
    from neurobrix import branch_door
    branch_door.choose("triton")
    assert branch_door.module("neurotax").__name__ == "neurobrix.triton.neurotax"
    branch_door._CHOSEN[0] = None
    branch_door.choose("compiled")
    assert branch_door.module("neurotax").__name__ == "neurobrix.pytorch.neurotax"
