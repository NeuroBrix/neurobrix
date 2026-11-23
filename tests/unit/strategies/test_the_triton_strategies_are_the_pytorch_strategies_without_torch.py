"""Prism's strategies are doubled (owner 23/11/2026, orders row 248; RED_LINES.md lines 16-22): the PyTorch branch's
are `neurobrix.pytorch.strategies` (the former core/strategies, as it was), the Triton branch's are
`neurobrix.triton.strategies`, the twin of every file the Triton side reached on the PyTorch side with every Torch line
removed; code common to both branches reaches the strategies of the branch it runs through the door
(`neurobrix.branch_door`).

What this test does if the code is wrong:
1. a module of the Triton strategies has no original, or lacks a function or class of its original. The only PyTorch
   modules without a twin are the PyTorch engine's own: `pinned_block.py` (its one registered host block, which the
   Triton loader replaces by pinned NBXTensors) and `utils/` (reached by no Triton file). The twins drop no member;
   the members only a twin holds are named below, each with its reason;
2. a module of the Triton strategies, imported in a process where torch and the PyTorch directory cannot be imported,
   fails or loads either;
3. the door hands a branch the other branch's strategies, or one branch's `get_strategy` serves the other's mode.

Run: python -m pytest tests/unit/strategies/test_the_triton_strategies_are_the_pytorch_strategies_without_torch.py
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
TORCH_SIDE, TRITON_SIDE = SRC / "pytorch" / "strategies", SRC / "triton" / "strategies"
#: the PyTorch modules with no Triton twin: the PyTorch engine's own
NO_TWIN = {"pinned_block.py", "utils/__init__.py", "utils/block_detector.py"}
#: what only a Triton module holds, and why
TRITON_ONLY = {
    "base.py": {"TritonStrategy"},                 # the base of every Triton strategy (the former triton/strategies/base.py)
    "block_scatter.py": {"_nbx_device_str"},       # an NBXTensor's device read from `_device_idx`
    "lazy_sequential.py": {"_names_accelerator"},  # a device name read by nbxtensor's accelerator prefixes
    "value_parking.py": {"_refuse_foreign"},       # a tensor that is no NBXTensor is refused by name
    "zero3.py": {"_triton_only"},                  # a ratchet state that is not the Triton engine's is refused by name
}


def _modules(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _members(path: Path):
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def test_every_module_has_its_twin_with_the_same_members():
    assert sorted(set(_modules(TORCH_SIDE)) - NO_TWIN) == _modules(TRITON_SIDE)
    for rel in _modules(TRITON_SIDE):
        torch_side, triton_side = _members(TORCH_SIDE / rel), _members(TRITON_SIDE / rel)
        assert torch_side - triton_side == set(), rel
        assert triton_side - torch_side == TRITON_ONLY.get(rel, set()), rel


def test_the_merged_zero3_keeps_the_resolution_of_the_former_triton_subclass():
    """The former `Zero3Strategy(TritonStrategy, <PyTorch Zero3Strategy>)`: TritonStrategy's transfer and sync won over
    the PyTorch ratchet's, the subclass's `_get_exec_device` over both. The merged class inherits TritonStrategy and
    defines none of TritonStrategy's methods itself."""
    from neurobrix.triton.strategies.base import TritonStrategy
    from neurobrix.triton.strategies.zero3 import Zero3Strategy
    assert Zero3Strategy.__mro__[1] is TritonStrategy
    for name in ("transfer_tensor", "transfer_dict", "_release_device_cache", "synchronize_device"):
        assert name not in Zero3Strategy.__dict__, name
    assert "_get_exec_device" in Zero3Strategy.__dict__


_CHILD = r"""
import importlib, pkgutil, sys
class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch.") or name == "neurobrix.pytorch" or name.startswith("neurobrix.pytorch."):
            raise ImportError("refused: " + name)
sys.meta_path.insert(0, Refuse())
import neurobrix.triton.strategies as S
for m in pkgutil.walk_packages(S.__path__, "neurobrix.triton.strategies."):
    importlib.import_module(m.name)
classes = [S.STRATEGY_REGISTRY[k] for k in S.STRATEGY_REGISTRY]
assert all(c.__module__.startswith("neurobrix.triton.strategies.") for c in classes), classes
loaded = sorted(k for k in sys.modules if k == "torch" or k.startswith("torch.") or k.startswith("neurobrix.pytorch"))
print("LOADED", loaded)
"""


def test_the_triton_strategies_import_with_torch_and_the_pytorch_side_absent():
    r = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "LOADED []" in r.stdout, r.stdout[-2000:]


def test_the_door_hands_each_branch_its_own_strategies(monkeypatch):
    from neurobrix import branch_door
    monkeypatch.setattr(branch_door, "_CHOSEN", [None])
    branch_door.choose("compiled")
    assert branch_door.module("strategies").__name__ == "neurobrix.pytorch.strategies"
    monkeypatch.setattr(branch_door, "_CHOSEN", [None])        # a new process
    branch_door.choose("triton")
    assert branch_door.module("strategies").__name__ == "neurobrix.triton.strategies"
    get_strategy, layer_streaming = branch_door.names("strategies", "", "get_strategy", "layer_streaming")
    assert get_strategy.__module__ == "neurobrix.triton.strategies"
    assert layer_streaming.__name__ == "neurobrix.triton.strategies.layer_streaming"


def test_each_get_strategy_refuses_the_other_branch_s_mode():
    from neurobrix.pytorch import strategies as P
    from neurobrix.triton import strategies as T
    with pytest.raises(RuntimeError, match="mode 'triton' runs the Triton branch"):
        P.get_strategy("single_gpu", P.StrategyContext(strategy_name="single_gpu", allocations={}, mode="triton"))
    with pytest.raises(RuntimeError, match="mode 'compiled' is the PyTorch branch's"):
        T.get_strategy("single_gpu", T.StrategyContext(strategy_name="single_gpu", allocations={}, mode="compiled"))
    assert set(T.STRATEGY_REGISTRY) == set(P.STRATEGY_REGISTRY)
