"""The PyTorch branch runs the torch it is validated with, 2.14.0 exactly (owner's order, row 275): pyproject's `pytorch`
extra pins the one module constant, an older or a newer torch is refused by name, a CUDA build of the release passes,
and the Triton side never reaches the check."""
import importlib
import types
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:            # python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]


def _tv():
    return importlib.import_module("neurobrix.pytorch.torch_version")


def _torch(version):
    return types.SimpleNamespace(__version__=version)


def test_the_extra_pins_exactly_the_constant_and_nothing_else():
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["pytorch"]
    assert extra == [f"torch=={_tv().VALIDATED_TORCH}"], extra


@pytest.mark.parametrize("version", ["2.12.0", "2.13.1", "2.15.0", "2.14.1", "2.1.0+cu126"])
def test_another_torch_is_refused_by_name_with_the_validated_version(version):
    with pytest.raises(RuntimeError) as err:
        _tv().require_validated_torch(_torch(version))
    msg = str(err.value)
    assert "ZERO FALLBACK" in msg and "validated with torch 2.14.0" in msg and f"torch {version}" in msg
    assert "neurobrix[pytorch]" in msg


@pytest.mark.parametrize("version", ["2.14.0", "2.14.0+cu126", "2.14.0+cpu"])
def test_the_validated_release_passes_with_or_without_a_local_label(version):
    _tv().require_validated_torch(_torch(version))


def test_the_executor_asks_it_for_the_pytorch_side_only():
    src = (ROOT / "src/neurobrix/core/runtime/executor.py").read_text()
    assert 'if _branch_door.choose(mode) == "pytorch":' in src
    assert 'module("torch_version").require_validated_torch()' in src
    triton = ROOT / "src/neurobrix/triton"
    assert not [p for p in triton.rglob("*.py") if "torch_version" in p.read_text() or "VALIDATED_TORCH" in p.read_text()]


def test_the_executor_refuses_another_torch_on_the_pytorch_side_and_never_on_the_triton_side(monkeypatch):
    import torch
    from neurobrix import branch_door
    from neurobrix.core.runtime.executor import RuntimeExecutor
    monkeypatch.setattr(branch_door, "_CHOSEN", [None])
    monkeypatch.setattr(torch, "__version__", "2.15.0")
    plan = types.SimpleNamespace(components={})
    with pytest.raises(RuntimeError, match="validated with torch 2.14.0"):
        RuntimeExecutor(runtime_package=None, execution_plan=plan, mode="compiled")
    monkeypatch.setattr(branch_door, "_CHOSEN", [None])
    try:
        RuntimeExecutor(runtime_package=None, execution_plan=plan, mode="triton")
    except RuntimeError as exc:
        assert "validated with torch" not in str(exc)
    except Exception:
        pass
