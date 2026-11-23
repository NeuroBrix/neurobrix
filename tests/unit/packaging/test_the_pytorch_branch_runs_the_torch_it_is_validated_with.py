"""The PyTorch branch runs the torch it is validated with, 2.14.0 exactly (owner's order, row 275): pyproject's `pytorch`
extra pins the one module constant, an older or a newer torch is refused by name, a CUDA build of the release passes,
and the Triton side never reaches the check."""
import importlib
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:            # python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]


def _tv():
    return importlib.import_module("neurobrix.pytorch.torch_version")


def _torch(monkeypatch, version):
    import torch
    monkeypatch.setattr(torch, "__version__", version)


def test_the_extra_pins_exactly_the_constant_and_nothing_else():
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["pytorch"]
    assert extra == [f"torch=={_tv().VALIDATED_TORCH}"], extra


@pytest.mark.parametrize("version", ["2.12.0", "2.13.1", "2.15.0", "2.14.1", "2.1.0+cu126"])
def test_another_torch_is_refused_by_name_with_the_validated_version(version, monkeypatch):
    _torch(monkeypatch, version)
    with pytest.raises(RuntimeError) as err:
        _tv().require_validated_torch()
    msg = str(err.value)
    assert "ZERO FALLBACK" in msg and "validated with torch 2.14.0" in msg and f"torch {version}" in msg
    assert "neurobrix[pytorch]" in msg


@pytest.mark.parametrize("version", ["2.14.0", "2.14.0+cu126", "2.14.0+cpu"])
def test_the_validated_release_passes_with_or_without_a_local_label(version, monkeypatch):
    _torch(monkeypatch, version)
    _tv().require_validated_torch()


def test_the_check_sits_at_the_branch_choice_and_the_triton_side_never_names_it():
    """qa 1301-1: at the executor it came AFTER the Prism solve; a first launch's reload probe loaded weights under an
    unvalidated torch before the refusal. The choice is the first thing every entry does (run, engine_choice, serve)."""
    door = (ROOT / "src/neurobrix/branch_door.py").read_text()
    assert 'module("torch_version").require_validated_torch()' in door
    assert "require_validated_torch" not in (ROOT / "src/neurobrix/core/runtime/executor.py").read_text()
    triton = ROOT / "src/neurobrix/triton"
    assert not [p for p in triton.rglob("*.py") if "torch_version" in p.read_text() or "VALIDATED_TORCH" in p.read_text()]


def test_every_entry_chooses_the_branch_before_it_reads_any_prism():
    """The run's launch chooses the branch before it reads the hardware through that branch's Prism (autodetect, which
    measures at a first launch); the serve and the engine choice choose at their start."""
    run = (ROOT / "src/neurobrix/cli/commands/run.py").read_text()
    assert run.index("_branch_door.choose(mode)") < run.index('autodetect, intel_refusal = _branch_door.names("prism"')
    serve = (ROOT / "src/neurobrix/serving/engine.py").read_text()
    assert serve.index("_branch_door.choose(mode)") < serve.index("prism")


def test_the_choice_refuses_another_torch_on_the_pytorch_side_and_chooses_nothing(monkeypatch):
    import torch
    from neurobrix import branch_door
    monkeypatch.setattr(branch_door, "_CHOSEN", [None])
    monkeypatch.setattr(torch, "__version__", "2.15.0")
    with pytest.raises(RuntimeError, match="validated with torch 2.14.0"):
        branch_door.choose("compiled")
    assert branch_door._CHOSEN[0] is None
    assert branch_door.choose("triton") == "triton"          # the Triton side never asks it


def test_the_validated_torch_is_chosen_once(monkeypatch):
    import torch
    from neurobrix import branch_door
    monkeypatch.setattr(branch_door, "_CHOSEN", [None])
    monkeypatch.setattr(torch, "__version__", "2.14.0+cu126")
    assert branch_door.choose("compiled") == "pytorch" and branch_door.choose("sequential") == "pytorch"
