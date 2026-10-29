"""The PyTorch branch asked for where torch is not installed is refused by name, before anything is loaded; the
Triton branch is not asked for torch at all (Hocine, 2026-10-29 15:52: torch an optional dependency of the PyTorch
branch, the Triton engine on nbxtensor + triton + numpy).

"Not installed" is made in a child process the way the import system itself reads it: `sys.modules["torch"] = None`
makes `importlib.util.find_spec("torch")` answer None and every `import torch` raise — what an environment without the
package gives, without building one (the torchless environment of the landing proofs runs the same entry point for
real).

What this test would do if the code were wrong: a run that reaches the container load under --compiled with no torch
(red: the sentinel load is reached), a refusal that does not name the branch and its install line (red), a Triton
mode refused for want of torch (red), base dependencies that still carry torch (red)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.child_env import child_env

REPO = Path(__file__).resolve().parents[3]

#: the entry point's own run command, torch made absent, the model lookup and the container load replaced by sentinels:
#: the refusal must come between them
_DRIVER = r"""
import sys
sys.modules["torch"] = None
from neurobrix.cli import create_parser
from neurobrix.cli.commands import run as R
import neurobrix.nbx as N
R.find_model = lambda name: "/nonexistent/" + name
class _Loaded(Exception):
    pass
def _load(*a, **k):
    raise _Loaded("the container load was reached")
N.NBXContainer.load = staticmethod(_load)
args = create_parser().parse_args(["run", "--model", "M", "--prompt", "p", *sys.argv[1:]])
try:
    rc = R.cmd_run(args)
except _Loaded as exc:
    print("REACHED", exc)
    rc = 7
print("RC", rc)
"""


def _run(*flags):
    env = child_env(PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO / 'nbxtensor' / 'src'}{os.pathsep}{REPO}",
                    PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", _DRIVER, *flags], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=300)
    return out.stdout + out.stderr


@pytest.mark.parametrize("flags", [(), ("--compiled",), ("--sequential",)])
def test_the_pytorch_branch_without_torch_stops_before_the_load(flags):
    text = _run(*flags)
    assert "RC 1" in text, text[-3000:]
    assert "REACHED" not in text, "the run went on to load the container without torch"
    assert "PyTorch engine" in text and 'neurobrix[pytorch]' in text and "--triton" in text, text[-3000:]
    assert "https://pytorch.org/get-started/locally/" in text, "the message does not say where the wheel is chosen"
    assert "Traceback" not in text, text[-3000:]


_MAIN = _DRIVER.replace(
    'args = create_parser().parse_args(["run", "--model", "M", "--prompt", "p", *sys.argv[1:]])\ntry:\n    rc = R.cmd_run(args)',
    'from neurobrix.cli import main\nsys.argv = ["neurobrix", "run", "--model", "M", "--prompt", "p", *sys.argv[1:]]\ntry:\n    rc = main()')


def test_through_the_entry_point_main_the_refusal_is_a_message_and_rc_1():
    assert _MAIN != _DRIVER
    env = child_env(PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO / 'nbxtensor' / 'src'}{os.pathsep}{REPO}",
                    PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", _MAIN, "--compiled"], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=300)
    text = out.stdout + out.stderr
    assert out.returncode == 1 and "REACHED" not in text and "Traceback" not in text, text[-3000:]
    assert "pip install \"neurobrix[pytorch]\"" in text and "pytorch.org" in text, text[-3000:]


def test_the_triton_branch_is_not_asked_for_torch():
    from neurobrix.core.runtime import engines
    for mode in engines.TRITON_MODES:
        engines.require_engine(mode)                    # no torch question asked: returns whatever is installed


def test_without_torch_every_pytorch_mode_is_refused(monkeypatch):
    from neurobrix.core.runtime import engines
    monkeypatch.setitem(sys.modules, "torch", None)
    for mode in engines.PYTORCH_MODES:
        with pytest.raises(engines.PyTorchBranchNotInstalledError, match="neurobrix\\[pytorch\\]"):
            engines.require_engine(mode)
    for mode in engines.TRITON_MODES:
        engines.require_engine(mode)


def test_a_mode_no_engine_answers_is_refused():
    from neurobrix.core.runtime import engines
    with pytest.raises(ValueError, match="none of"):
        engines.require_engine("native")


def test_torch_is_the_pytorch_extra_not_a_base_dependency():
    try:
        import tomllib as T
    except ImportError:                       # Python 3.10
        import tomli as T
    project = T.loads((REPO / "pyproject.toml").read_text())["project"]
    base = [d.split(";")[0].replace(" ", "") for d in project["dependencies"]]
    names = {d.split("@")[0].split(">")[0].split("=")[0].split("<")[0] for d in base}
    assert "torch" not in names, base
    assert {"nbxtensor", "triton", "numpy"} <= names, base
    assert any(d.startswith("torch") for d in project["optional-dependencies"]["pytorch"])
