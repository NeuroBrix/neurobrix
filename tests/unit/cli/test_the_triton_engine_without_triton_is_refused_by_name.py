"""A Triton-branch mode asked for where triton is not installed is refused by name, on `run` and on `serve`, before
anything is loaded: the install line where neurobrix asks for triton (the platforms PyPI ships a triton wheel for), or
that PyPI publishes none for this platform and the PyTorch engine runs here (native Windows: issue 6).

Before 2026-11-18 `serve --triton` without triton was not refused (serve.py asked `require_engine`, which let every
Triton mode through), and `run --triton` on a GPU host passed both backend gates (Metal: an Apple GPU only; triton-cpu:
a CPU-only profile only): both went on to a ModuleNotFoundError deep inside the run.

"Not installed" is made in a child process as the import system reads it: `sys.modules["triton"] = None` makes
`importlib.util.find_spec("triton")` answer None and every `import triton` raise. What this test would do if the code
were wrong: the container load (run) or the daemon (serve) reached (red); no refusal naming the platform (red)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.child_env import child_env

REPO = Path(__file__).resolve().parents[3]

#: triton made absent on a GPU host (the triton-cpu gate answers for a CPU-only profile, not this one); the model
#: lookup, the container load and the daemon replaced by sentinels: the refusal must come before them
_DRIVER = r"""
import sys
sys.modules["triton"] = None
from nbxtensor import terminal
from nbxtensor.backend import cpu_backend
cpu_backend._is_cpu_only_profile = lambda: False
from neurobrix.cli import LOGGERS, create_parser
terminal.configure(*terminal.resolve(0), roots=LOGGERS)
import neurobrix.nbx as N
class _Loaded(Exception):
    pass
def _load(*a, **k):
    raise _Loaded("the container load was reached")
N.NBXContainer.load = staticmethod(_load)
cmd = sys.argv[1]
if cmd == "run":
    from neurobrix.cli.commands import run as R
    R.find_model = lambda name: "/nonexistent/" + name
    args = create_parser().parse_args(["run", "--model", "M", "--prompt", "p", *sys.argv[2:]])
    call = lambda: R.cmd_run(args)
else:
    import neurobrix.cli.utils as U
    U.resolve_model_arg = lambda m: m
    from neurobrix.serving import client, server
    client.DaemonClient.is_running = staticmethod(lambda: False)
    server.ServingDaemon.__init__ = _load
    from neurobrix.cli.commands import serve as S
    args = create_parser().parse_args(["serve", "--model", "M", *sys.argv[2:]])
    call = lambda: S.cmd_serve(args)
try:
    rc = call()
except _Loaded as exc:
    print("REACHED", exc)
    rc = 7
except SystemExit as exc:
    rc = exc.code
print("RC", rc)
"""


def _child(*argv):
    env = child_env(PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}", PYTHONNOUSERSITE="1")
    out = subprocess.run([sys.executable, "-c", _DRIVER, *argv], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=300)
    return out.stdout + out.stderr


@pytest.mark.parametrize("command", ["run", "serve"])
@pytest.mark.parametrize("flag", ["--triton", "--triton-sequential"])
def test_a_triton_mode_without_triton_is_refused_before_anything_loads(command, flag):
    out = _child(command, flag)
    assert "REACHED" not in out, out[-3000:]
    assert "RC 1" in out, out[-3000:]
    assert f"execution mode '{flag[2:].replace('-', '_')}' runs on the Triton engine" in out, out[-3000:]


def _triton_line():
    text = (REPO / "pyproject.toml").read_text()
    found = [m for m in re.findall(r'^\s*"(triton[^"]*)",', text, re.M)]
    assert len(found) == 1, found
    return found[0]


@pytest.mark.parametrize("platform_, machine, expect", [
    ("win32", "AMD64", "PyPI publishes no triton for win32 AMD64"),
    ("darwin", "x86_64", "PyPI publishes no triton for darwin x86_64"),
    ("linux", "ppc64le", "PyPI publishes no triton for linux ppc64le"),
    ("linux", "x86_64", 'neurobrix asks for it on linux x86_64: pip install "triton"'),
    ("linux", "aarch64", 'neurobrix asks for it on linux aarch64: pip install "triton"'),
])
def test_the_refusal_says_where_a_triton_comes_from_or_that_none_exists(monkeypatch, platform_, machine, expect):
    """The message reads neurobrix's own triton requirement (its marker, pyproject.toml) on the platform simulated."""
    import importlib.metadata
    import platform
    from neurobrix.core.runtime import engines
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: [_triton_line(), "numpy>=1.24.0"])
    monkeypatch.setattr(sys, "platform", platform_)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    text = engines.triton_missing_message("execution mode 'triton'")
    assert expect in text, text
    if "PyPI publishes no triton" in text:
        assert 'pip install "neurobrix[pytorch]"' in text and "--compiled" in text, text


@pytest.mark.parametrize("torch_installed", [True, False])
def test_where_neurobrix_asks_for_triton_the_pytorch_engine_is_offered_by_its_flag_only_when_torch_is_installed(
        monkeypatch, torch_installed):
    """Linux x86_64 (neurobrix asks for triton there), triton absent: with torch installed the refusal offers
    --compiled; with torch absent too it names the PyTorch install, never --compiled alone (that flag would meet the
    torch refusal next). Decided by what is installed (the import finder), never by the platform. Before 2026-11-18 the
    torch-absent text offered "--compiled (or --sequential)" alone (red)."""
    import importlib.metadata
    import importlib.util
    import platform
    from neurobrix.core.runtime import engines
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: [_triton_line(), "numpy>=1.24.0"])
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "torch" and not torch_installed else real(name, *a))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    text = engines.triton_missing_message("execution mode 'triton'")
    assert 'neurobrix asks for it on linux x86_64: pip install "triton"' in text, text
    if torch_installed:
        assert text.endswith("Or run the PyTorch engine, which needs no triton: --compiled (or --sequential)."), text
    else:
        assert "needs no triton: --compiled" not in text, text
        assert 'pip install "neurobrix[pytorch]", and run with --compiled (or --sequential).' in text, text
        assert "https://pytorch.org/get-started/locally/" in text, text
