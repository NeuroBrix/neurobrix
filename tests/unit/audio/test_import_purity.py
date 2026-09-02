"""D-CORE-MODULE-INIT-TORCH (filed 2026-09-02, fixed the same day).

The triton STT path imports the shared numpy DSP
(`core/module/audio/mel_dsp`, `stt_longform`) and the CPU decode-bound
helper (`core/runtime/decode_bound`). Both are torch-free, but their
PACKAGE inits imported the scheduler / executor stacks — torch reached
the triton path through an init the static R33 grep cannot see. The
inits now export their names lazily (PEP 562). Pin: importing those
modules in a fresh interpreter leaves torch out of sys.modules; the
lazy names still resolve.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/audio/test_import_purity.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[3] / "src")


def _run(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"}, timeout=120).stdout.strip()


def test_shared_numpy_audio_modules_import_without_torch():
    out = _run(
        "import sys\n"
        "import neurobrix.core.module.audio.stt_longform\n"
        "import neurobrix.core.module.audio.mel_dsp\n"
        "print('torch' in sys.modules)")
    assert out == "False", f"torch reached the shared numpy DSP import chain: {out!r}"


def test_decode_bound_imports_without_the_runtime_stack():
    out = _run(
        "import sys\n"
        "from neurobrix.core.runtime.decode_bound import decode_bound\n"
        "print('torch' in sys.modules, decode_bound(5))")
    assert out.startswith("False "), f"torch reached the decode_bound import chain: {out!r}"


def test_lazy_names_still_resolve():
    out = _run(
        "from neurobrix.core.module import SchedulerFactory, AutoregressiveFactory\n"
        "from neurobrix.core.runtime import RuntimeExecutor, ShapeResolutionError, GraphExecutor\n"
        "import neurobrix.core.runtime as r\n"
        "try:\n"
        "    r.NoSuchName\n"
        "except AttributeError:\n"
        "    print('ok', SchedulerFactory.__name__, RuntimeExecutor.__name__)")
    assert out == "ok SchedulerFactory RuntimeExecutor", out
