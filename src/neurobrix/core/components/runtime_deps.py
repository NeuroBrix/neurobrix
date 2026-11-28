"""Generic runtime codec and external component dependency resolver.

Resolves optional, external runtime components (such as ``imageio``,
``imageio_ffmpeg``, or system ``ffmpeg``) at point of use without pulling
Torch or optional external dependencies into the shared engine orchestrator.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from types import ModuleType

_FFMPEG_PIP_SPEC = "imageio-ffmpeg>=0.4.0"
_FFMPEG_PURPOSE = "encoding/decoding video streams (H.264/MP4)"


def _can_prompt(allow_prompt: bool) -> bool:
    """Return True only when interactive confirmation on a real TTY is permitted."""
    return (
        bool(allow_prompt)
        and sys.stdin.isatty()
        and sys.stderr.isatty()
        and not os.environ.get("CI")
        and not os.environ.get("NBX_NONINTERACTIVE")
    )


def _prompt_yes_no(prompt: str) -> bool:
    """Ask a y/N question on the terminal; treat EOF/interrupt as 'no'."""
    try:
        reply = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return reply.strip().lower() in ("y", "yes")


def _missing_dep_error(import_name: str, pip_spec: str, purpose: str) -> RuntimeError:
    """Format an actionable RuntimeError with the exact manual install command."""
    cmd = f"{sys.executable} -m pip install {pip_spec}"
    return RuntimeError(
        f"Missing runtime dependency '{pip_spec}' (module '{import_name}') "
        f"required for {purpose}. Install it into the active Python environment "
        f"with:\n  {cmd}"
    )


def ensure_python_dep(
    import_name: str,
    pip_spec: str,
    purpose: str,
    *,
    allow_prompt: bool = True,
) -> ModuleType:
    """Import an optional Python runtime dependency or interactively install it.

    When ``import_name`` is missing:
    - If ``allow_prompt`` is True on an interactive TTY (and neither ``CI`` nor
      ``NBX_NONINTERACTIVE`` is set), explains why ``pip_spec`` is needed for
      ``purpose`` and prompts to install it via ``pip``.
    - Otherwise (declined, non-interactive, or CI), raises an actionable
      ``RuntimeError`` with the exact ``<sys.executable> -m pip install`` command.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        if not _can_prompt(allow_prompt):
            raise _missing_dep_error(import_name, pip_spec, purpose) from exc

        print(
            f"\n[NeuroBrix] Missing optional runtime dependency '{pip_spec}' "
            f"(module '{import_name}') required for {purpose}.",
            file=sys.stderr,
        )
        prompt = (
            f"Install '{pip_spec}' into the current Python environment "
            f"({sys.executable}) now? [y/N]: "
        )
        if not _prompt_yes_no(prompt):
            raise _missing_dep_error(import_name, pip_spec, purpose) from exc

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])
        except subprocess.CalledProcessError as pip_exc:
            raise RuntimeError(
                f"Failed to install '{pip_spec}' for {purpose} "
                f"(exit code {pip_exc.returncode}). Run manually:\n"
                f"  {sys.executable} -m pip install {pip_spec}"
            ) from pip_exc

        importlib.invalidate_caches()
        return importlib.import_module(import_name)


def resolve_ffmpeg_exe(*, allow_prompt: bool = True) -> str:
    """Resolve an ffmpeg executable via imageio-ffmpeg, system PATH, or pip prompt.

    Resolution order:
    1. Existing ``imageio_ffmpeg.get_ffmpeg_exe()`` if available and non-empty.
    2. System ``ffmpeg`` binary on ``PATH`` via ``shutil.which("ffmpeg")``.
    3. ``ensure_python_dep("imageio_ffmpeg", "imageio-ffmpeg>=0.4.0", ...)``
       and then ``imageio_ffmpeg.get_ffmpeg_exe()``.
    """
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if isinstance(exe, str) and exe.strip():
            return exe
    except (ImportError, RuntimeError, OSError):
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    imageio_ffmpeg = ensure_python_dep(
        "imageio_ffmpeg",
        _FFMPEG_PIP_SPEC,
        _FFMPEG_PURPOSE,
        allow_prompt=allow_prompt,
    )
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not isinstance(exe, str) or not exe.strip():
        raise RuntimeError(
            "imageio_ffmpeg.get_ffmpeg_exe() returned an empty executable path"
        )
    return exe
