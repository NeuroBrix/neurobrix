"""The two engines and what each needs installed.

The Triton branch (`triton`, `triton_sequential`) runs on nbxtensor, triton and numpy; torch is not among its
dependencies (R33). The PyTorch branch (`compiled`, `sequential`) is torch itself, an optional dependency
(`pip install "neurobrix[pytorch]"`). A PyTorch-branch mode asked for where torch is not installed is refused here,
by name, before anything is loaded — never a ModuleNotFoundError from deep inside a run, never a silent switch to the
other branch (ZERO FALLBACK).

Whether torch is installed is read from the import system's finder, which locates a package without importing it:
the Triton branch asks the question and stays torch-free.
"""
from __future__ import annotations

import importlib.util
import sys

#: the modes of the PyTorch branch (torch.Tensor, ATen); the others are the Triton branch's
PYTORCH_MODES = ("compiled", "sequential")
#: the modes of the Triton branch (NBXTensor, @triton.jit)
TRITON_MODES = ("triton", "triton_sequential")


class PyTorchBranchNotInstalledError(RuntimeError):
    """A PyTorch-branch mode was asked for and torch is not installed."""


def require_engine(mode: str) -> None:
    """Refuse, by name, a mode no engine answers or a PyTorch-branch mode without torch."""
    if mode in TRITON_MODES:
        return
    if mode not in PYTORCH_MODES:
        raise ValueError(f"execution mode {mode!r} is none of {PYTORCH_MODES + TRITON_MODES}")
    if importlib.util.find_spec("torch") is None:
        raise PyTorchBranchNotInstalledError(
            f"execution mode {mode!r} is the PyTorch branch, and torch is not installed in this environment "
            f"({sys.executable}). "
            f"Install it with: pip install \"neurobrix[pytorch]\" — or run the Triton branch, which needs no torch: "
            f"--triton or --triton-sequential.")
