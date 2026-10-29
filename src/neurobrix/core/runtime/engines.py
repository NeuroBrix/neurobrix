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
        raise PyTorchBranchNotInstalledError(torch_missing_message(f"execution mode {mode!r}"))


def torch_missing_message(what: str) -> str:
    """The one message a user without torch reads when `what` (a mode, a command) needs the PyTorch branch: what it
    needs, the install line, where the wheel for their hardware is chosen, and that the Triton branch needs none."""
    return (f"{what} runs on the PyTorch engine, and PyTorch is not installed in this environment ({sys.executable}).\n"
            f"  To use it, install the PyTorch build for your hardware (choose the wheel for your GPU and driver at "
            f"https://pytorch.org/get-started/locally/), then:\n"
            f"      pip install \"neurobrix[pytorch]\"\n"
            f"  Or run the Triton engine, which needs no PyTorch: add --triton (or --triton-sequential).")
