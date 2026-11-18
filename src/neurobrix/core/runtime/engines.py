"""The two engines and what each needs installed.

The Triton branch (`triton`, `triton_sequential`) runs on nbxtensor, triton and numpy; torch is not among its
dependencies (R33). The PyTorch branch (`compiled`, `sequential`) is torch itself, an optional dependency
(`pip install "neurobrix[pytorch]"`). A PyTorch-branch mode asked for where torch is not installed is refused here,
by name, before anything is loaded — never a ModuleNotFoundError from deep inside a run, never a silent switch to the
other branch (ZERO FALLBACK). With no mode asked for, the run takes the PyTorch engine where torch is installed and the
Triton engine where it is not (`pip install neurobrix`, the light install), and says which on its launch line.

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


class TritonBranchNotInstalledError(RuntimeError):
    """A Triton-branch mode (`what`) was asked for and triton is not installed: the library's own refusal where it has
    one (`said`, an Apple GPU's), else where a triton comes from on this platform (`triton_missing_message`)."""

    def __init__(self, what: str, said: str = ""):
        super().__init__(said or triton_missing_message(what))


def require_engine(mode: str) -> None:
    """Refuse, by name, a mode no engine answers, or a mode whose branch (torch, triton) is not installed."""
    if mode in TRITON_MODES:
        if importlib.util.find_spec("triton") is None:
            from nbxtensor.backend.metal_backend import TritonMetalNotInstalledError, ensure_triton_metal_or_raise
            try:
                ensure_triton_metal_or_raise()      # an Apple GPU: the library's refusal names its Metal triton
            except TritonMetalNotInstalledError as exc:
                raise TritonBranchNotInstalledError(f"execution mode {mode!r}", str(exc)) from None
            raise TritonBranchNotInstalledError(f"execution mode {mode!r}")
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


def triton_missing_message(what: str) -> str:
    """What a user without triton reads when `what` needs the Triton branch: the install line where neurobrix's triton
    requirement holds (where PyPI ships a triton wheel, pyproject.toml), else that PyPI has none for this platform and
    the PyTorch engine runs here. Read from the installed metadata; metadata that cannot say it is named."""
    import importlib.metadata
    import platform
    from packaging.requirements import Requirement
    try:
        reqs = [Requirement(r) for r in importlib.metadata.requires("neurobrix") or []]
    except importlib.metadata.PackageNotFoundError:
        return (f"{what} runs on the Triton engine and triton is not installed; this python has no package metadata "
                f"for neurobrix (a source tree on the path), so whether PyPI ships a triton for this platform cannot "
                f"be read: install it, pip install -e . from the source tree")
    here = {"sys_platform": sys.platform, "platform_machine": platform.machine()}     # what the message names
    req = next((r for r in reqs if r.name == "triton" and (r.marker is None or r.marker.evaluate(here))), None)
    here = " ".join(here.values())
    install = ("install the PyTorch build for your hardware (https://pytorch.org/get-started/locally/), then pip "
               "install \"neurobrix[pytorch]\", and run with --compiled")
    if req is not None:     # the PyTorch engine as it stands here: its flag with torch installed, else its install
        return (f"{what} runs on the Triton engine, and triton is not installed in this environment "
                f"({sys.executable}).\n  neurobrix asks for it on {here}: pip install \"{req.name}{req.specifier}\"\n"
                f"  Or run the PyTorch engine, which needs no triton: "
                f"{'--compiled' if importlib.util.find_spec('torch') is not None else install} (or --sequential).")
    return (f"{what} runs on the Triton engine, which needs triton, and PyPI publishes no triton for {here}: the Triton "
            f"engine does not run on this platform.\n  The PyTorch engine runs here: {install} (or no engine flag).")


def default_mode() -> tuple:
    """The mode of a run that names none, and the launch line that says it: `compiled` where torch is installed,
    `triton` where it is not — the light install (`pip install neurobrix`) runs at its first `neurobrix run` instead of
    meeting a refusal (the owner via the supervisor, 2026-11-06 22:10). A mode asked for by its flag is never changed."""
    if importlib.util.find_spec("torch") is not None:
        return "compiled", "engine: compiled (PyTorch installed; --triton runs the Triton engine)"
    return "triton", "engine: triton (torch not installed)"
