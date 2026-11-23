"""Which engine a command runs on: ONE rule for every command that runs a model (`run`, `upscale`).

`upscale` had its own: a `--mode` flag defaulting to `compiled` whatever was installed, and the Triton branch's hardware
declaration made for every mode. On a host with no GPU the declaration refused ("no hardware profile with a device")
while `run` answered the same model (2026-11-13). A command does not choose its engine on its own: it calls this.

  * the flags: `--compiled`, `--sequential`, `--triton`, `--triton-sequential`, mutually exclusive (`add_engine_flags`);
  * no flag: the Triton engine, whether torch is installed or not, said on the launch line
    (`core/runtime/engines.default_mode`); the PyTorch branch runs only on its flag;
  * a Triton mode: the backend for this hardware must be present (Metal, triton-cpu), refused with its install line;
  * a PyTorch mode without torch: refused by name (`core/runtime/engines.require_engine`);
  * the Triton branch's hardware declaration only where a Triton kernel can run (`declare_for`).
"""
from __future__ import annotations

from typing import Optional
from neurobrix import branch_door as _branch_door

#: the flag dest -> the execution mode it asks for
_FLAGS = (("compiled", "compiled"), ("sequential", "sequential"), ("triton", "triton"),
          ("triton_sequential", "triton_sequential"))


def add_engine_flags(p) -> None:
    """The four mutually exclusive engine flags, with one help, on a command's parser."""
    p.add_argument('--compiled', action='store_true',
                   help='PyTorch fused graph + cuDNN/cuBLAS (CompiledSequence). Production PyTorch performance. '
                        'The PyTorch branch runs only when this flag (or --sequential) asks for it; with no '
                        'execution flag the Triton engine runs.')
    p.add_argument('--sequential', action='store_true',
                   help='PyTorch eager op-by-op (no fusion, ATen dispatcher). Useful for debugging individual ATen '
                        'ops. Slower than --compiled.')
    p.add_argument('--triton', action='store_true',
                   help='Triton-pure compiled mode: every op runs on NeuroBrix\'s own Triton kernels, with no cuDNN '
                        'or cuBLAS, through a preallocated memory arena and fused kernels (TritonSequence). The '
                        'default when no execution flag is passed. Needs the triton package.')
    p.add_argument('--triton-sequential', action='store_true', dest='triton_sequential',
                   help='Triton-pure eager mode: Triton kernels op-by-op (no fusion, TritonSequentialDispatcher). '
                        'Equivalent of --sequential but for the Triton backend. Useful for debugging individual '
                        'Triton kernels.')


def asked_mode(args) -> Optional[str]:
    """The mode `args` names by its flag, else the mode a run that names none takes (`core/runtime/engines.default_mode`),
    read only: nothing checked, nothing said (`choose_engine` does both). None when several flags are passed (the
    refusal is `choose_engine`'s). The branch door is told this mode before anything reads the hardware, so that the
    command's very first Prism call is its branch's (the detection runs before the engine is checked)."""
    asked = [mode for dest, mode in _FLAGS if getattr(args, dest, False)]
    if len(asked) > 1:
        return None
    if asked:
        return asked[0]
    from neurobrix.core.runtime.engines import default_mode
    return default_mode()[0]


def choose_engine(args, log) -> Optional[str]:
    """The execution mode `args` asks for, its backend checked; None after the refusal is logged (the caller exits 1)."""
    asked = [mode for dest, mode in _FLAGS if getattr(args, dest, False)]
    if len(asked) > 1:
        log.error("Error: only one execution mode flag can be passed at a time: one of --compiled, --sequential, "
                  "--triton, --triton-sequential")
        return None
    if asked:
        mode = asked[0]
    else:
        from neurobrix.core.runtime.engines import default_mode
        mode, said = default_mode()
        log.info(f"[Engine] {said}")
    if mode in ("triton", "triton_sequential") and not _triton_backend_present(log):
        return None
    if asked:
        log.info(f"[Engine] engine: {mode}")       # the engine a flag asked for, said like the default
    # The PyTorch branch without torch installed: refused by name before anything is loaded (core/runtime/engines.py).
    from neurobrix.core.runtime import engines
    try:
        engines.require_engine(mode)
    except (engines.PyTorchBranchNotInstalledError, engines.TritonBranchNotInstalledError) as exc:
        log.error(f"Error: {exc}")
        return None
    _branch_door.choose(mode)                   # the branch this process runs: common code asks the door for its side
    return mode


def _triton_backend_present(log) -> bool:
    """The Triton backend for this hardware is installed (and, on a Mac, not on storage the machine clears); else the
    refusal with its install line is logged. Neither gate fetches anything on the user's behalf."""
    from nbxtensor.backend.cpu_backend import TritonCPUNotInstalledError, ensure_triton_cpu_or_raise
    from nbxtensor.backend.metal_backend import TritonMetalNotInstalledError, ensure_triton_metal_or_raise
    from neurobrix.core.paths import BackendOnClearedStorage, refuse_a_backend_on_cleared_storage
    # A census shadow (NBX_CENSUS=1, kernels/census.py) launches nothing: it runs with NO visible device on purpose
    # (the door that proves it cannot touch a card), so the presence gates, which answer for a process that WILL
    # launch, do not apply to it.
    from neurobrix.kernels import census as _census
    if _census.active():
        return True
    try:
        ensure_triton_metal_or_raise()          # Apple GPUs: the out-of-tree Metal backend
        refuse_a_backend_on_cleared_storage()   # the workshop's rule on the selected Metal backend
        ensure_triton_cpu_or_raise()            # a CPU-only profile: triton-cpu is a separate package
    except (TritonMetalNotInstalledError, TritonCPUNotInstalledError, BackendOnClearedStorage) as exc:
        log.error(f"Error: {exc}")
        return False
    return True


def declare_for(mode: str, hw_profile) -> None:
    """The Triton branch's hardware declaration (`triton.dtype.declare_hardware_profile`, which needs triton), in the
    Triton modes only, as `serving/engine.py` gates it: no PyTorch mode reads it, so a host without triton runs them."""
    TRITON_ENGINES = _branch_door.names("prism", "runtime_widths", "TRITON_ENGINES")[0]
    if mode in TRITON_ENGINES:
        from neurobrix.triton.dtype import declare_hardware_profile
        declare_hardware_profile(hw_profile)
