"""`neurobrix doctor` — diagnose an installation before it wastes the user's time: is the engine installed, can each of
its two engines (Triton on nbxtensor; PyTorch, optional) see the GPU, and if not, exactly which command fixes it. A false
all-clear is the fault it exists to prevent (until 2026-09-03 it checked PATH only, on machines whose torch build could
not see the card); it exits non-zero when nothing can run, so it is usable in a script."""

from __future__ import annotations

import logging
import shutil
import subprocess

#: the diagnosis's facts (build, driver, PCI ids) under -v; the verdicts on stdout
log = logging.getLogger(__name__)


def _nvidia_smi_gpus() -> list[tuple[str, str]]:
    """(name, compute capability) per physical GPU, straight from the driver — asked of neither engine: when one sees
    no CUDA, the question is whether a GPU is physically present."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        if "," in line:
            name, _, cap = line.partition(",")
            gpus.append((name.strip(), cap.strip()))
    return gpus


def _driver_version() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0].strip()
    except (subprocess.SubprocessError, OSError, IndexError):
        return None


def _torch_wheels() -> dict:
    """The wheel table, `config/torch_wheels.yml`: one measured build per architecture, each with its source."""
    import yaml
    from neurobrix.core.config.loader import CONFIG_ROOT
    with open(CONFIG_ROOT / "torch_wheels.yml") as f:
        return yaml.safe_load(f)


def _torch_install_hint(cap: str | None) -> str:
    """The command that installs the torch build measured on this GPU's architecture (the wheel table); an
    architecture the table does not list is sent to PyTorch's selector, never given a build nobody ran on it."""
    from neurobrix.core.prism.autodetect import _nvidia_cc_to_arch
    table = _torch_wheels()
    index_url = table["index_url"]
    arch = _nvidia_cc_to_arch(cap) if cap else None
    if arch in table["builds"]:
        build = table["builds"][arch]["build"]
        return (f"pip install --force-reinstall torch "
                f"--index-url {index_url}{build}")
    return (f"no PyTorch build has been measured on this GPU "
            f"({arch or 'architecture unknown'}); pick the CUDA build your "
            f"driver supports at {table['selector_url']}, then:\n"
            f"        pip install --force-reinstall torch "
            f"--index-url {index_url}<cuda-build>")


def _cpu_install_hint() -> str:
    """The CPU-only torch install command, from the wheel table."""
    table = _torch_wheels()
    return (f"pip install torch --index-url "
            f"{table['index_url']}{table['cpu_only']['build']}")


#: what a Mac needs for the Triton engine: PyPI publishes no Triton for macOS; nbxtensor's release carries it
_MAC_TRITON = "Triton for Metal not installed — run: python -m nbxtensor install-metal"


def _apple_silicon() -> bool:
    from nbxtensor.backend.metal_backend import is_apple_silicon
    return is_apple_silicon()


def _triton_engine() -> tuple[bool, str]:
    """(ready, what to say) for the Triton engine (`--triton`): nbxtensor and Triton import, and nbxtensor finds a GPU
    runtime. Torch-free: PyTorch is not this engine's dependency, and its absence is not this engine's fault."""
    try:
        import nbxtensor  # noqa: F401
        import triton
    except ImportError as exc:
        if _apple_silicon():
            return False, f"not installed ({exc}): {_MAC_TRITON}"
        return False, f"not installed ({exc}): pip install neurobrix"
    from nbxtensor.backend.cpu_backend import triton_cpu_installed
    from nbxtensor.memory.allocator import DeviceAllocator
    try:
        from nbxtensor.backend.runtime import gpu_backend
        backend = gpu_backend()
    except RuntimeError as exc:    # names what to install for a GPU on the bus (nbxtensor.backend.install_help)
        return False, f"no GPU runtime: {exc}"
    if DeviceAllocator.device_count() > 0:
        return True, f"ready on {backend} (nbxtensor {nbxtensor.__version__}, Triton {triton.__version__})"
    cpu = triton_cpu_installed()      # no device this process sees: what the engine can run on is the CPU, or nothing
    return cpu, f"no {backend} device visible{_hidden_said()}; " + ("runs on the CPU (triton-cpu)" if cpu else
                                                                       "triton-cpu not installed: nothing to run on")


def _hidden_said() -> str:
    """`: GPUs hidden by <variable>='<value>' (the driver sees N)` when the environment hides the devices, else ''."""
    from nbxtensor.backend.install_help import hidden_gpus
    hidden = hidden_gpus()
    return f": GPUs hidden by {hidden[0]}={hidden[1]!r} (the driver sees {hidden[2]})" if hidden else ""


def _torch_present() -> bool:
    """Whether PyTorch is installed, asked without importing it (the Triton engine's environment has none)."""
    import importlib.util
    import sys
    return "torch" in sys.modules or importlib.util.find_spec("torch") is not None


def check_compute_environment() -> list[str]:
    """Return the blocking problems, printing the diagnosis as it goes. Two engines: the Triton engine (`--triton`,
    nbxtensor, no PyTorch) and the PyTorch engine (`neurobrix[pytorch]`). Blocking only when neither can run; what the
    other lacks is said, with how to add it."""
    problems: list[str] = []
    triton_ready, triton_says = _triton_engine()
    print(f"  Triton engine:  {triton_says}")
    if _apple_silicon():
        # an Apple GPU sits on no PCI bus nbxtensor reads: said as what it is, never "no GPU found"
        print("  GPU drivers:    Apple GPU (Metal), part of macOS")
    else:
        from nbxtensor.backend.install_help import report
        log.debug("  GPU drivers:    " + report().replace("\n", "\n                  "))

    if not _torch_present():
        print("  PyTorch engine: not installed (optional; the Triton engine runs without it: --triton). To add it: "
              f"{'pip install torch' if _apple_silicon() else _torch_install_hint(_nvidia_smi_gpus()[0][1] if _nvidia_smi_gpus() else None)}, then "
              "pip install 'neurobrix[pytorch]'")
        if not triton_ready:
            problems.append(f"Neither engine can run: the Triton engine is {triton_says}; PyTorch is not installed")
        return problems
    import torch

    cuda_build = getattr(torch.version, "cuda", None)
    hip_build = getattr(torch.version, "hip", None)
    log.debug(f"  PyTorch:        {torch.__version__}")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    on_mps = mps is not None and mps.is_available()
    built = ("ROCm " + hip_build if hip_build else "CUDA " + cuda_build if cuda_build else
             "Apple MPS (Metal)" if on_mps else "CPU only")
    log.debug(f"  built for:      {built}")

    gpus = _nvidia_smi_gpus()
    driver = _driver_version()
    if driver:
        log.debug(f"  NVIDIA driver:  {driver}")

    if torch.cuda.is_available():
        arch_list = torch.cuda.get_arch_list()
        print(f"  PyTorch engine: {torch.__version__} ({built}), {torch.cuda.device_count()} GPU(s) visible")
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            sm = f"sm_{major}{minor}"
            supported = sm in arch_list or hip_build is not None
            mark = "ok" if supported else "NOT SUPPORTED by this PyTorch build"
            print(f"    [{i}] {torch.cuda.get_device_name(i)}  ({sm})  {mark}")
            if not supported:
                problems.append(f"GPU {i} is {sm}, which this PyTorch build does not include (it has: "
                                f"{', '.join(arch_list)}). Fix: {_torch_install_hint(f'{major}.{minor}')}")
        for p in problems if triton_ready else ():      # the Triton engine runs: advice, not a blocking problem
            print(f"    {p}")
        return [] if triton_ready else problems

    if on_mps:
        print(f"  PyTorch engine: {torch.__version__} (Apple GPU, mps)")
        return problems

    # No CUDA: no GPU, GPUs the environment hides, or GPUs this torch build cannot use — three different answers
    print(f"  PyTorch engine: {torch.__version__} ({built}), no GPU visible{_hidden_said()}: it runs on the CPU")
    if not gpus or _hidden_said():
        if cuda_build and not gpus:
            # several GB of CUDA wheels nothing uses; an extra cannot pull the CPU wheel (PEP 508 carries no index URL)
            print(f"    Note: PyTorch is installed with CUDA {cuda_build} on a machine with no NVIDIA GPU — several GB of "
                  f"unused wheels. For a CPU-only install: {_cpu_install_hint()}, then pip install neurobrix")
        return problems

    caps = ", ".join(f"{n} ({c})" for n, c in gpus)
    fix = (f"the driver sees {caps} and PyTorch reports no CUDA: almost always a torch build mismatch (the default index's "
           f"CUDA build may be newer than the driver; CUDA 13 dropped Volta). Fix: {_torch_install_hint(gpus[0][1])}")
    print(f"    {fix}")
    return [] if triton_ready else [f"Neither engine can use the GPU: the Triton engine is {triton_says}; {fix}"]


def cmd_doctor(args) -> int:
    """PATH diagnosis (the original check) followed by the compute check."""
    from neurobrix.cli.json_out import wants_json, human_lines_to_stderr, emit
    if wants_json(args):
        with human_lines_to_stderr(True):
            from neurobrix.cli._path_helper import print_path_diagnostics
            print_path_diagnostics()
            problems = check_compute_environment()
        emit("doctor", {"ok": not problems, "problems": list(problems)})
        return 0 if not problems else 1
    from neurobrix.cli._path_helper import print_path_diagnostics

    print_path_diagnostics()
    problems = check_compute_environment()

    if not problems:
        print("No blocking problem found: an engine can run here.")
        return 0

    print(f"{len(problems)} problem(s) that will block a real run:")
    for n, problem in enumerate(problems, 1):
        print(f"  {n}. {problem}")
    print("Re-run `neurobrix doctor` after applying the fix.")
    return 1
