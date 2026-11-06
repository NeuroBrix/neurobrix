"""`neurobrix doctor` — diagnose an installation before it wastes the user's time: is the engine installed, can each of
its two engines (Triton on nbxtensor; PyTorch, optional) see the GPU, and if not, exactly which command fixes it. A false
all-clear is the fault it exists to prevent (until 2026-09-03 it checked PATH only, on machines whose torch build could
not see the card); it exits non-zero when nothing can run, so it is usable in a script."""

from __future__ import annotations

import shutil
import subprocess


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


def _triton_engine() -> tuple[bool, str]:
    """(ready, what to say) for the Triton engine (`--triton`): nbxtensor and Triton import, and nbxtensor finds a GPU
    runtime. Torch-free: PyTorch is not this engine's dependency, and its absence is not this engine's fault."""
    try:
        import nbxtensor  # noqa: F401
        import triton
    except ImportError as exc:
        return False, f"not installed ({exc}): pip install neurobrix"
    try:
        from nbxtensor.backend.runtime import gpu_backend
        return True, f"ready on {gpu_backend()} (nbxtensor {nbxtensor.__version__}, Triton {triton.__version__})"
    except RuntimeError as exc:    # names what to install for a GPU on the bus (nbxtensor.backend.install_help)
        return False, f"no GPU runtime: {exc}"


def _torch_present() -> bool:
    """Whether PyTorch is installed, asked without importing it (the Triton engine's environment has none)."""
    import importlib.util
    import sys
    return "torch" in sys.modules or importlib.util.find_spec("torch") is not None


def check_compute_environment() -> list[str]:
    """Return a list of blocking problems, printing the diagnosis as it goes. Two engines: the Triton engine
    (`--triton`, nbxtensor, no PyTorch) and the PyTorch engine (the default, `neurobrix[pytorch]`). A machine where
    one of them can run has no blocking problem; the other is said, with how to add it."""
    problems: list[str] = []
    print("Compute environment")
    print("-" * 60)
    triton_ready, triton_says = _triton_engine()
    print(f"  Triton engine:  {triton_says}")
    from nbxtensor.backend.install_help import report
    print("  GPU drivers:    " + report().replace("\n", "\n                  "))

    if not _torch_present():
        print("  PyTorch engine: not installed (optional; the Triton engine runs without it: --triton). To add it: "
              f"{_torch_install_hint(_nvidia_smi_gpus()[0][1] if _nvidia_smi_gpus() else None)}, then "
              "pip install 'neurobrix[pytorch]'")
        if not triton_ready:
            problems.append(f"Neither engine can run: the Triton engine is {triton_says}; PyTorch is not installed")
        return problems
    import torch

    cuda_build = getattr(torch.version, "cuda", None)
    hip_build = getattr(torch.version, "hip", None)
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  built for:      {'ROCm ' + hip_build if hip_build else 'CUDA ' + str(cuda_build)}")

    gpus = _nvidia_smi_gpus()
    driver = _driver_version()
    if driver:
        print(f"  NVIDIA driver:  {driver}")

    if torch.cuda.is_available():
        arch_list = torch.cuda.get_arch_list()
        print(f"  GPU visible:    yes ({torch.cuda.device_count()} device(s))")
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            sm = f"sm_{major}{minor}"
            supported = sm in arch_list or hip_build is not None
            mark = "ok" if supported else "NOT SUPPORTED by this PyTorch build"
            print(f"    [{i}] {torch.cuda.get_device_name(i)}  ({sm})  {mark}")
            if not supported:
                problems.append(
                    f"GPU {i} is {sm}, which this PyTorch build does not include "
                    f"(it has: {', '.join(arch_list)}).\n"
                    f"      Fix: {_torch_install_hint(f'{major}.{minor}')}"
                )
        return problems

    # No CUDA. Distinguish "no GPU" from "GPU present, torch cannot use it" —
    # they are completely different problems for the user.
    print("  GPU visible:    NO")
    if not gpus:
        print("    No NVIDIA GPU detected by the driver either — CPU execution.")
        print("    This is a supported path: hardware auto-detection selects a "
              "CPU profile and")
        print("    small models run without any configuration.")
        if cuda_build:
            # not a problem, but several GB of CUDA wheels nothing uses; an extra cannot pull the CPU wheel (PEP 508
            # carries no index URL), so the recipe is printed where it is diagnosed
            print()
            print("    Note: PyTorch is installed with CUDA support "
                  f"(CUDA {cuda_build}) on a machine with no")
            print("    NVIDIA GPU — that is several GB of unused wheels. For a "
                  "CPU-only install:")
            print(f"      {_cpu_install_hint()}")
            print("      pip install neurobrix")
        return problems

    caps = ", ".join(f"{n} ({c})" for n, c in gpus)
    print(f"    But the driver DOES see: {caps}")
    print("    So PyTorch was installed for a CUDA version this driver or GPU "
          "cannot serve.")
    first_cap = gpus[0][1] if gpus else None
    problems.append(
        f"A GPU is present ({caps}) but PyTorch reports no CUDA.\n"
        f"      This is almost always a torch build mismatch: `pip install "
        f"neurobrix` takes\n"
        f"      whatever CUDA build the default index currently ships, which "
        f"may be newer\n"
        f"      than your driver — and CUDA 13 dropped Volta (sm_70) "
        f"entirely.\n"
        f"      Fix: {_torch_install_hint(first_cap)}"
    )
    return problems


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
    print()
    problems = check_compute_environment()
    print()

    if not problems:
        print("No blocking problem found: the engine is installed and can see "
              "your GPU.")
        return 0

    print(f"{len(problems)} problem(s) that will block a real run:")
    for n, problem in enumerate(problems, 1):
        print(f"  {n}. {problem}")
    print()
    print("Re-run `neurobrix doctor` after applying the fix.")
    return 1
