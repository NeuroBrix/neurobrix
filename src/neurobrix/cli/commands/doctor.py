"""`neurobrix doctor` — diagnose an installation before it wastes the user's time.

Until 2026-09-03 this command checked only whether the `neurobrix` script
was on PATH, and printed "No action needed" whenever it was. On a machine
with a GPU the engine cannot use, that is a false all-clear: the user is
told everything is fine, then meets an obscure failure on their first real
run and leaves. A diagnostic that cannot see the most common installation
fault is worse than no diagnostic.

The most common fault is not exotic. `pip install neurobrix` resolves
`torch` from the default index, which ships whatever CUDA build is current
— today that is CUDA 13, which requires a recent driver and **dropped
Volta (sm_70) entirely**. A user on a V100, or on any machine whose driver
predates the wheel's CUDA version, ends up with a torch that reports no
GPU at all. Nothing in the installation says so.

So this command answers three questions in the user's own terms: is the
engine installed, can it see your GPU, and if not, exactly which command
fixes it. It exits non-zero when it finds something that will block a real
run, so it is usable in a script.
"""

from __future__ import annotations

import shutil
import subprocess


def _nvidia_smi_gpus() -> list[tuple[str, str]]:
    """(name, compute capability) per physical GPU, straight from the driver.

    Asked independently of torch on purpose: when torch reports no CUDA, the
    whole question is whether a GPU is physically present, and torch cannot
    answer that once it has given up.
    """
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


def _torch_install_hint(cap: str | None) -> str:
    """The exact command that fixes a mismatched torch, per GPU generation."""
    if cap and cap.startswith("7.0"):
        return ("pip install --force-reinstall torch "
                "--index-url https://download.pytorch.org/whl/cu121")
    return ("pip install --force-reinstall torch "
            "--index-url https://download.pytorch.org/whl/cu124")


def check_compute_environment() -> list[str]:
    """Return a list of blocking problems, printing the diagnosis as it goes."""
    problems: list[str] = []
    print("Compute environment")
    print("-" * 60)

    try:
        import torch
    except ImportError as exc:
        print(f"  PyTorch:        NOT INSTALLED ({exc})")
        problems.append("PyTorch is not installed: pip install neurobrix")
        return problems

    cuda_build = getattr(torch.version, "cuda", None)
    hip_build = getattr(torch.version, "hip", None)
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  built for:      {'ROCm ' + hip_build if hip_build else 'CUDA ' + str(cuda_build)}")

    try:
        import triton
        print(f"  Triton:         {triton.__version__}  (--triton mode available)")
    except ImportError:
        print("  Triton:         not installed  (--triton mode unavailable; "
              "the default compiled engine still works)")

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
        print("    No NVIDIA GPU detected by the driver either.")
        print("    NeuroBrix runs on CPU for some models, but every benchmark "
              "and most models expect a GPU.")
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
