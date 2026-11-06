"""What to install when a GPU is in the machine and the library cannot use it: the vendor's runtime library or the
Triton build that compiles for it is missing. Read from the machine (PCI display devices, the runtime libraries the
backend tables name, Triton's registered backends), answered with the vendors' own commands and the install page.

The commands are the vendors' (read 2026-11-06): ROCm's quick start (rocm.docs.amd.com, "Quick start installation
guide", amdgpu-install from repo.radeon.com, Ubuntu 24.04 "noble"); Intel's client GPU packages (dgpu-docs.intel.com,
"Installing client GPUs", the kobuk-team/intel-graphics PPA: libze1 + libze-intel-gpu1) and the Triton XPU build
(intel/intel-xpu-backend-for-triton releases; the triton-xpu wheel on download.pytorch.org/whl/xpu). The Triton wheel
on PyPI carries the NVIDIA and AMD backends (its entry points: `amd`, `nvidia`).
"""
from __future__ import annotations

import ctypes
import glob
import os
from typing import Dict, List, Tuple

#: PCI vendor ids (pci-ids.ucw.cz) of the GPUs this library has a backend for
_PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}

#: the backend's runtime libraries (the backend tables'), and the Triton backend that compiles for it
_NEEDS = {"amd": ("hip", "amd"), "intel": ("xpu", "intel"), "nvidia": ("cuda", "nvidia")}

INSTALL_PAGE = "https://github.com/NeuroBrix/nbxtensor#install-on-amd--intel-gpus"

_STEPS = {
    "amd": ("the ROCm HIP runtime", [
        "wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/noble/amdgpu-install_6.4.60403-1_all.deb",
        "sudo apt install ./amdgpu-install_6.4.60403-1_all.deb",
        "sudo apt update && sudo apt install rocm",
        "sudo usermod -a -G render,video $LOGNAME     # then log out and in (or reboot)",
    ], "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html"),
    "intel": ("the Level Zero loader and Intel GPU driver", [
        "sudo add-apt-repository -y ppa:kobuk-team/intel-graphics",
        "sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc",
        "sudo usermod -a -G render $LOGNAME            # then log out and in",
    ], "https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html"),
    "nvidia": ("the NVIDIA driver and the CUDA runtime", [
        "sudo ubuntu-drivers install                   # the recommended NVIDIA driver",
        "pip install nvidia-cuda-runtime-cu12          # the CUDA runtime, if the toolkit is not installed",
    ], "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"),
}

_TRITON = {
    "amd": ("pip install triton", "the Triton wheel on PyPI carries the AMD backend"),
    "intel": ("pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu",
              "Intel's Triton build (intel-xpu-backend-for-triton); the PyPI wheel has no Intel backend"),
    "nvidia": ("pip install triton", "the Triton wheel on PyPI carries the NVIDIA backend"),
}


def pci_gpus(root: str = "/sys/bus/pci/devices") -> List[Tuple[str, int, int]]:
    """[(vendor name, vendor id, device id)] of the display controllers (PCI class 0x03) of the vendors in `_PCI_VENDORS`,
    read from sysfs; [] where there is none (not Linux)."""
    out = []
    for dev in sorted(glob.glob(os.path.join(root, "*"))):
        try:
            with open(os.path.join(dev, "class")) as f:
                cls = int(f.read().strip(), 16)
            with open(os.path.join(dev, "vendor")) as f:
                vendor = int(f.read().strip(), 16)
            with open(os.path.join(dev, "device")) as f:
                device = int(f.read().strip(), 16)
        except (OSError, ValueError):
            continue
        if (cls >> 16) == 0x03 and vendor in _PCI_VENDORS:
            out.append((_PCI_VENDORS[vendor], vendor, device))
    return out


def _runtime_libs(backend: str) -> List[str]:
    from nbxtensor.backend import runtime
    if backend in runtime._GPU_BACKENDS:
        return list(runtime._GPU_BACKENDS[backend]["rt_libs"])
    from nbxtensor.backend.level_zero import LIBS
    return list(LIBS)


def _loads(names: List[str]) -> bool:
    for name in names:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _triton_backends() -> List[str]:
    try:
        from triton.backends import backends
        return sorted(backends)
    except Exception:                      # noqa: BLE001 — no Triton, or one that does not import: none registered
        return []


def diagnose(root: str = "/sys/bus/pci/devices") -> List[Dict[str, object]]:
    """One entry per GPU vendor found on the PCI bus: what is missing to use it (`missing`: the runtime library,
    the Triton backend) and the steps that install it."""
    found, out = {}, []
    for vendor, _vid, device in pci_gpus(root):
        found.setdefault(vendor, []).append(device)
    registered = _triton_backends()
    for vendor, devices in sorted(found.items()):
        backend, triton_name = _NEEDS[vendor]
        what, steps, page = _STEPS[vendor]
        missing = []
        if not _loads(_runtime_libs(backend)):
            missing.append(what)
        if triton_name not in registered:
            missing.append(f"a Triton build with the {triton_name!r} backend ({_TRITON[vendor][1]})")
        out.append({"vendor": vendor, "devices": [f"{d:#06x}" for d in devices], "backend": backend,
                    "missing": missing, "steps": (steps if len(missing) and what in missing else [])
                    + ([_TRITON[vendor][0]] if triton_name not in registered else []), "page": page})
    return out


def report(root: str = "/sys/bus/pci/devices") -> str:
    """The diagnosis as text: per vendor, ready or what to install and how."""
    lines = []
    for d in diagnose(root):
        head = f"{d['vendor'].upper()} GPU ({', '.join(d['devices'])}), backend {d['backend']!r}: "
        if not d["missing"]:
            lines.append(head + "ready (runtime library and Triton backend found)")
            continue
        lines.append(head + "missing " + "; ".join(d["missing"]))
        lines.append("  install with:")
        lines.extend(f"    {s}" for s in d["steps"])
        lines.append(f"  vendor guide: {d['page']}")
        lines.append(f"  nbxtensor's install page: {INSTALL_PAGE}")
    return "\n".join(lines) if lines else "no NVIDIA, AMD or Intel GPU found on the PCI bus"


def no_runtime_message(root: str = "/sys/bus/pci/devices") -> str:
    """The refusal when no GPU runtime answers: which runtimes were tried, and — when a GPU IS in the machine — what to
    install for it, never a bare trace."""
    head = "No GPU runtime found (tried CUDA, ROCm/HIP, Level Zero and Metal)."
    if not pci_gpus(root):
        return head
    return f"{head}\n{report(root)}"


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":                 # python -m nbxtensor.backend.install_help
    raise SystemExit(main())
