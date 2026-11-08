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
import functools
import glob
import os
from typing import Dict, List, Tuple

#: PCI vendor ids (pci-ids.ucw.cz) of the GPUs this library has a backend for
_PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}

#: the backend's runtime libraries (the backend tables'), and the Triton backend that compiles for it
_NEEDS = {"amd": ("hip", "amd"), "intel": ("xpu", "intel"), "nvidia": ("cuda", "nvidia")}

@functools.lru_cache(maxsize=1)
def _steps() -> dict:
    """The vendors' install steps (`install_steps.yml`, beside this module, each with its source)."""
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "install_steps.yml")) as f:
        return yaml.safe_load(f)

def pci_gpus(root: str = "/sys/bus/pci/devices") -> List[Tuple[str, int, int]]:
    """[(vendor name, vendor id, device id)] of the display controllers (PCI class 0x03) of the vendors in `_PCI_VENDORS`,
    read from sysfs; [] where there is none (not Linux)."""
    return [(v, vid, dev) for v, vid, dev, _bus in _pci_gpus_at(root)]


def _bus_key(bus: str):
    """(domain, bus, device, function) of a PCI address, however its domain is padded ("0000:1A:00.0" from the
    runtime, "0000:1a:00.0" in sysfs, "00000000:1A:00.0" from NVML); None when it is no PCI address."""
    try:
        dom, b, rest = str(bus).strip().split(":")
        dev, fn = rest.split(".")
        return int(dom, 16), int(b, 16), int(dev, 16), int(fn, 16)
    except ValueError:
        return None


def _visible_bus_ids(count=None, bus_of=None):
    """{PCI address key} of the GPUs THIS process sees (the backend runtime's devices: `CUDA_VISIBLE_DEVICES`, a
    container's `--gpus`), with the backend whose runtime answered; None where no runtime answers or it cannot name its
    devices' bus. sysfs lists every display controller of the host: a container given one card of four listed the four
    (the terminal-noise inventory, 2026-11-08). `count`/`bus_of(i)` are injected by the tests."""
    try:
        from nbxtensor.backend import runtime as _rt
        backend = _rt.gpu_backend()
        if count is None:
            table = _rt.backend_table()
            fn = table.get("pci_bus_id")
            if not fn:
                return None
            lib = _rt.gpu_runtime()
            n = ctypes.c_int(0)
            if getattr(lib, table["device_count"])(ctypes.byref(n)) != 0:
                return None
            count = n.value

            def bus_of(i):
                buf = ctypes.create_string_buffer(64)
                return buf.value.decode() if getattr(lib, fn)(buf, 64, int(i)) == 0 else None
    except Exception:                      # noqa: BLE001 — no runtime: the bus is all this module can read
        return None
    keys = {_bus_key(bus_of(i)) for i in range(int(count))}
    return None if None in keys else (backend, keys)


def _pci_gpus_at(root: str) -> List[Tuple[str, int, int, str]]:
    """`pci_gpus` with each device's PCI address (its sysfs directory's name)."""
    return [(_PCI_VENDORS[v], v, d, bus) for v, d, bus in pci_display_devices(root) if v in _PCI_VENDORS]


def pci_display_devices(root: str = "/sys/bus/pci/devices") -> List[Tuple[int, int, str]]:
    """[(vendor id, device id, PCI address)] of every display controller on the bus, any vendor: PCI base class 0x03
    (VGA 0x0300, 3D 0x0302, other 0x0380 — the class an Alder Lake-P integrated GPU may report), read from sysfs (the
    kernel's sysfs-bus-pci ABI: `class`, `vendor`, `device` of each function); [] where there is none (not Linux)."""
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
        if (cls >> 16) == 0x03:
            out.append((vendor, device, os.path.basename(dev)))
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


def diagnose(root: str = "/sys/bus/pci/devices", visible=None) -> List[Dict[str, object]]:
    """One entry per GPU vendor found on the PCI bus: what is missing to use it (`missing`: the runtime library,
    the Triton backend) and the steps that install it. Where a runtime answers (`_visible_bus_ids`), its vendor's
    devices are the ones this process sees; `hidden` counts that vendor's others on the bus."""
    found, hidden, out = {}, {}, []
    seen = _visible_bus_ids() if visible is None else visible
    for vendor, _vid, device, bus in _pci_gpus_at(root):
        if seen and _NEEDS[vendor][0] == seen[0] and _bus_key(bus) not in seen[1]:
            hidden[vendor] = hidden.get(vendor, 0) + 1
            continue
        found.setdefault(vendor, []).append(device)
    for vendor in hidden:
        found.setdefault(vendor, [])
    registered = _triton_backends()
    for vendor, devices in sorted(found.items()):
        backend, triton_name = _NEEDS[vendor]
        row = _steps()["vendors"][vendor]
        what, steps, page = row["runtime"], row["steps"], row["guide"]
        missing = []
        if not _loads(_runtime_libs(backend)):
            missing.append(what)
        if triton_name not in registered:
            missing.append(f"a Triton build with the {triton_name!r} backend ({row['triton_note']})")
        out.append({"vendor": vendor, "devices": [f"{d:#06x}" for d in devices], "hidden": hidden.get(vendor, 0),
                    "backend": backend,
                    "missing": missing, "steps": (steps if len(missing) and what in missing else [])
                    + ([row["triton_step"]] if triton_name not in registered else []), "page": page})
    return out


def report(root: str = "/sys/bus/pci/devices", visible=None) -> str:
    """The diagnosis as text: per vendor, ready or what to install and how."""
    lines = []
    for d in diagnose(root, visible):
        seen = ", ".join(d["devices"]) or "none visible to this process"
        head = (f"{d['vendor'].upper()} GPU ({seen}"
                + (f"; {d['hidden']} more on the host, not visible here" if d["hidden"] else "")
                + f"), backend {d['backend']!r}: ")
        if not d["missing"]:
            lines.append(head + "ready (runtime library and Triton backend found)")
            continue
        lines.append(head + "missing " + "; ".join(d["missing"]))
        lines.append("  install with:")
        lines.extend(f"    {s}" for s in d["steps"])
        lines.append(f"  vendor guide: {d['page']}")
        lines.append(f"  nbxtensor's install page: {_steps()['install_page']}")
    return "\n".join(lines) if lines else "no NVIDIA, AMD or Intel GPU found on the PCI bus"


def no_runtime_message(root: str = "/sys/bus/pci/devices") -> str:
    """The refusal when no GPU runtime answers: which runtimes were tried, and — when a GPU IS in the machine — what to
    install for it, never a bare trace."""
    head = "No GPU runtime found (tried CUDA, ROCm/HIP, Level Zero and Metal)."
    if not pci_gpus(root):
        return head
    return f"{head}\n{report(root)}"


def hidden_gpus(root: str = "/sys/bus/pci/devices"):
    """(variable, its value, the GPUs of the backend's vendor on the PCI bus) when the variable the backend's runtime
    reads to hide devices (its table's `visible_env`) leaves this process fewer devices than the bus holds; None
    otherwise — read from the environment and this library's own listings, never from an error's text."""
    import os
    from nbxtensor.backend import runtime as _rt
    from nbxtensor.memory.allocator import DeviceAllocator
    try:
        backend, names = _rt.gpu_backend(), _rt.backend_table().get("visible_env") or ()
    except RuntimeError:
        return None
    set_ = [(n, os.environ[n]) for n in names if n in os.environ]
    on_bus = sum(1 for v, *_ in pci_gpus(root) if _NEEDS[v][0] == backend)
    return (*set_[0], on_bus) if set_ and on_bus > DeviceAllocator.device_count() else None


def compiler_missing(exc: BaseException):
    """The one line to say when `exc` came from a machine with no C (or C++) compiler for Triton to build a kernel with;
    None for any other failure. torch compiles some of its kernels with Triton at their first use (torch 2.14.1 sends
    `aten::bmm` to its Triton kernel), and Triton builds each one with the machine's compiler: `triton/runtime/build.py`,
    `_find_compiler`, raises RuntimeError("Failed to find C compiler ...") where none is found — read here from a frame of
    that function in the failure's chain, never from the message's text (a `python:3.10-slim` container has none)."""
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        tb = e.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if code.co_name == "_find_compiler" and code.co_filename.replace("\\", "/").endswith("triton/runtime/build.py"):
                which = "C++" if tb.tb_frame.f_locals.get("language") == "c++" else "C"
                return (f"the PyTorch engine needs a {which} compiler and this machine has none: torch compiles some "
                        f"of its kernels with Triton at their first use, and Triton builds them with one. Install a "
                        f"{which} compiler (e.g. apt install {'g++' if which == 'C++' else 'gcc'}), or run the Triton "
                        f"engine, which needs none: add --triton.")
            tb = tb.tb_next
        stack.extend([e.__cause__, e.__context__])
    return None


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":                 # python -m nbxtensor.backend.install_help
    raise SystemExit(main())
