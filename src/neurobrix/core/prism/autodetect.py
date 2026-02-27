"""
NeuroBrix Prism — Universal Hardware Auto-Detection (OS-First Architecture)

Detects CPU + GPU hardware and generates a hardware profile YAML.

Detection flow:
  1. Detect OS (Linux / macOS / Windows)
  2. Detect CPU (always — every machine has one)
  3. Detect GPUs via OS-specific cascade
  4. Detect interconnect (multi-GPU only)
  5. Build profile (GPU or CPU-only)

Supports:
  CPU        — x86_64 (AVX2/AVX512/AMX), ARM64 (NEON), RISC-V
  NVIDIA     — nvidia-smi
  AMD        — rocm-smi / rocminfo
  Intel      — xpu-smi / sycl-ls
  Apple      — system_profiler / MPS
  Tenstorrent — tt-smi
  Moore Threads — mtsmi
  Biren      — brsmi
  Iluvatar   — ixsmi
  Hygon DCU  — rocm-smi (DTK fork)
  Cambricon  — cnmon

Called when --hardware is omitted. Creates config/hardware/default.yml once.
"""

import platform
import re
import subprocess
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# CONSTANTS
# ============================================================================

# Same directory as all other hardware profiles (config/hardware/)
HARDWARE_DIR = Path(__file__).parent.parent.parent / "config" / "hardware"
DEFAULT_PROFILE_PATH = HARDWARE_DIR / "default.yml"

# ---------------------------------------------------------------------------
# NVIDIA: compute capability → architecture name
# Source: CUDA Toolkit docs — static hardware fact, nvidia-smi reports CC not name
# ---------------------------------------------------------------------------
_NVIDIA_ARCH_MAP = {
    "3": "kepler",
    "5": "maxwell",
    "6": "pascal",
    "7": "volta",       # 7.0 = V100
    "8": "ampere",      # 8.0 = A100, 8.6 = RTX 3090
    "9": "hopper",      # 9.0 = H100
    "10": "blackwell",  # 10.0 = B200
}
_NVIDIA_ARCH_REFINE = {
    "7.5": "turing",    # RTX 2080, T4
    "8.9": "ada",       # RTX 4090, L40S
}

# NVIDIA: compute capability major → supported dtypes
_NVIDIA_DTYPE_MAP = {
    "3": ["float32"],
    "5": ["float32", "float16"],
    "6": ["float32", "float16"],
    "7": ["float32", "float16"],
    "8": ["float32", "float16", "bfloat16"],
    "9": ["float32", "float16", "bfloat16", "float8_e4m3fn"],
    "10": ["float32", "float16", "bfloat16", "float8_e4m3fn"],
}

# Architecture → preferred dtype (tensor core sweet spot)
_PREFERRED_DTYPE = {
    # NVIDIA
    "kepler": "float32", "maxwell": "float16", "pascal": "float16",
    "volta": "float16", "turing": "float16",
    "ampere": "bfloat16", "ada": "bfloat16",
    "hopper": "bfloat16", "blackwell": "bfloat16",
    # AMD
    "cdna": "float16", "cdna2": "float16", "cdna3": "bfloat16",
    "rdna2": "float16", "rdna3": "float16", "rdna4": "float16",
    # Intel
    "ponte_vecchio": "bfloat16", "flex": "float16",
    "alchemist": "float16", "battlemage": "float16", "xe": "float16",
    # Apple
    "apple_silicon": "float16",
    # Others
    "wormhole": "float16", "grayskull": "float16", "blackhole": "bfloat16",
    "musa": "float16", "birensupa": "float16",
    "corex": "float16", "hygon_dcu": "float16",
    "mlu": "float16",
}

# ---------------------------------------------------------------------------
# AMD: rocminfo gfx ID → architecture name
# ---------------------------------------------------------------------------
_AMD_GFX_MAP = {
    "gfx900": "vega10",     # MI25
    "gfx906": "vega20",     # MI50/MI60
    "gfx908": "cdna",       # MI100
    "gfx90a": "cdna2",      # MI210/MI250/MI250X
    "gfx940": "cdna3",      # MI300A
    "gfx941": "cdna3",      # MI300A variant
    "gfx942": "cdna3",      # MI300X
    "gfx1030": "rdna2",     # RX 6900 XT
    "gfx1100": "rdna3",     # RX 7900 XTX
    "gfx1101": "rdna3",     # RX 7900 XT
    "gfx1150": "rdna4",     # RX 9070 (projected)
    "gfx1200": "rdna4",     # RX 9070
}

# AMD: architecture → supported dtypes
_AMD_DTYPE_MAP = {
    "cdna3": ["float32", "float16", "bfloat16", "float8_e4m3fn"],
    "cdna2": ["float32", "float16", "bfloat16"],
    "cdna": ["float32", "float16"],
    "vega20": ["float32", "float16"],
    "vega10": ["float32", "float16"],
}

# NVLink bandwidth per generation (per link, bidirectional)
_NVLINK_BW_PER_LINK = {
    1: 40,    # Pascal P100
    2: 50,    # Volta V100
    3: 50,    # Ampere A100
    4: 100,   # Hopper H100
    5: 200,   # Blackwell B200
}

# ---------------------------------------------------------------------------
# PCI vendor IDs — for lspci fallback detection
# ---------------------------------------------------------------------------
_PCI_VENDOR_MAP = {
    "10de": "nvidia",
    "1002": "amd",
    "8086": "intel",
    "106b": "apple",
    "1e52": "moore_threads",
    "1e56": "tenstorrent",
    "1f36": "biren",
    "d100": "iluvatar",
    "cabc": "cambricon",
}

# PCIe link speed → version mapping
_PCIE_SPEED_MAP = {
    "2.5": "1.0", "5.0": "2.0", "8.0": "3.0",
    "16.0": "4.0", "32.0": "5.0", "64.0": "6.0",
}

# PCIe version → x16 bidirectional bandwidth (GB/s)
_PCIE_BW_MAP = {
    "1.0": 8, "2.0": 16, "3.0": 32,
    "4.0": 64, "5.0": 128, "6.0": 256,
}


# ============================================================================
# PUBLIC API
# ============================================================================

def get_or_create_default_profile() -> str:
    """
    Ensure default.yml exists in config/hardware/ (same dir as all profiles).

    Returns:
        The hardware_id string "default" (usable with load_profile("default")).
    """
    if DEFAULT_PROFILE_PATH.exists():
        return "default"

    print("   [Auto-detect] No --hardware specified, detecting system hardware...")
    profile_data = detect_hardware()

    HARDWARE_DIR.mkdir(parents=True, exist_ok=True)

    class _NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: object) -> bool:  # noqa: ARG002
            return True

    with open(DEFAULT_PROFILE_PATH, "w") as f:
        summary = profile_data.get("notes", "Auto-detected hardware profile")
        first_line = summary.strip().split("\n")[0]
        f.write(f"# Hardware Profile: {first_line}\n")
        f.write("# Auto-generated by NeuroBrix hardware detection\n")
        f.write("# Regenerate: delete this file and run neurobrix without --hardware\n\n")
        yaml.dump(profile_data, f, Dumper=_NoAliasDumper,
                  default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"   [Auto-detect] Created profile: {DEFAULT_PROFILE_PATH}")
    return "default"


def load_default_profile():
    """
    Load the default hardware profile as a PrismProfile.

    Creates default.yml in config/hardware/ if it doesn't exist,
    then loads it through the standard load_profile() path.
    """
    get_or_create_default_profile()
    from neurobrix.core.prism.loader import load_profile
    return load_profile("default")


# ============================================================================
# MAIN DETECTION ORCHESTRATOR (OS-FIRST)
# ============================================================================

def detect_hardware() -> Dict[str, Any]:
    """
    Detect all hardware (CPU + GPUs + interconnects). Returns a hardware profile dict.

    OS-first detection:
    1. Detect OS → route to OS-specific detectors
    2. CPU detection (always — every machine has a CPU)
    3. GPU detection via OS-specific cascade
    4. Interconnect detection (multi-GPU only)
    """
    os_type = platform.system()  # "Linux", "Darwin", "Windows"

    # --- CPU (always) ---
    cpu = _detect_cpu(os_type)

    # --- GPUs (OS-specific cascade) ---
    devices, brand = _detect_gpus(os_type)

    # --- System vendor ---
    vendor = _detect_system_vendor(os_type)

    # --- Totals ---
    num_gpus = len(devices)
    total_vram_mb = sum(d["memory_mb"] for d in devices)
    total_vram_gb = round(total_vram_mb / 1024, 1) if total_vram_mb > 0 else 0
    ram_gb = round(cpu.get("ram_mb", 0) / 1024, 1) if cpu.get("ram_mb") else 0

    # --- Preferred dtype ---
    if devices:
        arch = devices[0].get("architecture", "unknown")
        preferred_dtype = _PREFERRED_DTYPE.get(arch, "float16")
    else:
        # CPU-only: derive from CPU features
        preferred_dtype = _cpu_preferred_dtype(cpu)

    # --- Profile ID ---
    if devices:
        model_short = _shorten_gpu_name(devices[0].get("model", "gpu"))
        if num_gpus == 1:
            mem_gb = round(devices[0]["memory_mb"] / 1024)
            profile_id = f"auto-{model_short}-{mem_gb}g"
        else:
            profile_id = f"auto-{num_gpus}x{model_short}-{total_vram_gb}g"
    else:
        cpu_short = _shorten_cpu_name(cpu.get("model", "cpu"))
        profile_id = f"auto-{cpu_short}-cpu-{int(ram_gb)}g"

    # --- Interconnect ---
    if num_gpus > 1:
        interconnect, pcie_fallback = _detect_interconnect(devices, brand)
    elif num_gpus == 1:
        pcie_ver = devices[0].get("pcie_version", "3.0")
        pcie_bw = _PCIE_BW_MAP.get(pcie_ver, 32)
        interconnect = {"groups": []}
        pcie_fallback = {"version": pcie_ver, "lanes": 16, "bandwidth_gbps": pcie_bw}
    else:
        # CPU-only: no PCIe relevant
        interconnect = {"groups": []}
        pcie_fallback = {"version": "0.0", "lanes": 0, "bandwidth_gbps": 0}

    # --- Topology label ---
    if num_gpus == 0:
        topology_label = "CPU-Only"
    elif num_gpus == 1:
        topology_label = "Single-GPU"
    else:
        topology_label = "Multi-GPU"

    # --- Summary ---
    summary: Dict[str, Any] = {
        "total_gpus": num_gpus,
        "total_vram_gb": total_vram_gb,
        "total_ram_gb": ram_gb,
        "topology": topology_label,
    }

    # --- Notes ---
    notes = _build_notes(cpu, devices, brand, vendor, interconnect)

    profile: Dict[str, Any] = {
        "id": profile_id,
        "vendor": vendor,
        "preferred_dtype": preferred_dtype,
        "summary": summary,
        "cpu": cpu,
        "devices": devices,
        "interconnect": interconnect,
        "pcie_fallback": pcie_fallback,
        "notes": notes,
    }

    return profile


# ============================================================================
# CPU DETECTION (per-OS)
# ============================================================================

def _detect_cpu(os_type: str) -> Dict[str, Any]:
    """Detect CPU info. Always succeeds — every machine has a CPU."""
    if os_type == "Linux":
        return _detect_cpu_linux()
    elif os_type == "Darwin":
        return _detect_cpu_darwin()
    elif os_type == "Windows":
        return _detect_cpu_windows()
    # Unknown OS — minimal info
    return {
        "model": platform.processor() or "Unknown CPU",
        "cores": 1,
        "threads": 1,
        "ram_mb": 4096,
        "architecture": platform.machine(),
        "features": [],
    }


def _detect_cpu_linux() -> Dict[str, Any]:
    """Detect CPU on Linux via /proc/cpuinfo + /proc/meminfo + lscpu."""
    model = "Unknown CPU"
    features: List[str] = []
    cores = 1
    threads = 1
    ram_mb = 4096
    arch = platform.machine()  # x86_64, aarch64, riscv64

    # --- /proc/cpuinfo ---
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        # Model name (x86)
        m = re.search(r'^model name\s*:\s*(.+)$', cpuinfo, re.MULTILINE)
        if m:
            model = m.group(1).strip()

        # ARM: look for "CPU implementer" + "Hardware" or "model name"
        if model == "Unknown CPU":
            m = re.search(r'^Hardware\s*:\s*(.+)$', cpuinfo, re.MULTILINE)
            if m:
                model = m.group(1).strip()

        # CPU flags (x86) or Features (ARM)
        flags_match = re.search(r'^(?:flags|Features)\s*:\s*(.+)$', cpuinfo, re.MULTILINE)
        if flags_match:
            all_flags = flags_match.group(1).strip().split()
            # Extract relevant features for inference
            relevant = {
                "avx", "avx2", "avx512f", "avx512_vnni", "avx512_bf16",
                "amx_bf16", "amx_int8", "amx_tile",
                "fma", "sse4_1", "sse4_2", "f16c",
                "asimd", "fphp", "asimdhp", "sve", "sve2",  # ARM
            }
            features = [f for f in all_flags if f in relevant]
    except (FileNotFoundError, PermissionError):
        pass

    # --- /proc/meminfo ---
    try:
        meminfo = Path("/proc/meminfo").read_text()
        m = re.search(r'^MemTotal:\s*(\d+)\s*kB', meminfo, re.MULTILINE)
        if m:
            ram_mb = int(m.group(1)) // 1024
    except (FileNotFoundError, PermissionError):
        pass

    # --- lscpu for core/thread counts ---
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Core(s) per socket:" in line:
                    val = line.split(":")[-1].strip()
                    try:
                        sockets_cores = int(val)
                    except ValueError:
                        sockets_cores = 1
                elif "Socket(s):" in line:
                    val = line.split(":")[-1].strip()
                    try:
                        sockets = int(val)
                    except ValueError:
                        sockets = 1
                elif "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                    val = line.split(":")[-1].strip()
                    try:
                        threads = int(val)
                    except ValueError:
                        pass

            # cores = cores_per_socket * sockets
            try:
                cores = sockets_cores * sockets  # type: ignore[possibly-undefined]
            except NameError:
                cores = threads
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback: nproc
        try:
            result = subprocess.run(
                ["nproc"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                threads = int(result.stdout.strip())
                cores = threads  # Can't distinguish without lscpu
        except Exception:
            pass

    return {
        "model": model,
        "cores": cores,
        "threads": threads,
        "ram_mb": ram_mb,
        "architecture": arch,
        "features": features,
    }


def _detect_cpu_darwin() -> Dict[str, Any]:
    """Detect CPU on macOS via sysctl."""
    model = "Unknown CPU"
    cores = 1
    threads = 1
    ram_mb = 8192
    arch = platform.machine()  # arm64 or x86_64
    features: List[str] = []

    # Model name
    model = _sysctl_str("machdep.cpu.brand_string") or "Unknown CPU"

    # If Apple Silicon and brand_string is empty, try hw.model
    if model == "Unknown CPU":
        model = _sysctl_str("hw.model") or "Unknown CPU"

    # Core counts
    cores = _sysctl_int("hw.physicalcpu") or 1
    threads = _sysctl_int("hw.logicalcpu") or cores

    # RAM
    memsize = _sysctl_int("hw.memsize")
    if memsize:
        ram_mb = memsize // (1024 * 1024)

    # CPU features
    if arch == "arm64":
        # Apple Silicon always has NEON + FP16
        features = ["neon", "fp16"]
    else:
        # Intel Mac — check for AVX
        leaf7 = _sysctl_str("machdep.cpu.leaf7_features") or ""
        ext_features = _sysctl_str("machdep.cpu.extfeatures") or ""
        cpu_features = _sysctl_str("machdep.cpu.features") or ""
        all_feat = (cpu_features + " " + leaf7 + " " + ext_features).lower()
        relevant = {"avx", "avx2", "avx512f", "fma", "sse4.1", "sse4.2", "f16c"}
        features = [f for f in relevant if f in all_feat]

    return {
        "model": model,
        "cores": cores,
        "threads": threads,
        "ram_mb": ram_mb,
        "architecture": arch,
        "features": features,
    }


def _detect_cpu_windows() -> Dict[str, Any]:
    """Detect CPU on Windows via wmic or PowerShell."""
    model = "Unknown CPU"
    cores = 1
    threads = 1
    ram_mb = 8192
    arch = platform.machine()
    features: List[str] = []

    # --- wmic CPU ---
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get",
             "Name,NumberOfCores,NumberOfLogicalProcessors",
             "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4 and parts[1]:
                    model = parts[1]
                    try:
                        cores = int(parts[2])
                    except (ValueError, IndexError):
                        pass
                    try:
                        threads = int(parts[3])
                    except (ValueError, IndexError):
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # PowerShell fallback
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_Processor | "
                 "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | "
                 "ConvertTo-Json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if isinstance(data, list):
                    data = data[0]
                model = data.get("Name", model)
                cores = data.get("NumberOfCores", cores)
                threads = data.get("NumberOfLogicalProcessors", threads)
        except Exception:
            pass

    # --- wmic RAM ---
    try:
        result = subprocess.run(
            ["wmic", "OS", "get", "TotalVisibleMemorySize", "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        ram_kb = int(parts[-1])
                        ram_mb = ram_kb // 1024
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Infer features from model name (best-effort on Windows)
    model_lower = model.lower()
    if "13th gen" in model_lower or "14th gen" in model_lower or "ultra" in model_lower:
        features = ["avx2", "avx512f"]
    elif "12th gen" in model_lower or "11th gen" in model_lower:
        features = ["avx2", "avx512f"]
    elif any(gen in model_lower for gen in ["5th gen", "sapphire", "emerald", "granite"]):
        features = ["avx2", "avx512f", "amx_bf16", "amx_int8"]
    elif "ryzen" in model_lower or "epyc" in model_lower:
        features = ["avx2"]
    elif "apple" in model_lower or "m1" in model_lower or "m2" in model_lower:
        features = ["neon", "fp16"]

    return {
        "model": model,
        "cores": cores,
        "threads": threads,
        "ram_mb": ram_mb,
        "architecture": arch,
        "features": features,
    }


def _cpu_preferred_dtype(cpu: Dict[str, Any]) -> str:
    """Determine preferred dtype for CPU-only inference."""
    features = cpu.get("features", [])
    # Intel AMX — native bf16 matrix ops
    if "amx_bf16" in features or "amx_tile" in features:
        return "bfloat16"
    # AVX512 BF16 extension
    if "avx512_bf16" in features:
        return "bfloat16"
    # ARM FP16 arithmetic (Apple Silicon, Graviton3+)
    if "fp16" in features or "fphp" in features or "asimdhp" in features:
        return "float16"
    return "float32"


# ============================================================================
# GPU DETECTION (OS-FIRST DISPATCH)
# ============================================================================

def _detect_gpus(os_type: str) -> Tuple[List[Dict[str, Any]], str]:
    """Detect GPUs using OS-specific cascade. Returns (devices, brand)."""
    if os_type == "Linux":
        return _detect_gpus_linux()
    elif os_type == "Darwin":
        return _detect_gpus_darwin()
    elif os_type == "Windows":
        return _detect_gpus_windows()
    return [], "unknown"


def _detect_gpus_linux() -> Tuple[List[Dict[str, Any]], str]:
    """
    Linux GPU detection cascade:
    nvidia-smi → rocm → xpu-smi → tt-smi → chinese SMI → lspci → PyTorch
    """
    for detector, brand in [
        (_parse_nvidia_smi, "nvidia"),
        (_parse_rocm, "amd"),
        (_parse_xpu, "intel"),
        (_parse_tenstorrent, "tenstorrent"),
        (_parse_moore_threads, "moore_threads"),
        (_parse_biren, "biren"),
        (_parse_iluvatar, "iluvatar"),
        (_parse_cambricon, "cambricon"),
    ]:
        devices = detector()
        if devices:
            return devices, brand

    # lspci fallback (Linux only)
    devices, brand = _parse_lspci()
    if devices:
        return devices, brand

    # PyTorch backend fallback
    return _detect_torch_fallback()


def _detect_gpus_darwin() -> Tuple[List[Dict[str, Any]], str]:
    """
    macOS GPU detection:
    system_profiler → PyTorch MPS → PyTorch CUDA (Hackintosh edge case)
    """
    devices = _parse_system_profiler()
    if devices:
        brand = devices[0].get("brand", "apple")
        return devices, brand

    # PyTorch MPS fallback
    return _detect_torch_fallback()


def _detect_gpus_windows() -> Tuple[List[Dict[str, Any]], str]:
    """
    Windows GPU detection cascade:
    nvidia-smi → rocm-smi → xpu-smi → WMI/PowerShell
    """
    for detector, brand in [
        (_parse_nvidia_smi, "nvidia"),
        (_parse_rocm, "amd"),
        (_parse_xpu, "intel"),
    ]:
        devices = detector()
        if devices:
            return devices, brand

    # WMI fallback (Windows only)
    devices, brand = _parse_wmi()
    if devices:
        return devices, brand

    # PyTorch backend fallback
    return _detect_torch_fallback()


# ============================================================================
# GPU VENDOR PARSERS (shared by Linux + Windows where CLI is identical)
# ============================================================================

def _parse_nvidia_smi() -> List[Dict[str, Any]]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,compute_cap,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        idx = int(parts[0])
        name = parts[1]
        mem_mb = int(float(parts[2]))
        cc = parts[3]
        pci_bus = parts[4]

        arch = _nvidia_cc_to_arch(cc)
        cc_major = cc.split(".")[0]
        dtypes = _NVIDIA_DTYPE_MAP.get(cc_major, ["float32", "float16"])
        pcie_ver = _detect_pcie_version_sysfs(pci_bus)

        devices.append({
            "index": idx,
            "brand": "nvidia",
            "model": name,
            "memory_mb": mem_mb,
            "compute_capability": cc,
            "supports_dtypes": dtypes,
            "architecture": arch,
            "pcie_version": pcie_ver,
        })

    return devices


def _parse_rocm() -> List[Dict[str, Any]]:
    """Detect AMD GPUs via rocminfo + rocm-smi."""
    devices = _parse_rocminfo()
    if devices:
        return devices
    return _parse_rocmsmi()


def _parse_rocminfo() -> List[Dict[str, Any]]:
    """Detect AMD GPUs via rocminfo (best for architecture detection)."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    current: Dict[str, str] = {}
    idx = 0

    for line in result.stdout.split("\n"):
        line = line.strip()

        if line.startswith("Name:") and "gfx" in line:
            gfx_id = line.split(":")[-1].strip()
            current["gfx_id"] = gfx_id

        elif line.startswith("Marketing Name:") and current:
            current["name"] = line.split(":", 1)[-1].strip()

        elif line.startswith("Device Type:") and "GPU" in line:
            current["is_gpu"] = "true"

        elif line.startswith("Size:") and current.get("is_gpu"):
            size_str = line.split(":")[-1].strip().split("(")[0].strip()
            try:
                mem_bytes = int(size_str)
                if mem_bytes > 1_000_000_000:
                    current["memory_bytes"] = str(mem_bytes)
            except ValueError:
                pass

        elif line == "" and current.get("is_gpu") and current.get("gfx_id"):
            gfx_id = current["gfx_id"]
            name = current.get("name", f"AMD GPU ({gfx_id})")
            mem_bytes = int(current.get("memory_bytes", "0"))
            mem_mb = mem_bytes // (1024 * 1024) if mem_bytes > 0 else 0

            if "hygon" in name.lower():
                current = {}
                continue

            arch = _AMD_GFX_MAP.get(gfx_id, _amd_gfx_to_arch(gfx_id))
            dtypes = _AMD_DTYPE_MAP.get(arch, ["float32", "float16"])

            devices.append({
                "index": idx,
                "brand": "amd",
                "model": name,
                "memory_mb": mem_mb if mem_mb > 0 else 16384,
                "compute_capability": gfx_id,
                "supports_dtypes": dtypes,
                "architecture": arch,
                "pcie_version": "4.0",
            })
            idx += 1
            current = {}

    return devices


def _parse_rocmsmi() -> List[Dict[str, Any]]:
    """Detect AMD GPUs via rocm-smi --showproductname."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    idx = 0
    for line in result.stdout.split("\n"):
        match = re.search(r'GPU\[(\d+)\].*?:\s*(.+)', line)
        if not match:
            continue
        gpu_idx = int(match.group(1))
        name = match.group(2).strip()
        mem_mb = _amd_get_vram_mb(gpu_idx)
        arch = _amd_name_to_arch(name)
        dtypes = _AMD_DTYPE_MAP.get(arch, ["float32", "float16"])

        devices.append({
            "index": gpu_idx,
            "brand": "amd",
            "model": name,
            "memory_mb": mem_mb,
            "compute_capability": "0.0",
            "supports_dtypes": dtypes,
            "architecture": arch,
            "pcie_version": "4.0",
        })
        idx += 1

    return devices


def _parse_xpu() -> List[Dict[str, Any]]:
    """Detect Intel GPUs via xpu-smi or sycl-ls."""
    devices = _parse_xpusmi()
    if devices:
        return devices
    return _parse_sycl()


def _parse_xpusmi() -> List[Dict[str, Any]]:
    """Detect Intel GPUs via xpu-smi discovery."""
    try:
        result = subprocess.run(
            ["xpu-smi", "discovery", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["xpu-smi", "discovery"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    idx = 0

    # Try JSON
    try:
        import json
        data = json.loads(result.stdout)
        for dev in data if isinstance(data, list) else []:
            name = dev.get("device_name", "Intel GPU")
            mem_mb = dev.get("memory_physical_size_byte", 0) // (1024 * 1024)
            pci_addr = dev.get("pci_bdf_address", "")
            pcie_ver = _detect_pcie_version_sysfs(pci_addr) if pci_addr else "4.0"
            arch = _intel_name_to_arch(name)

            devices.append({
                "index": idx,
                "brand": "intel",
                "model": name,
                "memory_mb": mem_mb if mem_mb > 0 else 16384,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": arch,
                "pcie_version": pcie_ver,
            })
            idx += 1
        if devices:
            return devices
    except Exception:
        pass

    # Text parse fallback
    for line in result.stdout.split("\n"):
        line = line.strip()
        if "Device Name" in line and ":" in line:
            name = line.split(":", 1)[-1].strip()
            arch = _intel_name_to_arch(name)
            devices.append({
                "index": idx,
                "brand": "intel",
                "model": name,
                "memory_mb": 16384,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": arch,
                "pcie_version": "4.0",
            })
            idx += 1

    return devices


def _parse_sycl() -> List[Dict[str, Any]]:
    """Detect Intel GPUs via sycl-ls."""
    try:
        result = subprocess.run(
            ["sycl-ls"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    idx = 0
    seen: set[str] = set()
    for line in result.stdout.split("\n"):
        if "gpu" in line.lower() and ("intel" in line.lower() or "arc" in line.lower()):
            name = "Intel GPU"
            match = re.search(r'\]\s+(.+)', line)
            if match:
                name = match.group(1).strip()
                if name in seen:
                    continue
                seen.add(name)

            arch = _intel_name_to_arch(name)
            devices.append({
                "index": idx,
                "brand": "intel",
                "model": name,
                "memory_mb": 16384,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": arch,
                "pcie_version": "4.0",
            })
            idx += 1

    return devices


def _parse_tenstorrent() -> List[Dict[str, Any]]:
    """Detect Tenstorrent accelerators via tt-smi."""
    try:
        result = subprocess.run(
            ["tt-smi", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["tt-smi"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    idx = 0

    # Try JSON
    try:
        import json
        data = json.loads(result.stdout)
        for dev in data.get("devices", data if isinstance(data, list) else []):
            name = dev.get("board_type", dev.get("name", "Tenstorrent Accelerator"))
            arch = "wormhole" if "wormhole" in name.lower() else \
                   "blackhole" if "blackhole" in name.lower() else "grayskull"
            mem_mb = 12288 if "n150" in name.lower() else \
                     24576 if "n300" in name.lower() else 12288

            devices.append({
                "index": idx,
                "brand": "tenstorrent",
                "model": name,
                "memory_mb": mem_mb,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16", "bfloat16"],
                "architecture": arch,
                "pcie_version": "4.0",
            })
            idx += 1
        if devices:
            return devices
    except Exception:
        pass

    # Text parse fallback
    for line in result.stdout.split("\n"):
        line_lower = line.lower()
        if "wormhole" in line_lower or "grayskull" in line_lower or "blackhole" in line_lower:
            name = line.strip().split("|")[1].strip() if "|" in line else line.strip()
            arch = "wormhole" if "wormhole" in line_lower else \
                   "blackhole" if "blackhole" in line_lower else "grayskull"
            mem_mb = 12288 if "n150" in line_lower else \
                     24576 if "n300" in line_lower else 12288

            devices.append({
                "index": idx,
                "brand": "tenstorrent",
                "model": name,
                "memory_mb": mem_mb,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16", "bfloat16"],
                "architecture": arch,
                "pcie_version": "4.0",
            })
            idx += 1

    return devices


def _parse_moore_threads() -> List[Dict[str, Any]]:
    """Detect Moore Threads GPUs via mtsmi."""
    return _run_generic_smi(["mtsmi", "-q"], ["mtsmi"], "moore_threads", "musa")


def _parse_biren() -> List[Dict[str, Any]]:
    """Detect Biren GPUs via brsmi."""
    return _run_generic_smi(["brsmi", "-q"], ["brsmi"], "biren", "birensupa")


def _parse_iluvatar() -> List[Dict[str, Any]]:
    """Detect Iluvatar CoreX GPUs via ixsmi."""
    return _run_generic_smi(["ixsmi", "-q"], ["ixsmi"], "iluvatar", "corex")


def _parse_cambricon() -> List[Dict[str, Any]]:
    """Detect Cambricon MLU via cnmon."""
    return _run_generic_smi(["cnmon", "info"], ["cnmon"], "cambricon", "mlu")


def _run_generic_smi(
    primary_cmd: List[str],
    fallback_cmd: List[str],
    brand: str,
    default_arch: str,
) -> List[Dict[str, Any]]:
    """Run a vendor SMI tool and parse its output."""
    try:
        result = subprocess.run(
            primary_cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            result = subprocess.run(
                fallback_cmd, capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    return _parse_smi_output(result.stdout, brand, default_arch)


# ============================================================================
# MACOS-ONLY: system_profiler
# ============================================================================

def _parse_system_profiler() -> List[Dict[str, Any]]:
    """Detect GPUs on macOS via system_profiler SPDisplaysDataType."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    try:
        import json
        data = json.loads(result.stdout)
    except Exception:
        return []

    displays = data.get("SPDisplaysDataType", [])
    devices = []
    idx = 0

    for gpu in displays:
        name = gpu.get("sppci_model", gpu.get("_name", "Unknown GPU"))
        vendor_str = gpu.get("spdisplays_vendor", "").lower()

        # Determine brand
        if "apple" in vendor_str or "apple" in name.lower():
            brand = "apple"
            arch = "apple_silicon"
            # Apple Silicon: unified memory — get from sysctl
            mem_mb = _detect_apple_memory_mb()
            dtypes = ["float32", "float16"]
            pcie_ver = "N/A"
        elif "amd" in vendor_str or "ati" in vendor_str:
            brand = "amd"
            arch = _amd_name_to_arch(name)
            # VRAM from system_profiler
            vram_str = gpu.get("spdisplays_vram", gpu.get("spdisplays_vram_shared", "0"))
            mem_mb = _parse_mac_vram(vram_str)
            dtypes = ["float32", "float16"]
            pcie_ver = "3.0"
        elif "intel" in vendor_str:
            brand = "intel"
            arch = "xe"
            vram_str = gpu.get("spdisplays_vram_shared", gpu.get("spdisplays_vram", "0"))
            mem_mb = _parse_mac_vram(vram_str)
            dtypes = ["float32"]
            pcie_ver = "3.0"
        else:
            brand = "apple"
            arch = "apple_silicon"
            mem_mb = _detect_apple_memory_mb()
            dtypes = ["float32", "float16"]
            pcie_ver = "N/A"

        # Check Metal support
        metal_support = gpu.get("spdisplays_metal", "")
        if "supported" in str(metal_support).lower():
            if "float16" not in dtypes:
                dtypes.append("float16")

        devices.append({
            "index": idx,
            "brand": brand,
            "model": name,
            "memory_mb": mem_mb if mem_mb > 0 else 8192,
            "compute_capability": "0.0",
            "supports_dtypes": dtypes,
            "architecture": arch,
            "pcie_version": pcie_ver,
        })
        idx += 1

    return devices


# ============================================================================
# WINDOWS-ONLY: WMI / PowerShell
# ============================================================================

def _parse_wmi() -> Tuple[List[Dict[str, Any]], str]:
    """Detect GPUs on Windows via WMI (Win32_VideoController)."""
    try:
        result = subprocess.run(
            ["wmic", "path", "Win32_VideoController", "get",
             "Name,AdapterRAM,PNPDeviceID", "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return _parse_powershell_gpu()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _parse_powershell_gpu()

    devices = []
    brand = "unknown"
    idx = 0

    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        # CSV: Node,AdapterRAM,Name,PNPDeviceID
        adapter_ram = parts[1]
        name = parts[2]
        pnp_id = parts[3] if len(parts) > 3 else ""

        if not name or name == "Name":
            continue

        # Skip Microsoft Basic Display Adapter (software renderer)
        if "microsoft" in name.lower() and "basic" in name.lower():
            continue

        # Detect brand from PNPDeviceID (contains VEN_XXXX)
        ven_match = re.search(r'VEN_([0-9A-Fa-f]{4})', pnp_id)
        if ven_match:
            ven_id = ven_match.group(1).lower()
            detected_brand = _PCI_VENDOR_MAP.get(ven_id, "unknown")
            if detected_brand != "unknown":
                brand = detected_brand

        # Memory
        try:
            mem_bytes = int(adapter_ram)
            mem_mb = mem_bytes // (1024 * 1024) if mem_bytes > 0 else 0
        except (ValueError, TypeError):
            mem_mb = 0

        # WMI AdapterRAM is capped at 4GB (32-bit field) — unreliable for modern GPUs
        if mem_mb > 0 and mem_mb <= 4096 and "rtx" in name.lower():
            mem_mb = 0  # Will use fallback

        devices.append({
            "index": idx,
            "brand": brand if brand != "unknown" else _infer_brand_from_name(name),
            "model": name,
            "memory_mb": mem_mb if mem_mb > 0 else 8192,
            "compute_capability": "0.0",
            "supports_dtypes": ["float32", "float16"],
            "architecture": "unknown",
            "pcie_version": "3.0",
        })
        idx += 1

    return devices, brand


def _parse_powershell_gpu() -> Tuple[List[Dict[str, Any]], str]:
    """Detect GPUs on Windows via PowerShell Get-CimInstance."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM,PNPDeviceID | ConvertTo-Json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return [], "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [], "unknown"

    try:
        import json
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            data = [data]
    except Exception:
        return [], "unknown"

    devices = []
    brand = "unknown"
    idx = 0

    for gpu in data:
        name = gpu.get("Name", "Unknown GPU")
        if "microsoft" in name.lower() and "basic" in name.lower():
            continue

        pnp_id = gpu.get("PNPDeviceID", "")
        ven_match = re.search(r'VEN_([0-9A-Fa-f]{4})', pnp_id)
        if ven_match:
            ven_id = ven_match.group(1).lower()
            detected = _PCI_VENDOR_MAP.get(ven_id, "unknown")
            if detected != "unknown":
                brand = detected

        adapter_ram = gpu.get("AdapterRAM", 0)
        mem_mb = int(adapter_ram) // (1024 * 1024) if adapter_ram else 0

        devices.append({
            "index": idx,
            "brand": brand if brand != "unknown" else _infer_brand_from_name(name),
            "model": name,
            "memory_mb": mem_mb if mem_mb > 0 else 8192,
            "compute_capability": "0.0",
            "supports_dtypes": ["float32", "float16"],
            "architecture": "unknown",
            "pcie_version": "3.0",
        })
        idx += 1

    return devices, brand


# ============================================================================
# LINUX-ONLY: lspci FALLBACK
# ============================================================================

def _parse_lspci() -> Tuple[List[Dict[str, Any]], str]:
    """Detect GPUs via lspci PCI vendor IDs + sysfs (Linux only)."""
    try:
        result = subprocess.run(
            ["lspci", "-Dnn"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return [], "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [], "unknown"

    devices = []
    brand = "unknown"
    idx = 0

    for line in result.stdout.split("\n"):
        if not any(cls in line for cls in ["VGA", "3D controller", "Processing accelerator"]):
            continue

        vid_match = re.search(r'\[([0-9a-f]{4}):([0-9a-f]{4})\]', line)
        if not vid_match:
            continue
        vendor_id = vid_match.group(1)
        detected_brand = _PCI_VENDOR_MAP.get(vendor_id)
        if not detected_brand:
            continue

        if devices and brand != detected_brand:
            continue

        brand = detected_brand

        pci_addr_match = re.match(r'([0-9a-f:.]+)', line)
        pci_addr = pci_addr_match.group(1) if pci_addr_match else ""

        name_match = re.search(r'\]:\s*(.+?)\s*\[', line)
        name = name_match.group(1).strip() if name_match else f"{brand} GPU"

        pcie_ver = _detect_pcie_version_sysfs(pci_addr) if pci_addr else "4.0"

        mem_mb = 0
        if brand == "amd":
            sysfs_mem = Path(f"/sys/bus/pci/devices/{pci_addr}/mem_info_vram_total")
            if sysfs_mem.exists():
                try:
                    mem_mb = int(sysfs_mem.read_text().strip()) // (1024 * 1024)
                except (ValueError, PermissionError):
                    pass

        devices.append({
            "index": idx,
            "brand": brand,
            "model": name,
            "memory_mb": mem_mb if mem_mb > 0 else 16384,
            "compute_capability": "0.0",
            "supports_dtypes": ["float32", "float16"],
            "architecture": "unknown",
            "pcie_version": pcie_ver,
        })
        idx += 1

    return devices, brand


# ============================================================================
# GENERIC SMI PARSER (shared by Moore Threads, Biren, Iluvatar, Cambricon)
# ============================================================================

def _parse_smi_output(
    output: str,
    brand: str,
    default_arch: str,
) -> List[Dict[str, Any]]:
    """
    Parse nvidia-smi-style output from vendor SMI tools.
    Most Chinese GPU vendors model their CLI after nvidia-smi.
    """
    devices = []
    idx = 0
    current_name = ""
    current_mem_mb = 0

    for line in output.split("\n"):
        line_stripped = line.strip()

        name_match = re.search(
            r'(?:Product Name|GPU Name|Device Name|Card Name)\s*:\s*(.+)',
            line_stripped, re.IGNORECASE,
        )
        if name_match:
            current_name = name_match.group(1).strip()

        mem_match = re.search(
            r'(?:Total|FB Memory|Memory Total)\s*:\s*(\d+)\s*(?:MiB|MB)',
            line_stripped, re.IGNORECASE,
        )
        if mem_match:
            current_mem_mb = int(mem_match.group(1))

        mem_bytes_match = re.search(
            r'(?:Total|Memory)\s*:\s*(\d+)\s*(?:bytes|B)\b',
            line_stripped, re.IGNORECASE,
        )
        if mem_bytes_match:
            val = int(mem_bytes_match.group(1))
            if val > 1_000_000_000:
                current_mem_mb = val // (1024 * 1024)

        mem_gb_match = re.search(
            r'(?:Total|Memory|VRAM)\s*:\s*(\d+)\s*(?:GiB|GB)',
            line_stripped, re.IGNORECASE,
        )
        if mem_gb_match:
            current_mem_mb = int(mem_gb_match.group(1)) * 1024

        table_match = re.match(r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|', line_stripped)
        if table_match:
            row_name = table_match.group(2).strip()
            if row_name and not row_name.startswith("-") and not row_name.startswith("="):
                if not current_name:
                    current_name = row_name

        gpu_boundary = re.search(r'GPU\s*\[?(\d+)', line_stripped, re.IGNORECASE)
        if gpu_boundary and current_name and idx > 0:
            devices.append({
                "index": idx - 1,
                "brand": brand,
                "model": current_name,
                "memory_mb": current_mem_mb if current_mem_mb > 0 else 16384,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": default_arch,
                "pcie_version": "4.0",
            })
            current_name = ""
            current_mem_mb = 0

        if gpu_boundary:
            idx = int(gpu_boundary.group(1)) + 1

    if current_name:
        devices.append({
            "index": max(0, idx - 1) if idx > 0 else 0,
            "brand": brand,
            "model": current_name,
            "memory_mb": current_mem_mb if current_mem_mb > 0 else 16384,
            "compute_capability": "0.0",
            "supports_dtypes": ["float32", "float16"],
            "architecture": default_arch,
            "pcie_version": "4.0",
        })

    return devices


# ============================================================================
# PYTORCH BACKEND FALLBACK (last resort — catches any GPU PyTorch can see)
# ============================================================================

def _detect_torch_fallback() -> Tuple[List[Dict[str, Any]], str]:
    """Last-resort detection via PyTorch backends."""
    try:
        import torch
    except ImportError:
        return [], "unknown"

    # CUDA (NVIDIA, AMD ROCm, Hygon DTK, Iluvatar IXUCA)
    if torch.cuda.is_available():
        devices = []
        is_hip = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None  # type: ignore[union-attr]
        brand = "amd" if is_hip else "nvidia"

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = props.name
            mem_mb = props.total_memory // (1024 * 1024)
            cc = f"{props.major}.{props.minor}"
            name_lower = name.lower()

            if "hygon" in name_lower:
                brand = "amd"
                arch = "hygon_dcu"
                dtypes = ["float32", "float16"]
                cc = getattr(props, 'gcnArchName', "0.0")
            elif "iluvatar" in name_lower or "bi-v" in name_lower:
                brand = "iluvatar"
                arch = "corex"
                dtypes = ["float32", "float16"]
                cc = "0.0"
            elif is_hip:
                arch = _amd_name_to_arch(name)
                dtypes = _AMD_DTYPE_MAP.get(arch, ["float32", "float16"])
                cc = getattr(props, 'gcnArchName', cc)
            else:
                arch = _nvidia_cc_to_arch(cc)
                cc_major = str(props.major)
                dtypes = _NVIDIA_DTYPE_MAP.get(cc_major, ["float32", "float16"])

            devices.append({
                "index": i,
                "brand": brand,
                "model": name,
                "memory_mb": mem_mb,
                "compute_capability": cc,
                "supports_dtypes": dtypes,
                "architecture": arch,
                "pcie_version": "3.0",
            })
        return devices, brand

    # Intel XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():  # type: ignore[attr-defined]
        devices = []
        for i in range(torch.xpu.device_count()):  # type: ignore[attr-defined]
            props = torch.xpu.get_device_properties(i)  # type: ignore[attr-defined]
            name = props.name
            mem_mb = props.total_memory // (1024 * 1024)
            arch = _intel_name_to_arch(name)

            devices.append({
                "index": i,
                "brand": "intel",
                "model": name,
                "memory_mb": mem_mb,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": arch,
                "pcie_version": "4.0",
            })
        return devices, "intel"

    # Moore Threads (torch.musa)
    if hasattr(torch, 'musa') and torch.musa.is_available():  # type: ignore[attr-defined]
        devices = []
        for i in range(torch.musa.device_count()):  # type: ignore[attr-defined]
            props = torch.musa.get_device_properties(i)  # type: ignore[attr-defined]
            name = props.name
            mem_mb = props.total_memory // (1024 * 1024)
            devices.append({
                "index": i,
                "brand": "moore_threads",
                "model": name,
                "memory_mb": mem_mb,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": "musa",
                "pcie_version": "4.0",
            })
        return devices, "moore_threads"

    # Cambricon MLU (torch.mlu)
    if hasattr(torch, 'mlu') and torch.mlu.is_available():  # type: ignore[attr-defined]
        devices = []
        for i in range(torch.mlu.device_count()):  # type: ignore[attr-defined]
            name = torch.mlu.get_device_name(i)  # type: ignore[attr-defined]
            props = torch.mlu.get_device_properties(i)  # type: ignore[attr-defined]
            mem_mb = props.total_memory // (1024 * 1024)
            devices.append({
                "index": i,
                "brand": "cambricon",
                "model": name,
                "memory_mb": mem_mb,
                "compute_capability": "0.0",
                "supports_dtypes": ["float32", "float16"],
                "architecture": "mlu",
                "pcie_version": "4.0",
            })
        return devices, "cambricon"

    # Apple MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices = [{
            "index": 0,
            "brand": "apple",
            "model": _detect_apple_chip(),
            "memory_mb": _detect_apple_memory_mb(),
            "compute_capability": "0.0",
            "supports_dtypes": ["float32", "float16"],
            "architecture": "apple_silicon",
            "pcie_version": "N/A",
        }]
        return devices, "apple"

    return [], "unknown"


# ============================================================================
# INTERCONNECT DETECTION
# ============================================================================

def _detect_interconnect(
    devices: List[Dict[str, Any]],
    brand: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect GPU interconnect topology. Returns (interconnect, pcie_fallback)."""
    num_gpus = len(devices)
    pcie_ver = devices[0].get("pcie_version", "3.0") if devices else "3.0"
    pcie_bw = _PCIE_BW_MAP.get(pcie_ver, 32)

    pcie_fallback = {
        "version": pcie_ver,
        "lanes": 16,
        "bandwidth_gbps": pcie_bw,
    }

    if num_gpus <= 1:
        return {"groups": []}, pcie_fallback

    members = list(range(num_gpus))

    if brand == "nvidia":
        return _detect_nvidia_interconnect(devices, members, pcie_fallback)
    elif brand == "amd":
        return _detect_amd_interconnect(members, pcie_fallback)
    elif brand == "tenstorrent":
        return _detect_tenstorrent_interconnect(members, pcie_fallback)
    elif brand == "cambricon":
        return _detect_cambricon_interconnect(members, pcie_fallback)

    return {
        "groups": [{
            "index": 0,
            "members": members,
            "type": "cpu_bounce",
            "tech": "pcie",
            "bandwidth_gbps": pcie_bw,
        }],
    }, pcie_fallback


def _detect_nvidia_interconnect(
    devices: List[Dict[str, Any]],
    members: List[int],
    pcie_fallback: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect NVIDIA NVLink topology via nvidia-smi topo -m."""
    num_gpus = len(devices)

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi topo failed")
    except Exception:
        return _pcie_interconnect(members, pcie_fallback), pcie_fallback

    nvlink_pairs: List[Tuple[int, int, int]] = []
    has_nvlink = False

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        match = re.match(r'GPU(\d+)', parts[0])
        if not match:
            continue
        gpu_a = int(match.group(1))

        for col_idx, val in enumerate(parts[1:], start=0):
            if val.startswith("NV"):
                has_nvlink = True
                num_links = int(val[2:]) if len(val) > 2 and val[2:].isdigit() else 1
                gpu_b = col_idx
                if gpu_a < gpu_b:
                    nvlink_pairs.append((gpu_a, gpu_b, num_links))

    if not has_nvlink:
        return _pcie_interconnect(members, pcie_fallback), pcie_fallback

    cc = devices[0].get("compute_capability", "7.0")
    cc_major = int(cc.split(".")[0])
    nvlink_gen = max(1, cc_major - 5)
    bw_per_link = _NVLINK_BW_PER_LINK.get(nvlink_gen, 50)

    # Aggregate NVLink bandwidth per GPU
    links_per_gpu: Dict[int, int] = {}
    for a, b, n in nvlink_pairs:
        links_per_gpu[a] = links_per_gpu.get(a, 0) + n
        links_per_gpu[b] = links_per_gpu.get(b, 0) + n
    max_links = max(links_per_gpu.values()) if links_per_gpu else 1
    total_bw = max_links * bw_per_link

    all_connected = all(
        any(a == i and b == j for a, b, _ in nvlink_pairs)
        for i in range(num_gpus) for j in range(i + 1, num_gpus)
    )

    topo_type = "full_mesh" if all_connected and num_gpus > 1 else \
                "nvlink_p2p" if num_gpus == 2 else "full_mesh"

    return {
        "groups": [{
            "index": 0,
            "members": members,
            "type": topo_type,
            "tech": "nvlink",
            "bandwidth_gbps": total_bw,
        }],
    }, pcie_fallback


def _detect_amd_interconnect(
    members: List[int],
    pcie_fallback: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect AMD Infinity Fabric / xGMI interconnect."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtopo"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "XGMI" in result.stdout.upper():
            return {
                "groups": [{
                    "index": 0,
                    "members": members,
                    "type": "full_mesh",
                    "tech": "xgmi",
                    "bandwidth_gbps": 200,
                }],
            }, pcie_fallback
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    for card in sorted(Path("/sys/class/drm/").glob("card*")):
        hive_path = card / "device" / "xgmi_hive_id"
        if hive_path.exists():
            try:
                hive_id = hive_path.read_text().strip()
                if hive_id and hive_id != "0":
                    return {
                        "groups": [{
                            "index": 0,
                            "members": members,
                            "type": "full_mesh",
                            "tech": "xgmi",
                            "bandwidth_gbps": 200,
                        }],
                    }, pcie_fallback
            except (PermissionError, ValueError):
                pass

    return _pcie_interconnect(members, pcie_fallback), pcie_fallback


def _detect_tenstorrent_interconnect(
    members: List[int],
    pcie_fallback: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect Tenstorrent ethernet mesh interconnect."""
    try:
        result = subprocess.run(
            ["tt-topology"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "eth" in result.stdout.lower():
            return {
                "groups": [{
                    "index": 0,
                    "members": members,
                    "type": "full_mesh",
                    "tech": "ethernet_mesh",
                    "bandwidth_gbps": 100,
                }],
            }, pcie_fallback
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _pcie_interconnect(members, pcie_fallback), pcie_fallback


def _detect_cambricon_interconnect(
    members: List[int],
    pcie_fallback: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Detect Cambricon MLULink interconnect."""
    try:
        result = subprocess.run(
            ["cnmon", "mlulink", "-t"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "mlulink" in result.stdout.lower():
            return {
                "groups": [{
                    "index": 0,
                    "members": members,
                    "type": "full_mesh",
                    "tech": "mlulink",
                    "bandwidth_gbps": 200,
                }],
            }, pcie_fallback
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _pcie_interconnect(members, pcie_fallback), pcie_fallback


def _pcie_interconnect(
    members: List[int],
    pcie_fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """Default PCIe-only interconnect group."""
    return {
        "groups": [{
            "index": 0,
            "members": members,
            "type": "cpu_bounce",
            "tech": "pcie",
            "bandwidth_gbps": pcie_fallback.get("bandwidth_gbps", 32),
        }],
    }


# ============================================================================
# SYSTEM VENDOR DETECTION (per-OS)
# ============================================================================

def _detect_system_vendor(os_type: str) -> str:
    """Detect system/motherboard vendor."""
    if os_type == "Darwin":
        return "apple"

    if os_type == "Linux":
        return _detect_vendor_linux()

    if os_type == "Windows":
        return _detect_vendor_windows()

    return "custom"


def _detect_vendor_linux() -> str:
    """Detect system vendor on Linux from DMI data."""
    paths = [
        "/sys/class/dmi/id/sys_vendor",
        "/sys/class/dmi/id/board_vendor",
        "/sys/class/dmi/id/chassis_vendor",
    ]
    known_vendors = {
        "dell": "dell", "hp": "hp", "hewlett": "hp",
        "lenovo": "lenovo", "supermicro": "supermicro",
        "asus": "asus", "gigabyte": "gigabyte", "msi": "msi",
        "apple": "apple", "inspur": "inspur", "huawei": "huawei",
        "sugon": "sugon", "h3c": "h3c",
    }
    for path in paths:
        try:
            with open(path) as f:
                raw = f.read().strip().lower()
            for keyword, name in known_vendors.items():
                if keyword in raw:
                    return name
            if raw:
                return raw
        except (FileNotFoundError, PermissionError):
            continue
    return "custom"


def _detect_vendor_windows() -> str:
    """Detect system vendor on Windows via wmic."""
    try:
        result = subprocess.run(
            ["wmic", "computersystem", "get", "Manufacturer", "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[-1] and parts[-1] != "Manufacturer":
                    raw = parts[-1].lower()
                    known = {
                        "dell": "dell", "hp": "hp", "hewlett": "hp",
                        "lenovo": "lenovo", "asus": "asus",
                        "gigabyte": "gigabyte", "msi": "msi",
                        "apple": "apple", "microsoft": "microsoft",
                    }
                    for keyword, name in known.items():
                        if keyword in raw:
                            return name
                    return raw
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "custom"


# ============================================================================
# UNIVERSAL HELPERS
# ============================================================================

def _nvidia_cc_to_arch(cc: str) -> str:
    """Convert NVIDIA compute capability to architecture name."""
    if cc in _NVIDIA_ARCH_REFINE:
        return _NVIDIA_ARCH_REFINE[cc]
    major = cc.split(".")[0]
    return _NVIDIA_ARCH_MAP.get(major, "unknown")


def _amd_gfx_to_arch(gfx_id: str) -> str:
    """Map unknown gfx IDs to architecture by prefix pattern."""
    if gfx_id.startswith("gfx9"):
        major = int(gfx_id[3:5]) if len(gfx_id) >= 5 else 90
        if major >= 94:
            return "cdna3"
        elif major >= 90:
            return "cdna2" if "a" in gfx_id else "cdna"
        return "vega20"
    elif gfx_id.startswith("gfx12"):
        return "rdna4"
    elif gfx_id.startswith("gfx11"):
        return "rdna3"
    elif gfx_id.startswith("gfx10"):
        return "rdna2"
    return "rdna"


def _amd_name_to_arch(name: str) -> str:
    """Infer AMD architecture from GPU marketing name."""
    n = name.lower()
    if "mi300" in n:
        return "cdna3"
    elif "mi250" in n or "mi210" in n:
        return "cdna2"
    elif "mi100" in n:
        return "cdna"
    elif any(x in n for x in ["7900", "7800", "7700", "7600"]):
        return "rdna3"
    elif any(x in n for x in ["9070", "9060"]):
        return "rdna4"
    elif any(x in n for x in ["6900", "6800", "6700"]):
        return "rdna2"
    return "rdna"


def _amd_get_vram_mb(gpu_idx: int) -> int:
    """Get AMD GPU VRAM via sysfs or rocm-smi."""
    for card in sorted(Path("/sys/class/drm/").glob("card*")):
        vram_path = card / "device" / "mem_info_vram_total"
        if vram_path.exists():
            try:
                vram_bytes = int(vram_path.read_text().strip())
                if vram_bytes > 1_000_000_000:
                    return vram_bytes // (1024 * 1024)
            except (ValueError, PermissionError):
                pass

    try:
        result = subprocess.run(
            ["rocm-smi", "-d", str(gpu_idx), "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Total" in line:
                    nums = re.findall(r'(\d+)', line)
                    if nums:
                        val = int(nums[-1])
                        if val > 1_000_000:
                            return val // (1024 * 1024)
    except Exception:
        pass

    return 16384


def _intel_name_to_arch(name: str) -> str:
    """Infer Intel GPU architecture from name."""
    n = name.lower()
    if any(x in n for x in ["b580", "b570", "battlemage"]):
        return "battlemage"
    elif any(x in n for x in ["a770", "a750", "a580", "a380", "alchemist"]):
        return "alchemist"
    elif "max" in n or "ponte" in n or "pvc" in n:
        return "ponte_vecchio"
    elif "flex" in n:
        return "flex"
    return "xe"


def _detect_pcie_version_sysfs(pci_addr: str) -> str:
    """Detect PCIe version from Linux sysfs for any vendor."""
    if not pci_addr or pci_addr == "N/A":
        return "3.0"
    try:
        addr = pci_addr.lower().strip()
        sysfs_path = Path(f"/sys/bus/pci/devices/{addr}/current_link_speed")
        if sysfs_path.exists():
            speed = sysfs_path.read_text().strip()
            gt_match = re.search(r'([\d.]+)\s*GT/s', speed)
            if gt_match:
                gt_val = gt_match.group(1)
                return _PCIE_SPEED_MAP.get(gt_val, "3.0")
    except Exception:
        pass
    return "3.0"


def _detect_apple_chip() -> str:
    """Detect Apple Silicon chip model."""
    return _sysctl_str("machdep.cpu.brand_string") or \
           _sysctl_str("hw.model") or "Apple Silicon"


def _detect_apple_memory_mb() -> int:
    """Detect Apple Silicon unified memory (shared CPU+GPU)."""
    memsize = _sysctl_int("hw.memsize")
    if memsize:
        return memsize // (1024 * 1024)
    return 8192


def _parse_mac_vram(vram_str: str) -> int:
    """Parse macOS VRAM string like '4 GB' or '1536 MB'."""
    if not vram_str:
        return 0
    m = re.search(r'(\d+)\s*(GB|MB)', str(vram_str), re.IGNORECASE)
    if m:
        val = int(m.group(1))
        unit = m.group(2).upper()
        return val * 1024 if unit == "GB" else val
    return 0


def _infer_brand_from_name(name: str) -> str:
    """Infer GPU brand from device name string."""
    n = name.lower()
    if any(x in n for x in ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"]):
        return "nvidia"
    elif any(x in n for x in ["radeon", "amd", "instinct"]):
        return "amd"
    elif any(x in n for x in ["intel", "arc", "iris"]):
        return "intel"
    return "unknown"


def _shorten_gpu_name(name: str) -> str:
    """Shorten GPU name for profile ID."""
    name = name.lower()
    for prefix in ["nvidia ", "geforce ", "tesla ", "quadro ", "amd ", "intel ",
                    "radeon ", "instinct ", "arc ", "cambricon ", "moore threads ",
                    "iluvatar ", "biren ", "tenstorrent "]:
        name = name.replace(prefix, "")
    for suffix in ["-sxm2", "-sxm4", "-sxm5", "-sxm", "-pcie", "-hbm2",
                    "-hbm2e", "-hbm3", " ti", " super"]:
        name = name.replace(suffix, "")
    name = re.sub(r'[^a-z0-9]+', '-', name).strip('-')
    return name


def _shorten_cpu_name(name: str) -> str:
    """Shorten CPU name for profile ID."""
    name = name.lower()
    for prefix in ["intel(r) ", "amd ", "apple ", "core(tm) ", "xeon(r) ",
                    "processor ", "cpu "]:
        name = name.replace(prefix, "")
    # Remove frequency
    name = re.sub(r'@\s*[\d.]+\s*[gm]hz', '', name)
    name = re.sub(r'[^a-z0-9]+', '-', name).strip('-')
    # Truncate to reasonable length
    if len(name) > 20:
        name = name[:20].rstrip('-')
    return name


def _sysctl_str(key: str) -> Optional[str]:
    """Read a macOS sysctl string value."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _sysctl_int(key: str) -> Optional[int]:
    """Read a macOS sysctl integer value."""
    val = _sysctl_str(key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def _build_notes(
    cpu: Dict[str, Any],
    devices: List[Dict[str, Any]],
    brand: str,
    vendor: str,
    interconnect: Dict[str, Any],
) -> str:
    """Build human-readable notes string."""
    lines = []

    num_gpus = len(devices)

    if num_gpus == 0:
        lines.append(f"CPU-only system: {cpu.get('model', 'Unknown CPU')}")
        ram_gb = round(cpu.get('ram_mb', 0) / 1024, 1)
        lines.append(f"RAM: {ram_gb} GB, {cpu.get('cores', 1)} cores / {cpu.get('threads', 1)} threads")
        features = cpu.get("features", [])
        if features:
            lines.append(f"CPU features: {', '.join(features)}")
        lines.append("No GPU detected. Models run on CPU.")
    else:
        if num_gpus == 1:
            lines.append(f"1 x {devices[0].get('model', 'GPU')}")
        else:
            models: Dict[str, int] = {}
            for d in devices:
                m = d.get("model", "GPU")
                models[m] = models.get(m, 0) + 1
            parts = [f"{count} x {model}" for model, count in models.items()]
            lines.append(" + ".join(parts))

        lines.append(f"GPU brand: {brand}")
        arch = devices[0].get("architecture", "unknown")
        lines.append(f"Architecture: {arch}")

        # CPU info for offload context
        ram_gb = round(cpu.get('ram_mb', 0) / 1024, 1)
        lines.append(f"CPU: {cpu.get('model', 'Unknown')} ({cpu.get('cores', '?')} cores, {ram_gb} GB RAM)")

        groups = interconnect.get("groups", [])
        if groups:
            g = groups[0]
            tech = g.get("tech", "none")
            if tech not in ("none", "pcie"):
                bw = g.get("bandwidth_gbps", 0)
                lines.append(f"Interconnect: {tech} ({bw} GB/s)")

    if vendor != "custom":
        lines.append(f"System vendor: {vendor}")

    lines.append("Auto-generated by NeuroBrix hardware detection.")
    return "\n".join(lines)
