"""
Kernel Specification for NeuroBrix.

Définit la structure d'un kernel et comment l'identifier uniquement.
UNIVERSEL: Supporte NVIDIA, AMD, Apple.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Set
from enum import Enum


class GpuVendor(Enum):
    """Vendors GPU supportés."""
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"


# =============================================================================
# ARCHITECTURE DEFINITIONS - TOUTES LES ARCHITECTURES CONNUES
# =============================================================================

ARCHITECTURE_SPECS = {
    # -------------------------------------------------------------------------
    # NVIDIA
    # -------------------------------------------------------------------------
    "volta": {
        "vendor": GpuVendor.NVIDIA,
        "compute_capability": "7.0",
        "supported_dtypes": {"float32", "float16"},
        "gpus": ["V100"],
        "num_warps": 4,
        "num_stages": 2,
        "block_m": 64,
        "block_n": 64,
        "block_k": 32,
    },
    "turing": {
        "vendor": GpuVendor.NVIDIA,
        "compute_capability": "7.5",
        "supported_dtypes": {"float32", "float16"},
        "gpus": ["RTX 2080", "RTX 2080 Ti", "T4", "GTX 1660"],
        "num_warps": 4,
        "num_stages": 2,
        "block_m": 64,
        "block_n": 64,
        "block_k": 32,
    },
    "ampere": {
        "vendor": GpuVendor.NVIDIA,
        "compute_capability": "8.0",
        "supported_dtypes": {"float32", "float16", "bfloat16", "tf32"},
        "gpus": ["A100", "A30"],
        "num_warps": 8,
        "num_stages": 3,
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
    },
    "ampere_consumer": {
        "vendor": GpuVendor.NVIDIA,
        "compute_capability": "8.6",
        "supported_dtypes": {"float32", "float16", "bfloat16"},
        "gpus": ["RTX 3090", "RTX 3080", "RTX 3070", "A10"],
        "num_warps": 8,
        "num_stages": 3,
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
    },
    "ada": {
        "vendor": GpuVendor.NVIDIA,
        "compute_capability": "8.9",
        "supported_dtypes": {"float32", "float16", "bfloat16", "float8"},
        "gpus": ["RTX 4090", "RTX 4080", "RTX 4070", "L40", "L4"],
        "num_warps": 8,
        "num_stages": 4,
        "block_m": 128,
        "block_n": 256,
        "block_k": 64,
    },
    "hopper": {
        "vendor": GpuVendor.NVIDIA,
        "compute_capability": "9.0",
        "supported_dtypes": {"float32", "float16", "bfloat16", "float8"},
        "gpus": ["H100", "H200"],
        "num_warps": 8,
        "num_stages": 4,
        "block_m": 128,
        "block_n": 256,
        "block_k": 64,
    },

    # -------------------------------------------------------------------------
    # AMD (ROCm)
    # -------------------------------------------------------------------------
    "rdna2": {
        "vendor": GpuVendor.AMD,
        "compute_capability": "gfx1030",
        "supported_dtypes": {"float32", "float16"},
        "gpus": ["RX 6800", "RX 6800 XT", "RX 6900 XT"],
        "wavefront_size": 32,
        "num_cus": 60,
    },
    "rdna3": {
        "vendor": GpuVendor.AMD,
        "compute_capability": "gfx1100",
        "supported_dtypes": {"float32", "float16", "bfloat16"},
        "gpus": ["RX 7900 XTX", "RX 7900 XT", "RX 7800 XT"],
        "wavefront_size": 32,
        "num_cus": 96,
    },
    "cdna2": {
        "vendor": GpuVendor.AMD,
        "compute_capability": "gfx90a",
        "supported_dtypes": {"float32", "float16", "bfloat16"},
        "gpus": ["MI210", "MI250", "MI250X"],
        "wavefront_size": 64,
        "num_cus": 104,
    },
    "cdna3": {
        "vendor": GpuVendor.AMD,
        "compute_capability": "gfx940",
        "supported_dtypes": {"float32", "float16", "bfloat16", "float8"},
        "gpus": ["MI300X", "MI300A"],
        "wavefront_size": 64,
        "num_cus": 304,
    },

    # -------------------------------------------------------------------------
    # APPLE (Metal)
    # -------------------------------------------------------------------------
    "m1": {
        "vendor": GpuVendor.APPLE,
        "compute_capability": "apple7",
        "supported_dtypes": {"float32", "float16"},
        "gpus": ["M1", "M1 Pro", "M1 Max", "M1 Ultra"],
        "simd_width": 32,
        "max_threads_per_group": 1024,
    },
    "m2": {
        "vendor": GpuVendor.APPLE,
        "compute_capability": "apple8",
        "supported_dtypes": {"float32", "float16"},
        "gpus": ["M2", "M2 Pro", "M2 Max", "M2 Ultra"],
        "simd_width": 32,
        "max_threads_per_group": 1024,
    },
    "m3": {
        "vendor": GpuVendor.APPLE,
        "compute_capability": "apple9",
        "supported_dtypes": {"float32", "float16"},
        "gpus": ["M3", "M3 Pro", "M3 Max"],
        "simd_width": 32,
        "max_threads_per_group": 1024,
        "supports_ray_tracing": True,
    },
}


def get_architecture_for_gpu(gpu_name: str) -> Optional[str]:
    """
    Trouve l'architecture pour un nom de GPU.

    Args:
        gpu_name: Nom du GPU (ex: "RTX 4090", "V100", "RX 7900 XTX")

    Returns:
        Nom de l'architecture ou None si inconnu
    """
    gpu_name_lower = gpu_name.lower()

    for arch_name, arch_spec in ARCHITECTURE_SPECS.items():
        for known_gpu in arch_spec["gpus"]:
            if known_gpu.lower() in gpu_name_lower or gpu_name_lower in known_gpu.lower():
                return arch_name

    return None


def list_supported_architectures() -> dict:
    """Liste toutes les architectures supportées par vendor."""
    result = {vendor: [] for vendor in GpuVendor}

    for arch_name, arch_spec in ARCHITECTURE_SPECS.items():
        result[arch_spec["vendor"]].append({
            "name": arch_name,
            "gpus": arch_spec["gpus"],
            "dtypes": list(arch_spec["supported_dtypes"]),
        })

    return result
