"""Shared autotune configurations for Triton kernels.

V100 safety: _safe_num_stages() clamps num_stages to 2 on pre-Ampere GPUs.
V100 (sm_70) lacks cp.async — Triton's software pipelining with num_stages > 2
emits async copy instructions that cause CUDA_ERROR_MISALIGNED_ADDRESS on Volta.
"""

import inspect
import warnings
from typing import Dict, List

import torch
import triton
from triton import next_power_of_2


# ---------------------------------------------------------------------------
# Autotune compatibility across Triton versions
# ---------------------------------------------------------------------------

_AUTOTUNE_DROPPED: set = set()


def nbx_autotune(*args, **kwargs):
    """`triton.autotune`, minus any keyword the installed Triton lacks.

    NeuroBrix passes `cache_results=True` so autotune results persist to
    disk instead of being re-measured every process. That keyword arrived
    in a later Triton than the one `pip install torch --index-url .../cu121`
    resolves (Triton 3.1 for torch 2.5), and passing it there raises

        TypeError: autotune() got an unexpected keyword argument 'cache_results'

    at the FIRST kernel launch — which made a clean install of NeuroBrix
    unable to run a single model on any GPU whose torch build ships an
    older Triton. That is what this shim exists to stop.

    The degradation is real but bounded, and it is announced rather than
    hidden: `cache_results` is a caching optimisation, so dropping it costs
    one autotune sweep per process and changes no result. Anything that
    would change a RESULT must never be dropped this way — it would be a
    silent fallback, which this engine does not do.
    """
    import triton

    try:
        supported = set(inspect.signature(triton.autotune).parameters)
    except (TypeError, ValueError):
        return triton.autotune(*args, **kwargs)

    unsupported = {k for k in kwargs if k not in supported}
    if unsupported:
        for name in sorted(unsupported - _AUTOTUNE_DROPPED):
            _AUTOTUNE_DROPPED.add(name)
            warnings.warn(
                f"Triton {getattr(triton, '__version__', '?')} does not support "
                f"@triton.autotune({name}=...); continuing without it. "
                f"Autotune results will be re-measured once per process instead "
                f"of persisted to disk (slower first launch, identical results). "
                f"Triton 3.6+ restores it.",
                RuntimeWarning,
                stacklevel=2,
            )
        kwargs = {k: v for k, v in kwargs.items() if k not in unsupported}
    return triton.autotune(*args, **kwargs)


def _safe_num_stages(n: int) -> int:
    """Clamp num_stages to the executing architecture's pipelining budget.

    The budget is a HARDWARE CAPABILITY and is read from
    `config/vendors/<vendor>/<arch>.yml` `pipelining.max_num_stages`
    (R23/R24) — never decided in code, never queried from the driver in a
    hot path. Volta's 2 is measured: sm_70 has no cp.async, and Triton's
    pipelining above 2 stages emits async copies that fault with
    CUDA_ERROR_MISALIGNED_ADDRESS in any kernel using tl.dot. Ampere and
    Hopper carry a cap above the widest config space, so it never binds.
    The CDNA profiles carry a conservative 2 pending first light.

    Fallback path — a bare process with no hardware profile (unit tests):
    the driver is the only signal available. Note that
    `get_device_capability()` on ROCm returns the *gfx* version
    (gfx90a -> (9, 0)), which is not an NVIDIA compute capability and must
    never be compared against 8 — doing so silently cleared unvalidated
    pipelining on every CDNA card. ROCm is clamped explicitly instead.
    """
    try:
        from neurobrix.kernels.wrappers import _arch_param, get_hardware_profile
        if get_hardware_profile() is not None:
            cap = _arch_param("pipelining", "max_num_stages", None)
            if cap:
                return min(n, int(cap))
    except Exception:
        pass

    if not torch.cuda.is_available():
        return n
    if getattr(torch.version, "hip", None):
        return min(n, 2)
    return 2 if torch.cuda.get_device_capability()[0] < 8 else n


def element_wise_configs() -> List[triton.Config]:
    """Autotune configs for element-wise (1D) kernels."""
    return [
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ]


def warps_configs() -> List[triton.Config]:
    """Autotune configs sweeping warp counts (for row-wise kernels)."""
    return [triton.Config({}, num_warps=2**i) for i in range(6)]


def batch_block_heuristic(args: Dict) -> int:
    """Heuristic for batch block size in softmax/norm kernels.

    For small feature dims (< 64) and large batch, processes multiple rows
    per program for efficiency.
    """
    return (min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128)
            if args['feat_dim'] < 64 else 1)


def reduction_configs() -> List[triton.Config]:
    """Autotune configs for row-wise reduction kernels (BLOCK_M x BLOCK_N)."""
    return [
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512}, num_warps=4),
    ]


def matmul_configs() -> List[triton.Config]:
    """Autotune configs for matmul kernels (CUDA).

    num_stages clamped to 2 on V100 via _safe_num_stages().
    """
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(3), num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(4), num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(5), num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=_safe_num_stages(5), num_warps=2),
    ]
