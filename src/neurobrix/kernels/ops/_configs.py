"""Shared autotune configurations for Triton kernels.

V100 safety: _safe_num_stages() clamps num_stages to 2 on pre-Ampere GPUs.
V100 (sm_70) lacks cp.async — Triton's software pipelining with num_stages > 2
emits async copy instructions that cause CUDA_ERROR_MISALIGNED_ADDRESS on Volta.
"""

from typing import Dict, List

import torch
import triton
from triton import next_power_of_2


def _safe_num_stages(n: int) -> int:
    """Clamp num_stages to 2 on pre-Ampere GPUs (V100 etc.).

    Triton uses cp.async for software pipelining with num_stages > 2.
    V100 (compute capability 7.0) does not support cp.async, causing
    misaligned address crashes in kernels that use tl.dot.
    """
    if not torch.cuda.is_available():
        return n
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
