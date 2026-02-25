# LayerNorm - Pure Triton
# Source: FlagGems (Apache 2.0)
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b

@triton.jit
def layer_norm_persistent_kernel(
    in_ptr, out_ptr, weight_ptr, bias_ptr,
    out_mean_ptr, out_rstd_ptr,
    M, N, eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x, axis=0) / N
    d = x - m
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=0)
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    # Store stats if needed (optional pointers)
    if out_mean_ptr is not None: tl.store(out_mean_ptr + pid, m)
    if out_rstd_ptr is not None: tl.store(out_rstd_ptr + pid, rstd)

    w = tl.load(weight_ptr + n_offsets, mask=mask) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + n_offsets, mask=mask) if bias_ptr is not None else 0.0
    
    out = (x - m) * rstd * w + b
    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

@triton.jit
def layer_norm_loop_kernel(
    in_ptr, out_ptr, weight_ptr, bias_ptr,
    out_mean_ptr, out_rstd_ptr,
    M, N, eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # Welford or simple 2-pass? FlagGems uses 2-pass with loop
    
    # 1. Mean
    m = 0.0
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        val = tl.load(in_ptr + pid * N + cols, mask, other=0.0).to(tl.float32)
        m += tl.sum(val, axis=0)
    m = m / N
    
    # 2. Var
    var = 0.0
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        val = tl.load(in_ptr + pid * N + cols, mask, other=0.0).to(tl.float32)
        diff = val - m
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = tl.math.rsqrt(var + eps)
    
    if out_mean_ptr is not None: tl.store(out_mean_ptr + pid, m)
    if out_rstd_ptr is not None: tl.store(out_rstd_ptr + pid, rstd)
    
    # 3. Normalize
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N
        val = tl.load(in_ptr + pid * N + cols, mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask) if weight_ptr is not None else 1.0
        b = tl.load(bias_ptr + cols, mask=mask) if bias_ptr is not None else 0.0
        
        out = (val - m) * rstd * w + b
        tl.store(out_ptr + pid * N + cols, out, mask=mask)
