"""PyTorch execution for ``custom::gated_delta_rule`` (compiled & sequential modes).

Dynamic over ANY sequence length ``L >= 1`` (decode ``L = 1`` and arbitrary
prefill ``L > 1``), matching ``nbxtensor.ops.gated_delta_rule`` and
``nbxtensor.ops.oracles.fp64.delta_rule_oracle``:

    S'_t = exp(g_t) * S_{t-1}
    u_t  = beta_t * (v_t - S'_t^T k_t)
    S_t  = S'_t + k_t (x) u_t
    o_t  = S_t^T q_t

Inputs:
    q, k : [B, L, Hk, dk]  (float16 / bfloat16 / float32)
    v    : [B, L, Hv, dv]  (float16 / bfloat16 / float32; Hv % Hk == 0)
    g    : [B, L, Hv]      (float32)
    beta : [B, L, Hv]      (float16 / bfloat16 / float32)
    state: Optional [B, Hv, dk, dv] (float32, None => zeros)

Outputs:
    out       : [B, L, Hv, dv] at v's dtype
    state_out : [B, Hv, dk, dv] float32 (returned when ``return_state=True``)
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


def _l2_norm_f32(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _serial_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct float32 recurrence for ``L < _CHUNK_SIZE`` (including decode ``L = 1``)."""
    B, L, Hv, dk = q.shape
    dv = v.shape[-1]
    s = (
        torch.zeros((B, Hv, dk, dv), dtype=torch.float32, device=v.device)
        if state is None
        else state.to(torch.float32).clone()
    )
    out = torch.empty((B, L, Hv, dv), dtype=torch.float32, device=v.device)
    for t in range(L):
        s = s * torch.exp(g[:, t, :, None, None])
        k_t = k[:, t]  # [B, Hv, dk]
        q_t = q[:, t]  # [B, Hv, dk]
        v_t = v[:, t]  # [B, Hv, dv]
        b_t = beta[:, t]  # [B, Hv]
        kv_mem = torch.einsum("bhkv,bhk->bhv", s, k_t)
        u = (v_t - kv_mem) * b_t.unsqueeze(-1)
        s = s + torch.einsum("bhk,bhv->bhkv", k_t, u)
        out[:, t] = torch.einsum("bhkv,bhk->bhv", s, q_t)
    return out, s


def _solve_chunk_wy(
    k_c: torch.Tensor, v_c: torch.Tensor, g_c: torch.Tensor, b_c: torch.Tensor,
    chunk_size: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the intra-chunk WY triangular system (I + A) W = diag(beta*exp(gc)) K and (I + A) U = diag(beta) V."""
    gc = g_c.cumsum(dim=-1)
    diff = gc.unsqueeze(-1) - gc.unsqueeze(-2)
    lower_mask = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.bool, device=device))
    strict_lower_mask = torch.tril(
        torch.ones((chunk_size, chunk_size), dtype=torch.bool, device=device), diagonal=-1
    )
    decay_tril = torch.exp(diff.masked_fill(~lower_mask, 0.0)).masked_fill(~lower_mask, 0.0)
    k_beta = k_c * b_c.unsqueeze(-1)
    v_beta = v_c * b_c.unsqueeze(-1)
    gram = torch.matmul(k_beta, k_c.transpose(-1, -2))
    A = (gram * decay_tril).masked_fill(~strict_lower_mask, 0.0)
    L_mat = torch.eye(chunk_size, dtype=torch.float32, device=device) + A
    W = torch.linalg.solve_triangular(L_mat, k_beta * torch.exp(gc).unsqueeze(-1), upper=False, unitriangular=True)
    U = torch.linalg.solve_triangular(L_mat, v_beta, upper=False, unitriangular=True)
    return gc, decay_tril, W, U


def _chunked_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor],
    chunk_size: int = _CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """WY-form chunked float32 recurrence for arbitrary ``L >= chunk_size``."""
    B, L, Hv, dk = q.shape
    dv = v.shape[-1]
    device = v.device

    # Transpose to [B, Hv, L, ...]
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    b_t = beta.transpose(1, 2).contiguous()

    pad = (chunk_size - (L % chunk_size)) % chunk_size
    if pad:
        q_t = F.pad(q_t, (0, 0, 0, pad))
        k_t = F.pad(k_t, (0, 0, 0, pad))
        v_t = F.pad(v_t, (0, 0, 0, pad))
        g_t = F.pad(g_t, (0, pad))
        b_t = F.pad(b_t, (0, pad))

    total_len = L + pad
    num_chunks = total_len // chunk_size

    q_c = q_t.reshape(B, Hv, num_chunks, chunk_size, dk)
    k_c = k_t.reshape(B, Hv, num_chunks, chunk_size, dk)
    v_c = v_t.reshape(B, Hv, num_chunks, chunk_size, dv)
    g_c = g_t.reshape(B, Hv, num_chunks, chunk_size)
    b_c = b_t.reshape(B, Hv, num_chunks, chunk_size)

    gc, decay_tril, W, U = _solve_chunk_wy(k_c, v_c, g_c, b_c, chunk_size, device)

    # Inter-chunk recurrence
    s = (
        torch.zeros((B, Hv, dk, dv), dtype=torch.float32, device=device)
        if state is None
        else state.to(torch.float32).clone()
    )
    out_chunks = torch.empty((B, Hv, num_chunks, chunk_size, dv), dtype=torch.float32, device=device)

    for c in range(num_chunks):
        q_i = q_c[:, :, c]          # [B, Hv, C, dk]
        k_i = k_c[:, :, c]          # [B, Hv, C, dk]
        w_i = W[:, :, c]            # [B, Hv, C, dk]
        u_i = U[:, :, c]            # [B, Hv, C, dv]
        gc_i = gc[:, :, c]          # [B, Hv, C]
        d_i = decay_tril[:, :, c]   # [B, Hv, C, C]

        v_new = u_i - torch.matmul(w_i, s)  # [B, Hv, C, dv]
        attn = torch.matmul(q_i, k_i.transpose(-1, -2)) * d_i
        o_inter = torch.matmul(q_i * torch.exp(gc_i).unsqueeze(-1), s)
        o_intra = torch.matmul(attn, v_new)
        out_chunks[:, :, c] = o_inter + o_intra

        gc_last = gc_i[:, :, -1]  # [B, Hv]
        s = s * torch.exp(gc_last)[:, :, None, None]
        k_decayed = k_i * torch.exp(gc_last.unsqueeze(-1) - gc_i).unsqueeze(-1)
        s = s + torch.matmul(k_decayed.transpose(-1, -2), v_new)

    out = out_chunks.reshape(B, Hv, total_len, dv)[:, :, :L, :].transpose(1, 2).contiguous()
    return out, s


def execute_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    *,
    qk_l2norm: bool = True,
    eps: float = 1e-6,
    scale: Optional[float] = None,
    return_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Execute ``custom::gated_delta_rule`` in PyTorch for any ``L >= 1``."""
    B, L, Hk, dk = q.shape
    Hv = v.shape[2]
    if Hk <= 0 or Hv % Hk != 0:
        raise ValueError(
            f"gated_delta_rule: Hv={Hv} must be a positive multiple of Hk={Hk}"
        )
    if L < 1:
        raise ValueError(f"gated_delta_rule: sequence length L={L} must be >= 1")

    eff_scale = float(dk ** -0.5) if scale is None else float(scale)
    out_dtype = v.dtype

    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    gf = g.to(torch.float32)
    bf = beta.to(torch.float32)

    if qk_l2norm:
        qf = _l2_norm_f32(qf, float(eps))
        kf = _l2_norm_f32(kf, float(eps))

    qf = qf * eff_scale

    if Hv != Hk:
        groups = Hv // Hk
        qf = qf.repeat_interleave(groups, dim=2)
        kf = kf.repeat_interleave(groups, dim=2)

    if L < _CHUNK_SIZE:
        out_f32, state_out = _serial_recurrence(qf, kf, vf, gf, bf, state)
    else:
        out_f32, state_out = _chunked_recurrence(qf, kf, vf, gf, bf, state)

    out = out_f32.to(out_dtype)
    return (out, state_out) if return_state else out


execute_gated_delta_rule_torch.self_manages_dtype = True  # type: ignore[attr-defined]
