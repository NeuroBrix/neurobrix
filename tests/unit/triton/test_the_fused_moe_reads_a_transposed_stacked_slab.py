"""The triton fused MoE computes the dense softmax-first block from transposed slab views.

The op-level probe before any whole-model run (CARD REQUIRED — skipped without one).
Qwen3-VL's stacked slabs are W_in [E, H, 2F] and W_out [E, F, H] (transformers
4.57): the per-expert matrices the grouped GEMM reads are transposed VIEWS out of
them (`expert_weight_lists`, spec in_axis 0), so the kernel's B tile walks n
contiguously and k by the slab's row stride — a stride pattern no other model
hands it. This probe runs `execute_moe_fused` (the engine's own entry, both the
resident path and the zero3 promotion of host-resident views, which now crosses
`NBXTensor.to_cuda` for non-dense views) against the fp64 oracle of the TRACED
dense form: topk of the softmax, renormalised or not, every expert computed and
weighted by the scattered routing matrix, summed over the experts.

What this test would do if the code were wrong: a swapped half, a wrong stride
pair, or a host view copied as the slab's leading bytes changes the result by
O(1) against the oracle; the bound is the fp16 rounding of a three-GEMM chain.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor  # noqa: E402

E, K_, H, F, T = 8, 2, 64, 32, 5


def _oracle(h, probs, w_in, w_out, g_off, renorm):
    h, w_in, w_out = (x.astype(np.float64) for x in (h, w_in, w_out))
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :K_]
    sc = np.take_along_axis(probs.astype(np.float64), idx, -1)
    if renorm:
        sc = sc / sc.sum(-1, keepdims=True)
    R = np.zeros((T, E))
    np.put_along_axis(R, idx, sc, -1)
    u_off = F - g_off
    out = np.zeros((T, H))
    for e in range(E):
        g = h @ w_in[e][:, g_off:g_off + F]
        u = h @ w_in[e][:, u_off:u_off + F]
        a = (g / (1.0 + np.exp(-g))) * u
        out += R[:, e:e + 1] * (a @ w_out[e])
    return out


def _host(arr):
    t = NBXTensor.empty_cpu(arr.shape, NBXDtype.float16)
    ctypes.memmove(t.data_ptr(), np.ascontiguousarray(arr).ctypes.data, arr.nbytes)
    return t


@pytest.mark.parametrize("where", ["resident", "host"])
@pytest.mark.parametrize("g_off", [0, F])
@pytest.mark.parametrize("renorm", [True, False])
def test_the_grouped_gemm_reads_the_transposed_views(where, g_off, renorm):
    from neurobrix.core.runtime.graph.moe_fusion import expert_weight_lists
    from neurobrix.triton.moe import execute_moe_fused
    rng = np.random.default_rng(3 + g_off + 2 * renorm)
    h = (rng.standard_normal((T, H)) * 0.5).astype(np.float16)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    w_in = (rng.standard_normal((E, H, 2 * F)) * 0.1).astype(np.float16)
    w_out = (rng.standard_normal((E, F, H)) * 0.1).astype(np.float16)

    hs = NBXTensor.from_numpy(h)           # the first allocation: no card -> skipped
    gs = NBXTensor.from_numpy(probs)
    if where == "resident":
        W_in, W_out = NBXTensor.from_numpy(w_in), NBXTensor.from_numpy(w_out)
    else:
        W_in, W_out = _host(w_in), _host(w_out)
    attrs = {"num_experts": E,
             "stacked_experts": {"input_linear_tid": "in", "output_linear_tid": "out",
                                 "ffn_dim": F, "input_linear_in_axis": 0,
                                 "gate_offset": g_off, "output_linear_in_axis": 0},
             "expert_gate_weight_ids": [], "expert_up_weight_ids": [],
             "expert_down_weight_ids": []}
    gate, up, down = expert_weight_lists(attrs, {"in": W_in, "out": W_out}.get)
    out = execute_moe_fused(gs, hs, gate, up, down, top_k=K_, num_experts=E,
                            norm_topk_prob=renorm,
                            cache_key=f"transposed-slab-{where}-{g_off}-{renorm}")
    got = np.asarray(out.numpy(), dtype=np.float64).reshape(T, H)
    ref = _oracle(h, probs, w_in, w_out, g_off, renorm)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 2e-2, f"{where} g_off={g_off} renorm={renorm}: max rel {rel:.3e}"
