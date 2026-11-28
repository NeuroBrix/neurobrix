"""CPU unit tests for ``custom::gated_delta_rule`` fusion, twin execution, and Prism accounting."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.oracles.fp64 import delta_rule_oracle
from nbxtensor.tensor import NBXTensor
from neurobrix.core.runtime.graph.gated_delta_rule_fusion import (
    FUSED_OP,
    detect_and_fuse_gated_delta_rule,
)
from neurobrix.core.runtime.graph.gated_delta_rule_torch import execute_gated_delta_rule_torch
from neurobrix.kernels.classification import ATEN_CLASSIFICATION, OpExecution
from neurobrix.triton.gated_delta_rule import execute_gated_delta_rule

_QWEN38_GRAPH = Path(
    "/mnt/nvme_ai_work/built_models_1127/Qwen3.8-27B/Qwen3.8-27B/"
    "components/model.language_model/graph.json"
)
_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def _unfused_chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    qk_l2norm: bool = True,
    eps: float = 1e-6,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference unfused PyTorch ``chunk_gated_delta_rule`` matching Qwen3.8-27B's graph subgraph."""
    B, L, Hk, dk = q.shape
    Hv, dv = v.shape[2], v.shape[3]
    out_dtype = v.dtype
    if Hv != Hk:
        groups = Hv // Hk
        q = q.unsqueeze(3).expand(B, L, Hk, groups, dk).contiguous().view(B, L, Hv, dk)
        k = k.unsqueeze(3).expand(B, L, Hk, groups, dk).contiguous().view(B, L, Hv, dk)
    qf, kf, vf, gf, bf = (x.to(torch.float32) for x in (q, k, v, g, beta))
    if qk_l2norm:
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + eps)
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + eps)
    qf = qf.transpose(1, 2).contiguous()
    kf = kf.transpose(1, 2).contiguous()
    vf = vf.transpose(1, 2).contiguous()
    bf = bf.transpose(1, 2).contiguous()
    gf = gf.transpose(1, 2).contiguous()

    pad = (chunk_size - (L % chunk_size)) % chunk_size
    if pad:
        qf = F.pad(qf, (0, 0, 0, pad))
        kf = F.pad(kf, (0, 0, 0, pad))
        vf = F.pad(vf, (0, 0, 0, pad))
        bf = F.pad(bf, (0, pad))
        gf = F.pad(gf, (0, pad))
    eff_scale = dk ** -0.5 if scale is None else scale
    qf = qf * eff_scale
    v_beta = vf * bf.unsqueeze(-1)
    k_beta = kf * bf.unsqueeze(-1)

    nt = (L + pad) // chunk_size
    q_c, k_c, kb_c, vb_c = (x.reshape(B, Hv, nt, chunk_size, -1) for x in (qf, kf, k_beta, v_beta))
    g_c = gf.reshape(B, Hv, nt, chunk_size)
    triu_mask = torch.triu(torch.ones((chunk_size, chunk_size), dtype=torch.bool), diagonal=0)
    gc = g_c.cumsum(dim=-1)
    decay = torch.tril(torch.exp(torch.tril(gc.unsqueeze(-1) - gc.unsqueeze(-2))))
    attn = -((kb_c @ k_c.transpose(-1, -2)) * decay).masked_fill(triu_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float32)
    u = attn @ vb_c
    w = attn @ (kb_c * torch.exp(gc).unsqueeze(-1))

    s = torch.zeros((B, Hv, dk, dv), dtype=torch.float32)
    out = torch.zeros_like(u)
    for i in range(nt):
        qi, ki, wi, ui = q_c[:, :, i], k_c[:, :, i], w[:, :, i], u[:, :, i]
        gci, di = gc[:, :, i], decay[:, :, i]
        v_new = ui - wi @ s
        a_i = (qi @ ki.transpose(-1, -2)) * di
        out[:, :, i] = (qi * torch.exp(gci).unsqueeze(-1)) @ s + a_i @ v_new
        gc_last = gci[:, :, -1]
        s = s * torch.exp(gc_last)[:, :, None, None] + (
            ki * torch.exp(gc_last.unsqueeze(-1) - gci).unsqueeze(-1)
        ).transpose(-1, -2) @ v_new
    return out.reshape(B, Hv, L + pad, dv)[:, :, :L, :].transpose(1, 2).contiguous().to(out_dtype)


def _make_inputs(B: int, L: int, Hk: int, Hv: int, dk: int, dv: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, L, Hk, dk)).astype(np.float32) * 0.5
    k = rng.standard_normal((B, L, Hk, dk)).astype(np.float32) * 0.5
    v = rng.standard_normal((B, L, Hv, dv)).astype(np.float32) * 0.5
    # Negative decay log-gates in realistic range [-2.0, -0.05]
    g = -np.abs(rng.standard_normal((B, L, Hv)).astype(np.float32)) * 0.3 - 0.05
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, L, Hv)).astype(np.float32)))
    return q, k, v, g, beta


@pytest.mark.parametrize("seq_len", [1, 19, 64, 129, 200])
@pytest.mark.parametrize("qk_l2norm", [True, False])
def test_numerical_parity_vs_fp64_oracle_and_unfused(seq_len: int, qk_l2norm: bool):
    B, Hk, Hv, dk, dv = 2, 2, 6, 16, 16
    eps = 1e-6
    scale = dk ** -0.5
    q_np, k_np, v_np, g_np, beta_np = _make_inputs(B, seq_len, Hk, Hv, dk, dv, seed=seq_len + int(qk_l2norm))

    # 1. fp64 oracle
    oracle_out, oracle_state = delta_rule_oracle(
        q_np, k_np, v_np, g_np, beta_np, None, qk_l2norm=qk_l2norm, eps=eps, scale=scale
    )()

    # 2. PyTorch dynamic (serial for L < 64, chunked for L >= 64)
    pt_out, pt_state = execute_gated_delta_rule_torch(
        torch.from_numpy(q_np),
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        torch.from_numpy(g_np),
        torch.from_numpy(beta_np),
        None,
        qk_l2norm=qk_l2norm,
        eps=eps,
        scale=scale,
        return_state=True,
    )

    # 3. Unfused reference chunk_gated_delta_rule
    unfused_out = _unfused_chunk_gated_delta_rule_ref(
        torch.from_numpy(q_np),
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        torch.from_numpy(g_np),
        torch.from_numpy(beta_np),
        qk_l2norm=qk_l2norm,
        eps=eps,
        scale=scale,
    )

    np.testing.assert_allclose(pt_out.numpy(), oracle_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(pt_state.numpy(), oracle_state, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(pt_out.numpy(), unfused_out.numpy(), rtol=1e-4, atol=1e-5)


def test_recurrent_state_continuity_prefill_plus_decode():
    """Running L=19 in one call must match L=18 prefill followed by L=1 decode carrying state."""
    B, L, Hk, Hv, dk, dv = 1, 19, 2, 6, 16, 16
    q_np, k_np, v_np, g_np, beta_np = _make_inputs(B, L, Hk, Hv, dk, dv, seed=77)
    scale = dk ** -0.5

    # Whole sequence
    pt_full, pt_s_full = execute_gated_delta_rule_torch(
        torch.from_numpy(q_np), torch.from_numpy(k_np), torch.from_numpy(v_np),
        torch.from_numpy(g_np), torch.from_numpy(beta_np),
        qk_l2norm=True, eps=1e-6, scale=scale, return_state=True,
    )

    # Split: 18 + 1
    pt_p, s18 = execute_gated_delta_rule_torch(
        torch.from_numpy(q_np[:, :18]), torch.from_numpy(k_np[:, :18]), torch.from_numpy(v_np[:, :18]),
        torch.from_numpy(g_np[:, :18]), torch.from_numpy(beta_np[:, :18]),
        qk_l2norm=True, eps=1e-6, scale=scale, return_state=True,
    )
    pt_d, pt_s_d = execute_gated_delta_rule_torch(
        torch.from_numpy(q_np[:, 18:]), torch.from_numpy(k_np[:, 18:]), torch.from_numpy(v_np[:, 18:]),
        torch.from_numpy(g_np[:, 18:]), torch.from_numpy(beta_np[:, 18:]), s18,
        qk_l2norm=True, eps=1e-6, scale=scale, return_state=True,
    )

    np.testing.assert_allclose(torch.cat([pt_p, pt_d], dim=1).numpy(), pt_full.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pt_s_d.numpy(), pt_s_full.numpy(), rtol=1e-5, atol=1e-6)


def test_triton_refuses_cpu_nbxtensor_and_numpy_zero_fallback():
    """Triton branch must explicitly refuse CPU NBXTensor and NumPy ndarray (ZERO FALLBACK)."""
    q_cpu = NBXTensor.empty_cpu((1, 4, 2, 16))
    k_cpu = NBXTensor.empty_cpu((1, 4, 2, 16))
    v_cpu = NBXTensor.empty_cpu((1, 4, 4, 16))
    g_cpu = NBXTensor.empty_cpu((1, 4, 4))
    b_cpu = NBXTensor.empty_cpu((1, 4, 4))

    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        execute_gated_delta_rule(q_cpu, k_cpu, v_cpu, g_cpu, b_cpu)

    q_np, k_np, v_np, g_np, b_np = _make_inputs(1, 4, 2, 4, 16, 16, seed=11)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        execute_gated_delta_rule(q_np, k_np, v_np, g_np, b_np)

    # Also refuse CPU state even when primary operands look like GPU NBXTensors
    gpu_q = SimpleNamespace(shape=(1, 4, 2, 16), is_cpu=False, _device="cuda")
    gpu_k = SimpleNamespace(shape=(1, 4, 2, 16), is_cpu=False, _device="cuda")
    gpu_v = SimpleNamespace(shape=(1, 4, 4, 16), is_cpu=False, _device="cuda")
    gpu_g = SimpleNamespace(shape=(1, 4, 4), is_cpu=False, _device="cuda")
    gpu_b = SimpleNamespace(shape=(1, 4, 4), is_cpu=False, _device="cuda")
    s_cpu = NBXTensor.empty_cpu((1, 4, 16, 16))
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        execute_gated_delta_rule(gpu_q, gpu_k, gpu_v, gpu_g, gpu_b, s_cpu)


def test_triton_delegates_to_nbxtensor_gated_delta_rule_op(monkeypatch):
    """Verify Triton branch delegates directly to ``nbxtensor.ops.gated_delta_rule.gated_delta_rule_op``."""
    calls = []
    sentinel_out = object()
    sentinel_state = object()

    def spy_op(q, k, v, g, beta, state=None, **kwargs):
        calls.append(((q, k, v, g, beta, state), kwargs))
        return sentinel_out, sentinel_state

    monkeypatch.setattr("nbxtensor.ops.gated_delta_rule.gated_delta_rule_op", spy_op)

    gpu_q = SimpleNamespace(shape=(2, 19, 2, 16), is_cpu=False, _device="cuda")
    gpu_k = SimpleNamespace(shape=(2, 19, 2, 16), is_cpu=False, _device="cuda")
    gpu_v = SimpleNamespace(shape=(2, 19, 6, 16), is_cpu=False, _device="cuda")
    gpu_g = SimpleNamespace(shape=(2, 19, 6), is_cpu=False, _device="cuda")
    gpu_b = SimpleNamespace(shape=(2, 19, 6), is_cpu=False, _device="cuda")
    gpu_s = SimpleNamespace(shape=(2, 6, 16, 16), is_cpu=False, _device="cuda")

    out_only = execute_gated_delta_rule(gpu_q, gpu_k, gpu_v, gpu_g, gpu_b)
    assert out_only is sentinel_out
    assert len(calls) == 1
    args0, kw0 = calls[0]
    assert args0 == (gpu_q, gpu_k, gpu_v, gpu_g, gpu_b, None)
    assert kw0 == {
        "qk_l2norm": True,
        "eps": 1e-6,
        "scale": pytest.approx(16 ** -0.5),
        "compute_dtype": NBXDtype.float32,
    }

    out, state_out = execute_gated_delta_rule(
        gpu_q, gpu_k, gpu_v, gpu_g, gpu_b, gpu_s,
        qk_l2norm=False, eps=1e-5, scale=0.25, return_state=True,
    )
    assert out is sentinel_out
    assert state_out is sentinel_state
    assert len(calls) == 2
    args1, kw1 = calls[1]
    assert args1 == (gpu_q, gpu_k, gpu_v, gpu_g, gpu_b, gpu_s)
    assert kw1 == {
        "qk_l2norm": False,
        "eps": 1e-5,
        "scale": 0.25,
        "compute_dtype": NBXDtype.float32,
    }


def test_bfloat16_output_dtype_preserved():
    B, L, Hk, Hv, dk, dv = 1, 70, 2, 6, 16, 16
    q_np, k_np, v_np, g_np, beta_np = _make_inputs(B, L, Hk, Hv, dk, dv, seed=99)
    out = execute_gated_delta_rule_torch(
        torch.from_numpy(q_np).to(torch.bfloat16),
        torch.from_numpy(k_np).to(torch.bfloat16),
        torch.from_numpy(v_np).to(torch.bfloat16),
        torch.from_numpy(g_np),  # g is float32 by contract
        torch.from_numpy(beta_np).to(torch.bfloat16),
        qk_l2norm=True,
        eps=1e-6,
        scale=dk ** -0.5,
    )
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (B, L, Hv, dv)


def test_ast_zero_torch_in_triton_and_pure_stdlib_in_fusion():
    # 1. Zero torch in triton/gated_delta_rule.py (and triton/sequence.py, triton/sequential.py)
    for rel in ("triton/gated_delta_rule.py", "triton/sequence.py", "triton/sequential.py",
                "triton/prism/work.py", "triton/prism/runtime_widths.py"):
        tree = ast.parse((_SRC_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not (alias.name == "torch" or alias.name.startswith("torch.")), f"torch import in {rel}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not (mod == "torch" or mod.startswith("torch.")), f"torch from-import in {rel}"

    # 2. Zero NumPy fallback helpers or NumPy recurrence arithmetic in triton/gated_delta_rule.py
    gdr_tree = ast.parse((_SRC_ROOT / "triton/gated_delta_rule.py").read_text())
    fn_names = {n.name for n in ast.walk(gdr_tree) if isinstance(n, ast.FunctionDef)}
    assert "_gated_delta_rule_numpy" not in fn_names
    assert "_to_f32_numpy" not in fn_names
    assert "_from_f32_numpy" not in fn_names
    assert fn_names == {"execute_gated_delta_rule"}
    np_attrs = {
        n.attr for n in ast.walk(gdr_tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "np"
    }
    assert np_attrs == {"ndarray"}, f"Unexpected numpy usage in triton/gated_delta_rule.py: {np_attrs}"

    # 3. Pure Python stdlib in gated_delta_rule_fusion.py
    fusion_path = _SRC_ROOT / "core/runtime/graph/gated_delta_rule_fusion.py"
    tree = ast.parse(fusion_path.read_text())
    stdlib = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in stdlib, f"Non-stdlib import {alias.name!r} in gated_delta_rule_fusion.py"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top in stdlib, f"Non-stdlib from-import {node.module!r} in gated_delta_rule_fusion.py"


def test_classification_and_prism_twin_accounting():
    assert ATEN_CLASSIFICATION[FUSED_OP] == OpExecution.TRITON

    from neurobrix.pytorch.prism.work import op_flops as pt_flops
    from neurobrix.triton.prism.work import op_flops as tr_flops

    ins = [[1, 129, 16, 128], [1, 129, 16, 128], [1, 129, 48, 128], [1, 129, 48], [1, 129, 48]]
    outs = [[1, 129, 48, 128]]
    attrs = {"qk_l2norm": True}
    f_pt = pt_flops(FUSED_OP, ins, outs, attrs)
    f_tr = tr_flops(FUSED_OP, ins, outs, attrs)
    expected = 1 * 129 * 48 * (7 * 128 * 128 + 2 * 128 + 128 + 2 * (3 * 128 + 1))
    assert f_pt == expected
    assert f_tr == expected


def _build_synthetic_gated_delta_rule_dag() -> dict:
    """Build a minimal synthetic DAG containing one unrolled ``chunk_gated_delta_rule`` subgraph."""
    tensors: dict = {
        "input::q": {"shape": [2, 5, 2, 16], "dtype": "bfloat16", "is_input": True},
        "input::k": {"shape": [2, 5, 2, 16], "dtype": "bfloat16", "is_input": True},
        "input::v": {"shape": [2, 5, 4, 16], "dtype": "bfloat16", "is_input": True},
        "input::g": {"shape": [2, 5, 4], "dtype": "float32", "is_input": True},
        "input::beta": {"shape": [2, 5, 4], "dtype": "bfloat16", "is_input": True},
        "buffer::out_init": {"shape": [2, 4, 1, 64, 16], "dtype": "float32", "is_buffer": True},
    }
    ops: dict = {}
    order: list = []
    counter = 0

    def _t(tid: str) -> dict:
        return {"type": "tensor", "tensor_id": tid}

    def _s(val) -> dict:
        return {"type": "scalar", "value": val}

    def _l(val: list) -> dict:
        return {"type": "list", "value": val}

    def add_op(op_type: str, args: list, out_shape: list, out_dtype: str = "float32") -> str:
        nonlocal counter
        op_prefix = op_type.replace("::", ".")
        uid = f"{op_prefix}::{counter}"
        counter += 1
        out_tid = f"{uid}::out_0"
        tensors[out_tid] = {"shape": out_shape, "dtype": out_dtype, "device": "cuda"}
        ops[uid] = {
            "op_uid": uid,
            "op_type": op_type,
            "attributes": {"args": args},
            "output_tensor_ids": [out_tid],
            "device": "cuda",
        }
        order.append(uid)
        return out_tid

    # GQA head expansion on q and k: [2, 5, 2, 16] -> [2, 5, 4, 16]
    def head_expand(x_tid: str) -> str:
        u = add_op("aten::unsqueeze", [_t(x_tid), _s(3)], [2, 5, 2, 1, 16], "bfloat16")
        e = add_op("aten::expand", [_t(u), _l([2, 5, 2, 2, 16])], [2, 5, 2, 2, 16], "bfloat16")
        c = add_op("aten::clone", [_t(e)], [2, 5, 2, 2, 16], "bfloat16")
        return add_op("aten::view", [_t(c), _l([2, 5, 4, 16])], [2, 5, 4, 16], "bfloat16")

    q_exp = head_expand("input::q")
    k_exp = head_expand("input::k")

    # L2 normalization on q and k
    def l2norm(x_tid: str, eps: float = 1e-6) -> str:
        sq = add_op("aten::mul", [_t(x_tid), _t(x_tid)], [2, 5, 4, 16])
        sm = add_op("aten::sum", [_t(sq), _l([-1]), _s(True)], [2, 5, 4, 1])
        ad = add_op("aten::add", [_t(sm), _s(eps)], [2, 5, 4, 1])
        rs = add_op("aten::rsqrt", [_t(ad)], [2, 5, 4, 1])
        return add_op("aten::mul", [_t(x_tid), _t(rs)], [2, 5, 4, 16])

    q_norm = l2norm(q_exp)
    k_norm = l2norm(k_exp)

    # transpose(1, 2) -> constant_pad_nd
    def pad_trans(x_tid: str, tr_shape: list, pad_shape: list, pad_spec: list) -> str:
        tr = add_op("aten::transpose", [_t(x_tid), _s(1), _s(2)], tr_shape)
        cp = add_op("aten::contiguous", [_t(tr)], tr_shape)
        return add_op("aten::constant_pad_nd", [_t(cp), _l(pad_spec), _s(0)], pad_shape)

    q_pad = pad_trans(q_norm, [2, 4, 5, 16], [2, 4, 64, 16], [0, 0, 0, 59])
    k_pad = pad_trans(k_norm, [2, 4, 5, 16], [2, 4, 64, 16], [0, 0, 0, 59])
    v_pad = pad_trans("input::v", [2, 4, 5, 16], [2, 4, 64, 16], [0, 0, 0, 59])
    g_pad = pad_trans("input::g", [2, 4, 5], [2, 4, 64], [0, 59])
    b_pad = pad_trans("input::beta", [2, 4, 5], [2, 4, 64], [0, 59])

    # g_chunked -> cumsum -> decay_mask [2, 4, 1, 64, 64]
    g_chunked = add_op("aten::view", [_t(g_pad), _l([2, 4, 1, 64])], [2, 4, 1, 64])
    g_cumsum = add_op("aten::cumsum", [_t(g_chunked), _s(-1)], [2, 4, 1, 64])
    g_u1 = add_op("aten::unsqueeze", [_t(g_cumsum), _s(-1)], [2, 4, 1, 64, 1])
    g_u2 = add_op("aten::unsqueeze", [_t(g_cumsum), _s(-2)], [2, 4, 1, 1, 64])
    g_sub = add_op("aten::sub", [_t(g_u1), _t(g_u2)], [2, 4, 1, 64, 64])
    g_tril1 = add_op("aten::tril", [_t(g_sub)], [2, 4, 1, 64, 64])
    g_exp = add_op("aten::exp", [_t(g_tril1)], [2, 4, 1, 64, 64])
    decay_mask = add_op("aten::tril", [_t(g_exp)], [2, 4, 1, 64, 64])

    # k_beta @ k^T * decay_mask
    b_unsq1 = add_op("aten::unsqueeze", [_t(b_pad), _s(-1)], [2, 4, 64, 1])
    kb_mul = add_op("aten::mul", [_t(k_pad), _t(b_unsq1)], [2, 4, 64, 16])
    kb_chunked = add_op("aten::view", [_t(kb_mul), _l([2, 4, 1, 64, 16])], [2, 4, 1, 64, 16])
    k_chunked = add_op("aten::view", [_t(k_pad), _l([2, 4, 1, 64, 16])], [2, 4, 1, 64, 16])
    kb_bmm_in = add_op("aten::view", [_t(kb_chunked), _l([8, 64, 16])], [8, 64, 16])
    k_t = add_op("aten::transpose", [_t(k_chunked), _s(-1), _s(-2)], [2, 4, 1, 16, 64])
    k_bmm_in = add_op("aten::view", [_t(k_t), _l([8, 16, 64])], [8, 16, 64])
    kb_k_bmm = add_op("aten::bmm", [_t(kb_bmm_in), _t(k_bmm_in)], [8, 64, 64])
    kb_k_5d = add_op("aten::view", [_t(kb_k_bmm), _l([2, 4, 1, 64, 64])], [2, 4, 1, 64, 64])
    _ = add_op("aten::mul", [_t(decay_mask), _t(kb_k_5d)], [2, 4, 1, 64, 64])

    # v_beta path
    b_unsq2 = add_op("aten::unsqueeze", [_t(b_pad), _s(-1)], [2, 4, 64, 1])
    vb_mul = add_op("aten::mul", [_t(v_pad), _t(b_unsq2)], [2, 4, 64, 16])
    _ = add_op("aten::view", [_t(vb_mul), _l([2, 4, 1, 64, 16])], [2, 4, 1, 64, 16])

    # q_scaled & chunk-0 attn product -> select_scatter -> view -> slice -> transpose -> _to_copy
    q_scaled = add_op("aten::mul", [_t(q_pad), _s(0.25)], [2, 4, 64, 16])
    q_chunked = add_op("aten::view", [_t(q_scaled), _l([2, 4, 1, 64, 16])], [2, 4, 1, 64, 16])
    q0 = add_op("aten::select", [_t(q_chunked), _s(2), _s(0)], [2, 4, 64, 16])
    k0 = add_op("aten::select", [_t(k_chunked), _s(2), _s(0)], [2, 4, 64, 16])
    q0_bmm_in = add_op("aten::view", [_t(q0), _l([8, 64, 16])], [8, 64, 16])
    k0_t = add_op("aten::transpose", [_t(k0), _s(-1), _s(-2)], [2, 4, 16, 64])
    k0_bmm_in = add_op("aten::view", [_t(k0_t), _l([8, 16, 64])], [8, 16, 64])
    qk0_bmm = add_op("aten::bmm", [_t(q0_bmm_in), _t(k0_bmm_in)], [8, 64, 64])
    qk0_4d = add_op("aten::view", [_t(qk0_bmm), _l([2, 4, 64, 64])], [2, 4, 64, 64])
    dm0 = add_op("aten::select", [_t(decay_mask), _s(2), _s(0)], [2, 4, 64, 64])
    dm0_mul = add_op("aten::mul", [_t(dm0), _t(qk0_4d)], [2, 4, 64, 64])

    ss0 = add_op(
        "aten::select_scatter",
        [_t("buffer::out_init"), _t(dm0_mul), _s(2), _s(0)],
        [2, 4, 1, 64, 16],
    )
    out_view = add_op("aten::view", [_t(ss0), _l([2, 4, 64, 16])], [2, 4, 64, 16])
    out_slice = add_op("aten::slice", [_t(out_view), _s(2), _s(0), _s(5)], [2, 4, 5, 16])
    out_trans = add_op("aten::transpose", [_t(out_slice), _s(1), _s(2)], [2, 5, 4, 16])
    out_final = add_op("aten::_to_copy", [_t(out_trans)], [2, 5, 4, 16], "bfloat16")

    return {
        "input_tensor_ids": ["input::q", "input::k", "input::v", "input::g", "input::beta"],
        "output_tensor_ids": [out_final],
        "tensors": tensors,
        "ops": ops,
        "execution_order": order,
    }


def test_fusion_on_synthetic_minimal_graph():
    """Fusion pass must detect and fuse a minimal synthetic ``chunk_gated_delta_rule`` graph."""
    dag = _build_synthetic_gated_delta_rule_dag()
    n_before = len(dag["execution_order"])
    expected_out = dag["output_tensor_ids"][0]

    detect_and_fuse_gated_delta_rule(dag)

    fused = [uid for uid in dag["execution_order"] if dag["ops"][uid]["op_type"] == FUSED_OP]
    rem_cumsum = [uid for uid in dag["execution_order"] if dag["ops"][uid]["op_type"] == "aten::cumsum"]

    assert len(fused) == 1
    assert len(rem_cumsum) == 0
    assert len(dag["execution_order"]) < n_before

    op0 = dag["ops"][fused[0]]
    assert op0["input_tensor_ids"] == ["input::q", "input::k", "input::v", "input::g", "input::beta"]
    assert op0["output_tensor_ids"] == [expected_out]
    assert op0["input_shapes"][0] == [2, 5, 2, 16]
    assert op0["input_shapes"][2] == [2, 5, 4, 16]
    assert op0["output_shapes"] == [[2, 5, 4, 16]]
    assert op0["attributes"]["qk_l2norm"] is True
    assert op0["attributes"]["eps"] == pytest.approx(1e-6)
    assert op0["attributes"]["scale"] == pytest.approx(0.25)

    # Topological / producer completeness check
    produced = set(dag.get("input_tensor_ids", []))
    for tid, meta in dag["tensors"].items():
        if (meta.get("is_parameter") or meta.get("is_buffer") or meta.get("is_input")
                or meta.get("constant") or tid.startswith(("param::", "input::", "buffer::"))):
            produced.add(tid)
    for uid in dag["execution_order"]:
        op = dag["ops"][uid]
        for itid in op.get("input_tensor_ids", []):
            assert itid in produced, f"Missing producer for {itid} consumed by {uid}"
        for otid in op.get("output_tensor_ids", []):
            produced.add(otid)

    # Idempotent
    n_after = len(dag["execution_order"])
    detect_and_fuse_gated_delta_rule(dag)
    assert len(dag["execution_order"]) == n_after


@pytest.mark.skipif(not _QWEN38_GRAPH.exists(), reason="Qwen3.8-27B graph.json not mounted")
def test_fusion_on_real_qwen38_27b_graph():
    with _QWEN38_GRAPH.open() as f:
        dag = json.load(f)

    n_before = len(dag["execution_order"])
    assert "aten.select::142" in dag["ops"]

    detect_and_fuse_gated_delta_rule(dag)

    fused = [uid for uid in dag["execution_order"] if dag["ops"][uid]["op_type"] == FUSED_OP]
    rem_cumsum = [uid for uid in dag["execution_order"] if dag["ops"][uid]["op_type"] == "aten::cumsum"]

    assert len(fused) == 48
    assert len(rem_cumsum) == 0
    assert "aten.select::142" not in dag["ops"]
    assert len(dag["execution_order"]) < n_before - 50000

    # Verify first fused op contract
    op0 = dag["ops"][fused[0]]
    assert len(op0["input_tensor_ids"]) == 5
    assert op0["input_shapes"][0] == [5, 129, 16, 128]
    assert op0["input_shapes"][2] == [5, 129, 48, 128]
    assert op0["output_shapes"] == [[5, 129, 48, 128]]
    assert op0["attributes"]["qk_l2norm"] is True
    assert op0["attributes"]["eps"] == pytest.approx(1e-6)
    assert op0["attributes"]["scale"] == pytest.approx(128 ** -0.5)

    # Topological / producer completeness check
    produced = set(dag.get("input_tensor_ids", []))
    for tid, meta in dag["tensors"].items():
        if (meta.get("is_parameter") or meta.get("is_buffer") or meta.get("is_input")
                or meta.get("constant") or tid.startswith(("param::", "input::", "buffer::"))):
            produced.add(tid)
    for uid in dag["execution_order"]:
        op = dag["ops"][uid]
        for itid in op.get("input_tensor_ids", []):
            assert itid in produced, f"Missing producer for {itid} consumed by {uid}"
        for otid in op.get("output_tensor_ids", []):
            produced.add(otid)

    # Idempotent
    n_after = len(dag["execution_order"])
    detect_and_fuse_gated_delta_rule(dag)
    assert len(dag["execution_order"]) == n_after
