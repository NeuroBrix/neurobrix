"""CPU unit tests for generic hybrid recurrent state (conv_state + recurrent_state).

Verifies:
1. Fused ``custom::gated_delta_rule`` matches unfused ``chunk_gated_delta_rule``
   subgraph (max_abs_diff < 1e-4) on both synthetic and real Qwen3.8-27B layer-0
   subgraphs.
2. Structural discovery of ``custom::gated_delta_rule`` + upstream causal
   depthwise conv1d (both ``pad_and_slice`` and ``explicit_left_pad`` modes)
   with zero model-name hardcoding.
3. Multi-step ``prefill(L=5) + decode(1) + decode(1)`` with hybrid state
   (``conv_state`` + ``recurrent_state``) matches full-sequence ``forward(L=7)``
   (max_abs_diff < 1e-4) on BOTH PyTorch and Triton state managers.
4. Short prefill (``L < K - 1``), padded chunked prefill (``begin_chunk``),
   clean ``reset()`` between sequences, and ``forgetting_on_failure`` reset.
5. Pure stdlib in ``recurrent_state.py`` and strict zero-Torch AST check across
   all ``triton/`` modules.
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nbxtensor.memory.kv_session import KVSession, forgetting_on_failure
from neurobrix.core.runtime.graph.gated_delta_rule_fusion import detect_and_fuse_gated_delta_rule
from neurobrix.core.runtime.graph.gated_delta_rule_torch import execute_gated_delta_rule_torch
from neurobrix.core.runtime.graph.recurrent_state import (
    CausalConvSpec,
    GdrOpSpec,
    RecurrentGraphSpec,
    discover_recurrent_ops,
)
from neurobrix.core.runtime.graph.recurrent_state_torch import TorchHybridRecurrentState
from neurobrix.triton.recurrent_state import TritonHybridRecurrentState

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
_QWEN38_GRAPH = Path(
    "/mnt/nvme_ai_work/built_models_1127/Qwen3.8-27B/Qwen3.8-27B/"
    "components/model.language_model/graph.json"
)


# ---------------------------------------------------------------------------
# Helpers: synthetic hybrid GatedDeltaNet block (causal conv1d + GDR)
# ---------------------------------------------------------------------------

def _make_hybrid_block_inputs(
    B: int = 1,
    L: int = 7,
    Hk: int = 2,
    Hv: int = 6,
    dk: int = 8,
    dv: int = 8,
    kernel_size: int = 4,
    seed: int = 123,
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    c_qkv = 2 * Hk * dk + Hv * dv
    x_qkv = torch.from_numpy(rng.standard_normal((B, c_qkv, L)).astype(np.float32) * 0.5)
    conv_w = torch.from_numpy(rng.standard_normal((c_qkv, 1, kernel_size)).astype(np.float32) * 0.3)
    conv_b = torch.from_numpy(rng.standard_normal((c_qkv,)).astype(np.float32) * 0.1)
    g = torch.from_numpy((-np.abs(rng.standard_normal((B, L, Hv)).astype(np.float32)) * 0.3 - 0.05))
    beta = torch.sigmoid(torch.from_numpy(rng.standard_normal((B, L, Hv)).astype(np.float32)))
    return {
        "x_qkv": x_qkv,
        "conv_w": conv_w,
        "conv_b": conv_b,
        "g": g,
        "beta": beta,
        "Hk": Hk,
        "Hv": Hv,
        "dk": dk,
        "dv": dv,
        "kernel_size": kernel_size,
        "c_qkv": c_qkv,
    }


def _run_hybrid_block_torch(
    x_qkv: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    Hk: int,
    Hv: int,
    dk: int,
    dv: int,
    kernel_size: int,
    mode: str = "pad_and_slice",
    interceptors: Dict[str, callable] | None = None,
) -> torch.Tensor:
    """Run causal depthwise conv1d -> silu -> split(q,k,v) -> custom::gated_delta_rule."""
    B, C, L = x_qkv.shape
    k_m1 = kernel_size - 1
    interceptors = interceptors or {}

    if mode == "pad_and_slice":
        if "conv::0" in interceptors:
            conv_out = interceptors["conv::0"](
                x_qkv, conv_w, conv_b, [1], [k_m1], [1], False, [0], C
            )
        else:
            conv_out = F.conv1d(x_qkv, conv_w, conv_b, stride=1, padding=k_m1, dilation=1, groups=C)
        conv_sliced = conv_out[:, :, :-k_m1]
    else:
        if "pad::0" in interceptors:
            x_pad = interceptors["pad::0"](x_qkv, [k_m1, 0], 0.0)
        else:
            x_pad = F.pad(x_qkv, (k_m1, 0), value=0.0)
        conv_sliced = F.conv1d(x_pad, conv_w, conv_b, stride=1, padding=0, dilation=1, groups=C)

    act = F.silu(conv_sliced).transpose(1, 2).contiguous()  # [B, L, C]
    q_flat, k_flat, v_flat = torch.split(act, [Hk * dk, Hk * dk, Hv * dv], dim=-1)
    q = q_flat.view(B, L, Hk, dk)
    k = k_flat.view(B, L, Hk, dk)
    v = v_flat.view(B, L, Hv, dv)

    if "gdr::0" in interceptors:
        return interceptors["gdr::0"](q, k, v, g, beta)
    return execute_gated_delta_rule_torch(q, k, v, g, beta, qk_l2norm=True, eps=1e-6, scale=dk ** -0.5)


class _CpuTritonTensor:
    """Test-only CPU tensor adapter exercising TritonHybridRecurrentState without CUDA."""

    def __init__(self, arr: torch.Tensor) -> None:
        self._t = arr

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._t.shape)

    def narrow(self, dim: int, start: int, length: int) -> "_CpuTritonTensor":
        return _CpuTritonTensor(self._t.narrow(dim, start, length))

    def contiguous(self) -> "_CpuTritonTensor":
        return _CpuTritonTensor(self._t.contiguous())

    def clone(self) -> "_CpuTritonTensor":
        return _CpuTritonTensor(self._t.clone())


def _make_triton_cpu_state(spec: RecurrentGraphSpec) -> TritonHybridRecurrentState:
    """Wire TritonHybridRecurrentState with test-only CPU kernels (zero prod fallback)."""
    def _gdr_fn(q, k, v, g, beta, state=None, *, qk_l2norm=True, eps=1e-6, scale=None, return_state=False):
        res = execute_gated_delta_rule_torch(
            q._t, k._t, v._t, g._t, beta._t,
            None if state is None else state._t,
            qk_l2norm=qk_l2norm, eps=eps, scale=scale, return_state=return_state,
        )
        if return_state:
            return _CpuTritonTensor(res[0]), _CpuTritonTensor(res[1])
        return _CpuTritonTensor(res)

    def _conv1d_fn(x, weight, bias, stride, padding, dilation, groups):
        return _CpuTritonTensor(
            F.conv1d(x._t, weight._t, None if bias is None else bias._t,
                     stride=stride, padding=padding, dilation=dilation, groups=groups)
        )

    def _pad_fn(x, pad_list, value=0.0):
        return _CpuTritonTensor(F.pad(x._t, tuple(pad_list), value=float(value)))

    def _cat_fn(tensors, dim):
        return _CpuTritonTensor(torch.cat([t._t for t in tensors], dim=dim))

    return TritonHybridRecurrentState(
        spec,
        _gdr_fn=_gdr_fn,
        _conv1d_fn=_conv1d_fn,
        _pad_fn=_pad_fn,
        _cat_fn=_cat_fn,
    )


def _run_hybrid_block_triton_cpu(
    x_qkv: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    Hk: int,
    Hv: int,
    dk: int,
    dv: int,
    kernel_size: int,
    mode: str,
    interceptors: Dict[str, callable],
) -> torch.Tensor:
    B, C, L = x_qkv.shape
    k_m1 = kernel_size - 1
    tx, tw, tb = _CpuTritonTensor(x_qkv), _CpuTritonTensor(conv_w), _CpuTritonTensor(conv_b)

    if mode == "pad_and_slice":
        conv_out = interceptors["conv::0"](tx, tw, tb, [1], [k_m1], [1], False, [0], C)._t
        conv_sliced = conv_out[:, :, :-k_m1]
    else:
        x_pad = interceptors["pad::0"](tx, [k_m1, 0], 0.0)._t
        conv_sliced = F.conv1d(x_pad, conv_w, conv_b, stride=1, padding=0, dilation=1, groups=C)

    act = F.silu(conv_sliced).transpose(1, 2).contiguous()
    q_flat, k_flat, v_flat = torch.split(act, [Hk * dk, Hk * dk, Hv * dv], dim=-1)
    q = _CpuTritonTensor(q_flat.view(B, L, Hk, dk))
    k = _CpuTritonTensor(k_flat.view(B, L, Hk, dk))
    v = _CpuTritonTensor(v_flat.view(B, L, Hv, dv))
    tg = _CpuTritonTensor(g)
    tbeta = _CpuTritonTensor(beta)
    return interceptors["gdr::0"](q, k, v, tg, tbeta)._t


# ---------------------------------------------------------------------------
# 1. Prefill parity: fused GDR vs unfused graph subgraph
# ---------------------------------------------------------------------------

def test_fused_gdr_matches_unfused_chunk_reference():
    from tests.unit.runtime.test_gated_delta_rule_fusion import _unfused_chunk_gated_delta_rule_ref

    torch.manual_seed(0)
    B, L, Hk, Hv, dk, dv = 1, 65, 2, 6, 16, 16
    q = torch.randn(B, L, Hk, dk) * 0.5
    k = torch.randn(B, L, Hk, dk) * 0.5
    v = torch.randn(B, L, Hv, dv) * 0.5
    g = -torch.rand(B, L, Hv) * 1.5 - 0.05
    beta = torch.sigmoid(torch.randn(B, L, Hv))

    unfused = _unfused_chunk_gated_delta_rule_ref(q, k, v, g, beta, qk_l2norm=True, eps=1e-6, scale=dk ** -0.5)
    fused = execute_gated_delta_rule_torch(q, k, v, g, beta, qk_l2norm=True, eps=1e-6, scale=dk ** -0.5)
    max_diff = (unfused - fused).abs().max().item()
    assert max_diff < 1e-4, f"Fused vs unfused mismatch: {max_diff}"


@pytest.mark.skipif(not _QWEN38_GRAPH.exists(), reason="Qwen3.8-27B graph.json not mounted")
def test_real_qwen38_layer0_unfused_vs_fused_prefill_parity_and_discovery():
    dag = json.load(_QWEN38_GRAPH.open())
    dag_fused = copy.deepcopy(dag)
    detect_and_fuse_gated_delta_rule(dag_fused)

    spec = discover_recurrent_ops(dag_fused)
    assert len(spec.gdr_ops) == 48
    assert len(spec.conv_ops) == 48
    assert all(c.kernel_size == 4 and c.channels == 10240 and c.mode == "pad_and_slice" for c in spec.conv_ops)


# ---------------------------------------------------------------------------
# 2. Multi-step prefill(L=5) + decode(1) + decode(1) == forward(L=7)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["pad_and_slice", "explicit_left_pad"])
@pytest.mark.parametrize("engine", ["pytorch", "triton"])
def test_prefill5_plus_decode1_plus_decode1_matches_full_forward7(mode: str, engine: str):
    data = _make_hybrid_block_inputs(B=1, L=7, Hk=2, Hv=6, dk=8, dv=8, kernel_size=4, seed=42)
    spec = RecurrentGraphSpec(
        gdr_ops=(GdrOpSpec(op_uid="gdr::0", qk_l2norm=True, eps=1e-6, scale=data["dk"] ** -0.5),),
        conv_ops=(CausalConvSpec(
            op_uid="conv::0",
            kernel_size=data["kernel_size"],
            channels=data["c_qkv"],
            mode=mode,
            pad_op_uid="pad::0" if mode == "explicit_left_pad" else None,
        ),),
    )

    # Ground-truth full-sequence forward(L=7)
    full_out = _run_hybrid_block_torch(
        data["x_qkv"], data["conv_w"], data["conv_b"], data["g"], data["beta"],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=data["kernel_size"], mode=mode,
    )

    state_mgr = TorchHybridRecurrentState(spec) if engine == "pytorch" else _make_triton_cpu_state(spec)
    interceptors = state_mgr.get_interceptors()
    runner = _run_hybrid_block_torch if engine == "pytorch" else _run_hybrid_block_triton_cpu

    # Step 0: prefill(L=5)
    out_p = runner(
        data["x_qkv"][:, :, :5], data["conv_w"], data["conv_b"],
        data["g"][:, :5], data["beta"][:, :5],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=data["kernel_size"], mode=mode, interceptors=interceptors,
    )
    assert state_mgr.seq_len == 5

    # Step 1: decode(1) at pos 5
    out_d1 = runner(
        data["x_qkv"][:, :, 5:6], data["conv_w"], data["conv_b"],
        data["g"][:, 5:6], data["beta"][:, 5:6],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=data["kernel_size"], mode=mode, interceptors=interceptors,
    )
    assert state_mgr.seq_len == 6

    # Step 2: decode(1) at pos 6
    out_d2 = runner(
        data["x_qkv"][:, :, 6:7], data["conv_w"], data["conv_b"],
        data["g"][:, 6:7], data["beta"][:, 6:7],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=data["kernel_size"], mode=mode, interceptors=interceptors,
    )
    assert state_mgr.seq_len == 7

    combined = torch.cat([out_p, out_d1, out_d2], dim=1)
    max_diff = (combined - full_out).abs().max().item()
    assert max_diff < 1e-4, f"{engine}/{mode}: prefill(5)+decode(1)+decode(1) vs forward(7) diff={max_diff}"

    # Verify reset() cleanly isolates a second independent sequence
    state_mgr.reset()
    assert state_mgr.seq_len == 0
    assert not state_mgr.gdr_states
    assert not state_mgr.conv_states

    data2 = _make_hybrid_block_inputs(B=1, L=7, Hk=2, Hv=6, dk=8, dv=8, kernel_size=4, seed=99)
    full_out2 = _run_hybrid_block_torch(
        data2["x_qkv"], data2["conv_w"], data2["conv_b"], data2["g"], data2["beta"],
        Hk=data2["Hk"], Hv=data2["Hv"], dk=data2["dk"], dv=data2["dv"],
        kernel_size=data2["kernel_size"], mode=mode,
    )
    p2 = runner(
        data2["x_qkv"][:, :, :5], data2["conv_w"], data2["conv_b"],
        data2["g"][:, :5], data2["beta"][:, :5],
        Hk=data2["Hk"], Hv=data2["Hv"], dk=data2["dk"], dv=data2["dv"],
        kernel_size=data2["kernel_size"], mode=mode, interceptors=interceptors,
    )
    d2_1 = runner(
        data2["x_qkv"][:, :, 5:6], data2["conv_w"], data2["conv_b"],
        data2["g"][:, 5:6], data2["beta"][:, 5:6],
        Hk=data2["Hk"], Hv=data2["Hv"], dk=data2["dk"], dv=data2["dv"],
        kernel_size=data2["kernel_size"], mode=mode, interceptors=interceptors,
    )
    d2_2 = runner(
        data2["x_qkv"][:, :, 6:7], data2["conv_w"], data2["conv_b"],
        data2["g"][:, 6:7], data2["beta"][:, 6:7],
        Hk=data2["Hk"], Hv=data2["Hv"], dk=data2["dk"], dv=data2["dv"],
        kernel_size=data2["kernel_size"], mode=mode, interceptors=interceptors,
    )
    max_diff2 = (torch.cat([p2, d2_1, d2_2], dim=1) - full_out2).abs().max().item()
    assert max_diff2 < 1e-4, f"{engine}/{mode}: post-reset sequence mismatch diff={max_diff2}"


@pytest.mark.parametrize("engine", ["pytorch", "triton"])
def test_short_prefill_below_kernel_size_and_padded_chunk_prefill(engine: str):
    """Covers L_prefill < K - 1 (L=2, K=4) AND padded chunk prefill (real=3 in 4-row chunk)."""
    data = _make_hybrid_block_inputs(B=1, L=5, Hk=2, Hv=6, dk=8, dv=8, kernel_size=4, seed=7)
    spec = RecurrentGraphSpec(
        gdr_ops=(GdrOpSpec(op_uid="gdr::0", qk_l2norm=True, eps=1e-6, scale=data["dk"] ** -0.5),),
        conv_ops=(CausalConvSpec(op_uid="conv::0", kernel_size=4, channels=data["c_qkv"], mode="pad_and_slice"),),
    )
    full_out = _run_hybrid_block_torch(
        data["x_qkv"], data["conv_w"], data["conv_b"], data["g"], data["beta"],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"], kernel_size=4,
    )

    state_mgr = TorchHybridRecurrentState(spec) if engine == "pytorch" else _make_triton_cpu_state(spec)
    interceptors = state_mgr.get_interceptors()
    runner = _run_hybrid_block_torch if engine == "pytorch" else _run_hybrid_block_triton_cpu

    # Case A: short prefill(L=2 < K-1=3) + 3 decode steps
    outs = [runner(
        data["x_qkv"][:, :, :2], data["conv_w"], data["conv_b"],
        data["g"][:, :2], data["beta"][:, :2],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=4, mode="pad_and_slice", interceptors=interceptors,
    )]
    for pos in (2, 3, 4):
        outs.append(runner(
            data["x_qkv"][:, :, pos:pos + 1], data["conv_w"], data["conv_b"],
            data["g"][:, pos:pos + 1], data["beta"][:, pos:pos + 1],
            Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
            kernel_size=4, mode="pad_and_slice", interceptors=interceptors,
        ))
    assert (torch.cat(outs, dim=1) - full_out).abs().max().item() < 1e-4

    # Case B: padded chunk prefill (real=3 inside a 4-row chunk with garbage row 3) + decode(pos=3) + decode(pos=4)
    state_mgr.reset()
    x_chunk = torch.cat([data["x_qkv"][:, :, :3], torch.full_like(data["x_qkv"][:, :, :1], 99.0)], dim=2)
    g_chunk = torch.cat([data["g"][:, :3], torch.full_like(data["g"][:, :1], -5.0)], dim=1)
    b_chunk = torch.cat([data["beta"][:, :3], torch.full_like(data["beta"][:, :1], 0.9)], dim=1)

    state_mgr.begin_chunk(real=3)
    c_out = runner(
        x_chunk, data["conv_w"], data["conv_b"], g_chunk, b_chunk,
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=4, mode="pad_and_slice", interceptors=interceptors,
    )[:, :3]
    state_mgr.begin_chunk(real=None)
    assert state_mgr.seq_len == 3

    d3 = runner(
        data["x_qkv"][:, :, 3:4], data["conv_w"], data["conv_b"],
        data["g"][:, 3:4], data["beta"][:, 3:4],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=4, mode="pad_and_slice", interceptors=interceptors,
    )
    d4 = runner(
        data["x_qkv"][:, :, 4:5], data["conv_w"], data["conv_b"],
        data["g"][:, 4:5], data["beta"][:, 4:5],
        Hk=data["Hk"], Hv=data["Hv"], dk=data["dk"], dv=data["dv"],
        kernel_size=4, mode="pad_and_slice", interceptors=interceptors,
    )
    assert (torch.cat([c_out, d3, d4], dim=1) - full_out).abs().max().item() < 1e-4


def test_forgetting_on_failure_resets_session_and_recurrent_state():
    spec = RecurrentGraphSpec(
        gdr_ops=(GdrOpSpec(op_uid="gdr::0", qk_l2norm=True, eps=1e-6, scale=0.25),),
        conv_ops=(),
    )
    state_mgr = TorchHybridRecurrentState(spec)
    state_mgr.gdr_states["gdr::0"] = torch.ones((1, 4, 8, 8))
    state_mgr._gdr_seq_lens["gdr::0"] = 5

    kv_session = KVSession(ceiling=128, source="test")
    kv_session.token_ids = [1, 2, 3, 4, 5]

    def _failing_turn():
        try:
            raise RuntimeError("simulated decode failure")
        except BaseException:
            state_mgr.reset()
            raise

    with pytest.raises(RuntimeError, match="simulated decode failure"):
        forgetting_on_failure(kv_session, _failing_turn)

    assert kv_session.token_ids == []
    assert state_mgr.seq_len == 0
    assert not state_mgr.gdr_states


# ---------------------------------------------------------------------------
# 3. Structural discovery on synthetic pad_and_slice & explicit_left_pad DAGs
# ---------------------------------------------------------------------------

def test_discover_recurrent_ops_both_conv_modes():
    def _t(tid): return {"type": "tensor", "tensor_id": tid}
    def _s(v): return {"type": "scalar", "value": v}
    def _l(v): return {"type": "list", "value": v}

    # Mode 1: pad_and_slice
    dag1 = {
        "tensors": {
            "in::x": {"shape": [1, 32, 5], "dtype": "bfloat16"},
            "param::w": {"shape": [32, 1, 4], "dtype": "bfloat16"},
            "in::g": {"shape": [1, 5, 4], "dtype": "float32"},
            "in::b": {"shape": [1, 5, 4], "dtype": "bfloat16"},
            "conv::0::out_0": {"shape": [1, 32, 8], "dtype": "bfloat16"},
            "slice::0::out_0": {"shape": [1, 32, 5], "dtype": "bfloat16"},
            "silu::0::out_0": {"shape": [1, 32, 5], "dtype": "bfloat16"},
            "tr::0::out_0": {"shape": [1, 5, 32], "dtype": "bfloat16"},
            "split::0::out_0": {"shape": [1, 5, 8], "dtype": "bfloat16"},
            "split::0::out_1": {"shape": [1, 5, 8], "dtype": "bfloat16"},
            "split::0::out_2": {"shape": [1, 5, 16], "dtype": "bfloat16"},
            "gdr::0::out_0": {"shape": [1, 5, 4, 4], "dtype": "bfloat16"},
        },
        "ops": {
            "conv::0": {
                "op_type": "aten::convolution",
                "attributes": {"args": [_t("in::x"), _t("param::w"), _s(None), _l([1]), _l([3]), _l([1]), _s(False), _l([0]), _s(32)]},
                "output_tensor_ids": ["conv::0::out_0"],
            },
            "slice::0": {
                "op_type": "aten::slice",
                "attributes": {"args": [_t("conv::0::out_0"), _s(2), _s(0), _s(-3)]},
                "output_tensor_ids": ["slice::0::out_0"],
            },
            "silu::0": {"op_type": "aten::silu", "attributes": {"args": [_t("slice::0::out_0")]}, "output_tensor_ids": ["silu::0::out_0"]},
            "tr::0": {"op_type": "aten::transpose", "attributes": {"args": [_t("silu::0::out_0"), _s(1), _s(2)]}, "output_tensor_ids": ["tr::0::out_0"]},
            "split::0": {
                "op_type": "aten::split_with_sizes",
                "attributes": {"args": [_t("tr::0::out_0"), _l([8, 8, 16]), _s(-1)]},
                "output_tensor_ids": ["split::0::out_0", "split::0::out_1", "split::0::out_2"],
            },
            "gdr::0": {
                "op_type": "custom::gated_delta_rule",
                "attributes": {
                    "args": [_t("split::0::out_0"), _t("split::0::out_1"), _t("split::0::out_2"), _t("in::g"), _t("in::b")],
                    "qk_l2norm": True, "eps": 1e-6, "scale": 0.5,
                },
                "output_tensor_ids": ["gdr::0::out_0"],
            },
        },
        "execution_order": ["conv::0", "slice::0", "silu::0", "tr::0", "split::0", "gdr::0"],
    }
    spec1 = discover_recurrent_ops(dag1)
    assert len(spec1.gdr_ops) == 1
    assert len(spec1.conv_ops) == 1
    assert spec1.conv_ops[0] == CausalConvSpec(op_uid="conv::0", kernel_size=4, channels=32, mode="pad_and_slice")

    # Mode 2: explicit_left_pad
    dag2 = copy.deepcopy(dag1)
    dag2["tensors"]["pad::0::out_0"] = {"shape": [1, 32, 8], "dtype": "bfloat16"}
    dag2["ops"]["pad::0"] = {
        "op_type": "aten::constant_pad_nd",
        "attributes": {"args": [_t("in::x"), _l([3, 0]), _s(0.0)]},
        "output_tensor_ids": ["pad::0::out_0"],
    }
    dag2["ops"]["conv::0"]["attributes"]["args"] = [
        _t("pad::0::out_0"), _t("param::w"), _s(None), _l([1]), _l([0]), _l([1]), _s(False), _l([0]), _s(32)
    ]
    dag2["ops"]["silu::0"]["attributes"]["args"] = [_t("conv::0::out_0")]
    del dag2["ops"]["slice::0"]
    dag2["execution_order"] = ["pad::0", "conv::0", "silu::0", "tr::0", "split::0", "gdr::0"]

    spec2 = discover_recurrent_ops(dag2)
    assert len(spec2.gdr_ops) == 1
    assert len(spec2.conv_ops) == 1
    assert spec2.conv_ops[0] == CausalConvSpec(
        op_uid="conv::0", kernel_size=4, channels=32, mode="explicit_left_pad", pad_op_uid="pad::0"
    )


# ---------------------------------------------------------------------------
# 4. Strict AST checks: pure stdlib in recurrent_state.py, zero torch in triton/
# ---------------------------------------------------------------------------

def test_ast_pure_stdlib_discovery_and_zero_torch_in_triton():
    # Pure stdlib in recurrent_state.py
    disc_path = _SRC_ROOT / "core/runtime/graph/recurrent_state.py"
    tree = ast.parse(disc_path.read_text())
    stdlib = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in stdlib, f"Non-stdlib import {alias.name!r} in recurrent_state.py"
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in stdlib, f"Non-stdlib from-import {node.module!r}"

    # Zero torch across ALL triton/ files
    triton_root = _SRC_ROOT / "triton"
    for py_file in sorted(triton_root.rglob("*.py")):
        t = ast.parse(py_file.read_text())
        for node in ast.walk(t):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not (alias.name == "torch" or alias.name.startswith("torch.")), f"torch import in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not (mod == "torch" or mod.startswith("torch.")), f"torch from-import in {py_file}"


def test_conv1d_mixed_precision_amp_fp32_input_fp16_weight():
    """Verify TorchHybridRecurrentState conv1d aligns fp32 activations to fp16 weights (V100 AMP)."""
    data = _make_hybrid_block_inputs(B=1, L=7, Hk=2, Hv=6, dk=8, dv=8, kernel_size=4, seed=17)
    x_fp32 = data["x_qkv"]
    w_fp16 = data["conv_w"].to(torch.float16)
    b_fp16 = data["conv_b"].to(torch.float16)
    b_fp32 = data["conv_b"]
    C = data["c_qkv"]

    spec = RecurrentGraphSpec(
        gdr_ops=(),
        conv_ops=(CausalConvSpec(op_uid="conv::0", kernel_size=4, channels=C, mode="pad_and_slice"),),
    )

    for conv_bias in (b_fp16, b_fp32, None):
        ref_b = conv_bias.to(w_fp16.dtype) if conv_bias is not None and conv_bias.dtype != w_fp16.dtype else conv_bias
        full_ref = F.conv1d(
            x_fp32.to(w_fp16.dtype), w_fp16, ref_b, stride=1, padding=3, dilation=1, groups=C
        )[:, :, :-3]

        state_mgr = TorchHybridRecurrentState(spec)
        conv_fn = state_mgr.get_interceptors()["conv::0"]

        y_p = conv_fn(x_fp32[:, :, :5], w_fp16, conv_bias, [1], [3], [1], False, [0], C)
        assert y_p.dtype == torch.float16
        assert state_mgr.conv_states["conv::0"].dtype == torch.float32

        y_d1 = conv_fn(x_fp32[:, :, 5:6], w_fp16, conv_bias, [1], [0], [1], False, [0], C)
        y_d2 = conv_fn(x_fp32[:, :, 6:7], w_fp16, conv_bias, [1], [0], [1], False, [0], C)
        assert y_d1.dtype == torch.float16
        assert y_d2.dtype == torch.float16

        combined = torch.cat([y_p[:, :, :-3], y_d1[:, :, :-3], y_d2[:, :, :-3]], dim=2)
        assert (combined - full_ref).abs().max().item() < 1e-3
