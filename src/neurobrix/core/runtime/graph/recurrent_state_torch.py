"""PyTorch hybrid recurrent state manager (``conv_state`` + ``recurrent_state``).

Carries per-op causal depthwise 1D convolution rolling windows (``[B, C, K-1]``)
and ``custom::gated_delta_rule`` recurrent state matrices (``[B, Hv, dk, dv]``
in float32) across prefill, chunked prefill, and ``L=1`` autoregressive decode
steps.

All ops are discovered structurally via ``discover_recurrent_ops`` — zero
model-name hardcoding and zero magic layer counts.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from neurobrix.core.runtime.graph.gated_delta_rule_torch import execute_gated_delta_rule_torch
from neurobrix.core.runtime.graph.recurrent_state import (
    CausalConvSpec,
    GdrOpSpec,
    RecurrentGraphSpec,
)


class TorchHybridRecurrentState:
    """Manages recurrent and causal-conv1d state across PyTorch forward calls."""

    def __init__(self, spec: RecurrentGraphSpec) -> None:
        self.spec = spec
        self.gdr_states: Dict[str, torch.Tensor] = {}
        self.conv_states: Dict[str, torch.Tensor] = {}
        self._gdr_seq_lens: Dict[str, int] = {}
        self._active_real_len: Optional[int] = None

    @property
    def seq_len(self) -> int:
        """Number of real tokens ingested into the recurrent state since reset."""
        if not self._gdr_seq_lens:
            return 0
        return min(self._gdr_seq_lens.values())

    def begin_chunk(self, real: Optional[int] = None) -> None:
        """Mark the number of non-padded tokens in the upcoming forward pass."""
        self._active_real_len = int(real) if real is not None else None

    def reset(self) -> None:
        """Clear all recurrent and causal-conv1d states (new sequence / failure)."""
        self.gdr_states.clear()
        self.conv_states.clear()
        self._gdr_seq_lens.clear()
        self._active_real_len = None

    def _make_gdr_interceptor(self, gspec: GdrOpSpec) -> Callable:
        op_uid = gspec.op_uid
        default_l2 = gspec.qk_l2norm
        default_eps = gspec.eps
        default_scale = gspec.scale

        def _intercept_gdr(q, k, v, g, beta, state=None, **kwargs):
            prev = self.gdr_states.get(op_uid) if state is None else state
            qk_l2norm = bool(kwargs.get("qk_l2norm", default_l2))
            eps = float(kwargs.get("eps", default_eps))
            scale = kwargs.get("scale", default_scale)
            seq_l = int(q.shape[1])
            real = self._active_real_len
            if real is not None and 0 < real < seq_l:
                out_real, new_state = execute_gated_delta_rule_torch(
                    q[:, :real], k[:, :real], v[:, :real], g[:, :real], beta[:, :real],
                    prev, qk_l2norm=qk_l2norm, eps=eps, scale=scale, return_state=True,
                )
                out_pad = execute_gated_delta_rule_torch(
                    q[:, real:], k[:, real:], v[:, real:], g[:, real:], beta[:, real:],
                    new_state, qk_l2norm=qk_l2norm, eps=eps, scale=scale, return_state=False,
                )
                out = torch.cat([out_real, out_pad], dim=1)
                ingested = real
            else:
                out, new_state = execute_gated_delta_rule_torch(
                    q, k, v, g, beta, prev,
                    qk_l2norm=qk_l2norm, eps=eps, scale=scale, return_state=True,
                )
                ingested = seq_l

            self.gdr_states[op_uid] = new_state.detach()
            self._gdr_seq_lens[op_uid] = self._gdr_seq_lens.get(op_uid, 0) + ingested
            return out

        _intercept_gdr.self_manages_dtype = True  # type: ignore[attr-defined]
        return _intercept_gdr

    def _make_conv_pad_and_slice_interceptor(self, cspec: CausalConvSpec) -> Callable:
        op_uid = cspec.op_uid
        k_size = cspec.kernel_size
        k_m1 = k_size - 1
        default_groups = cspec.channels

        def _intercept_conv(x, weight, bias=None, *rest, **kwargs):
            # Support both aten::convolution (9 args) and aten::conv1d (7 args)
            if len(rest) >= 6:
                groups = int(rest[5])
            elif len(rest) >= 4:
                groups = int(rest[3])
            else:
                groups = int(kwargs.get("groups", default_groups))

            seq_l = int(x.shape[2])
            real = self._active_real_len
            real_l = real if (real is not None and 0 < real < seq_l) else seq_l
            prev = self.conv_states.get(op_uid)
            bias_conv = bias.to(weight.dtype) if bias is not None and bias.dtype != weight.dtype else bias

            if prev is None:
                x_conv = x.to(weight.dtype) if x.dtype != weight.dtype else x
                y = F.conv1d(x_conv, weight, bias_conv, stride=1, padding=k_m1, dilation=1, groups=groups)
                x_real = x[:, :, :real_l]
                if real_l >= k_m1:
                    new_cs = x_real[:, :, -k_m1:].contiguous().clone()
                else:
                    new_cs = F.pad(x_real, (k_m1 - real_l, 0)).contiguous()
            else:
                prev_t = prev.to(device=x.device, dtype=x.dtype)
                x_cat = torch.cat([prev_t, x], dim=2)
                x_conv = x_cat.to(weight.dtype) if x_cat.dtype != weight.dtype else x_cat
                y_valid = F.conv1d(x_conv, weight, bias_conv, stride=1, padding=0, dilation=1, groups=groups)
                # Right-pad by K-1 so downstream aten::slice(dim=2, 0, -(K-1)) yields y_valid
                y = F.pad(y_valid, (0, k_m1))
                x_cat_real = x_cat[:, :, : k_m1 + real_l]
                new_cs = x_cat_real[:, :, -k_m1:].contiguous().clone()

            self.conv_states[op_uid] = new_cs.detach()
            return y

        return _intercept_conv

    def _make_explicit_left_pad_interceptor(self, cspec: CausalConvSpec) -> Callable:
        op_uid = cspec.op_uid
        k_m1 = cspec.kernel_size - 1

        def _intercept_left_pad(x, pad_list, value=0.0):
            seq_l = int(x.shape[2])
            real = self._active_real_len
            real_l = real if (real is not None and 0 < real < seq_l) else seq_l
            prev = self.conv_states.get(op_uid)

            if prev is None:
                out = F.pad(x, (k_m1, 0), value=float(value))
                x_real = x[:, :, :real_l]
                if real_l >= k_m1:
                    new_cs = x_real[:, :, -k_m1:].contiguous().clone()
                else:
                    new_cs = F.pad(x_real, (k_m1 - real_l, 0), value=float(value)).contiguous()
            else:
                prev_t = prev.to(device=x.device, dtype=x.dtype)
                out = torch.cat([prev_t, x], dim=2)
                x_cat_real = out[:, :, : k_m1 + real_l]
                new_cs = x_cat_real[:, :, -k_m1:].contiguous().clone()

            self.conv_states[op_uid] = new_cs.detach()
            return out

        return _intercept_left_pad

    def get_interceptors(self) -> Dict[str, Callable]:
        """Return ``{op_uid: interceptor_callable}`` for all discovered ops."""
        interceptors: Dict[str, Callable] = {}
        for gspec in self.spec.gdr_ops:
            interceptors[gspec.op_uid] = self._make_gdr_interceptor(gspec)
        for cspec in self.spec.conv_ops:
            if cspec.mode == "pad_and_slice":
                interceptors[cspec.op_uid] = self._make_conv_pad_and_slice_interceptor(cspec)
            elif cspec.mode == "explicit_left_pad" and cspec.pad_op_uid:
                interceptors[cspec.pad_op_uid] = self._make_explicit_left_pad_interceptor(cspec)
        return interceptors


def attach_torch_recurrent_state(executor, dag, kv_session=None, kv_wrapper=None):
    """Discover recurrent ops in ``dag``, attach/reuse state on ``executor``, and register interceptors."""
    from neurobrix.core.runtime.graph.recurrent_state import discover_recurrent_ops
    spec = discover_recurrent_ops(dag)
    if spec.is_empty:
        return None
    prev = executor.__dict__.get("_recurrent_state") if kv_session is not None else None
    if prev is not None and prev.spec == spec:
        state = prev
        if kv_wrapper is None or kv_session.held(executor) is not kv_wrapper:
            state.reset()
    else:
        state = TorchHybridRecurrentState(spec)
        if kv_session is not None:
            executor.__dict__["_recurrent_state"] = state
    executor.register_op_uid_interceptors(state.get_interceptors())
    return state
