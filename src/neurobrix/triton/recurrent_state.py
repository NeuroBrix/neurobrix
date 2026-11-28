"""Triton zero-Torch hybrid recurrent state manager (``conv_state`` + ``recurrent_state``).

Structural twin of ``core/runtime/graph/recurrent_state_torch.py`` with ZERO
``torch`` imports and ZERO CPU fallback. All tensor operations execute on GPU
``NBXTensor`` via ``nbxtensor`` ops.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from nbxtensor.ops import conv_wrappers as _nbx_conv
from nbxtensor.ops import shape_wrappers as _nbx_shape
from nbxtensor.tensor import NBXTensor

from neurobrix.core.runtime.graph.recurrent_state import (
    CausalConvSpec,
    GdrOpSpec,
    RecurrentGraphSpec,
)
from neurobrix.triton.gated_delta_rule import execute_gated_delta_rule


def _default_cat(tensors, dim: int):
    return NBXTensor.cat(tensors, dim)


def _default_narrow_clone(x: Any, dim: int, start: int, length: int) -> Any:
    return x.narrow(dim, start, length).contiguous().clone()


def _default_pad(x: Any, pad_list: list, value: float = 0.0) -> Any:
    return _nbx_shape.constant_pad_nd_wrapper(x, pad_list, value)


def _default_conv1d(x: Any, weight: Any, bias: Any, stride: int, padding: int, dilation: int, groups: int) -> Any:
    return _nbx_conv.conv1d_wrapper(x, weight, bias, stride, padding, dilation, groups)


class TritonHybridRecurrentState:
    """Manages recurrent and causal-conv1d state across Triton forward calls."""

    def __init__(
        self,
        spec: RecurrentGraphSpec,
        *,
        _gdr_fn: Optional[Callable] = None,
        _conv1d_fn: Optional[Callable] = None,
        _pad_fn: Optional[Callable] = None,
        _cat_fn: Optional[Callable] = None,
        _narrow_clone_fn: Optional[Callable] = None,
    ) -> None:
        self.spec = spec
        self.gdr_states: Dict[str, Any] = {}
        self.conv_states: Dict[str, Any] = {}
        self._gdr_seq_lens: Dict[str, int] = {}
        self._active_real_len: Optional[int] = None
        self._gdr_fn = _gdr_fn or execute_gated_delta_rule
        self._conv1d_fn = _conv1d_fn or _default_conv1d
        self._pad_fn = _pad_fn or _default_pad
        self._cat_fn = _cat_fn or _default_cat
        self._narrow_clone_fn = _narrow_clone_fn or _default_narrow_clone

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
                out_real, new_state = self._gdr_fn(
                    self._narrow_clone_fn(q, 1, 0, real),
                    self._narrow_clone_fn(k, 1, 0, real),
                    self._narrow_clone_fn(v, 1, 0, real),
                    self._narrow_clone_fn(g, 1, 0, real),
                    self._narrow_clone_fn(beta, 1, 0, real),
                    prev,
                    qk_l2norm=qk_l2norm,
                    eps=eps,
                    scale=scale,
                    return_state=True,
                )
                rem = seq_l - real
                out_pad = self._gdr_fn(
                    self._narrow_clone_fn(q, 1, real, rem),
                    self._narrow_clone_fn(k, 1, real, rem),
                    self._narrow_clone_fn(v, 1, real, rem),
                    self._narrow_clone_fn(g, 1, real, rem),
                    self._narrow_clone_fn(beta, 1, real, rem),
                    new_state,
                    qk_l2norm=qk_l2norm,
                    eps=eps,
                    scale=scale,
                    return_state=False,
                )
                out = self._cat_fn([out_real, out_pad], 1)
                ingested = real
            else:
                out, new_state = self._gdr_fn(
                    q, k, v, g, beta, prev,
                    qk_l2norm=qk_l2norm,
                    eps=eps,
                    scale=scale,
                    return_state=True,
                )
                ingested = seq_l

            self.gdr_states[op_uid] = new_state
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

            if prev is None:
                y = self._conv1d_fn(x, weight, bias, 1, k_m1, 1, groups)
                if real_l >= k_m1:
                    new_cs = self._narrow_clone_fn(x, 2, real_l - k_m1, k_m1)
                else:
                    x_real = self._narrow_clone_fn(x, 2, 0, real_l)
                    new_cs = self._pad_fn(x_real, [k_m1 - real_l, 0], 0.0)
            else:
                x_cat = self._cat_fn([prev, x], 2)
                y_valid = self._conv1d_fn(x_cat, weight, bias, 1, 0, 1, groups)
                # Right-pad by K-1 so downstream aten::slice(dim=2, 0, -(K-1)) yields y_valid
                y = self._pad_fn(y_valid, [0, k_m1], 0.0)
                new_cs = self._narrow_clone_fn(x_cat, 2, real_l, k_m1)

            self.conv_states[op_uid] = new_cs
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
                out = self._pad_fn(x, [k_m1, 0], float(value))
                if real_l >= k_m1:
                    new_cs = self._narrow_clone_fn(x, 2, real_l - k_m1, k_m1)
                else:
                    x_real = self._narrow_clone_fn(x, 2, 0, real_l)
                    new_cs = self._pad_fn(x_real, [k_m1 - real_l, 0], float(value))
            else:
                out = self._cat_fn([prev, x], 2)
                new_cs = self._narrow_clone_fn(out, 2, real_l, k_m1)

            self.conv_states[op_uid] = new_cs
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


def attach_triton_recurrent_state(executor, dag, kv_session=None, kv_interceptor=None):
    """Discover recurrent ops in ``dag``, attach/reuse state on ``executor``, and register interceptors."""
    from neurobrix.core.runtime.graph.recurrent_state import discover_recurrent_ops
    spec = discover_recurrent_ops(dag)
    if spec.is_empty:
        return None
    prev = executor.__dict__.get("_recurrent_state") if kv_session is not None else None
    if prev is not None and prev.spec == spec:
        state = prev
        if kv_interceptor is None or kv_session.held(executor) is not kv_interceptor.cache:
            state.reset()
    else:
        state = TritonHybridRecurrentState(spec)
        if kv_session is not None:
            executor.__dict__["_recurrent_state"] = state
    executor.register_op_uid_interceptors(state.get_interceptors())
    return state


def make_triton_gdr_op(return_state: bool):
    """Return a compiled-sequence callable for custom::gated_delta_rule."""
    from neurobrix.triton.gated_delta_rule import execute_gated_delta_rule

    def _gdr_triton_fn(*args, _rs=return_state, **kwargs):
        return execute_gated_delta_rule(*args, return_state=_rs, **kwargs)
    _gdr_triton_fn.self_manages_dtype = True  # type: ignore[attr-defined]
    return _gdr_triton_fn
