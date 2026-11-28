"""Triton-branch execution for ``custom::gated_delta_rule`` (zero torch, zero fallback).

Delegates directly to ``nbxtensor.ops.gated_delta_rule.gated_delta_rule_op``
with float32 compute precision. CPU operands (CPU ``NBXTensor`` or NumPy
``ndarray``) are explicitly refused (ZERO FALLBACK).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from nbxtensor.dtype.nbx_dtype import NBXDtype
import nbxtensor.ops.gated_delta_rule as _nbx_gated_delta_rule


def execute_gated_delta_rule(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state: Optional[Any] = None,
    *,
    qk_l2norm: bool = True,
    eps: float = 1e-6,
    scale: Optional[float] = None,
    return_state: bool = False,
) -> Union[Any, Tuple[Any, Any]]:
    """Execute ``custom::gated_delta_rule`` on GPU via ``nbxtensor.ops.gated_delta_rule.gated_delta_rule_op``."""
    operands = (q, k, v, g, beta) if state is None else (q, k, v, g, beta, state)
    for operand in operands:
        if (
            isinstance(operand, np.ndarray)
            or getattr(operand, "is_cpu", False)
            or getattr(operand, "_device", None) == "cpu"
        ):
            raise RuntimeError(
                "ZERO FALLBACK: custom::gated_delta_rule on the Triton branch requires "
                "GPU NBXTensor operands (CPU fallback is refused)"
            )

    dk = int(q.shape[-1])
    eff_scale = float(dk ** -0.5) if scale is None else float(scale)

    out, state_out = _nbx_gated_delta_rule.gated_delta_rule_op(
        q,
        k,
        v,
        g,
        beta,
        state,
        qk_l2norm=bool(qk_l2norm),
        eps=float(eps),
        scale=eff_scale,
        compute_dtype=NBXDtype.float32,
    )
    return (out, state_out) if return_state else out


execute_gated_delta_rule.self_manages_dtype = True  # type: ignore[attr-defined]
