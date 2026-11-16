"""The causal convolution's contract, read by its wrapper (`conv_wrappers.causal_conv1d_wrapper`) and by its launches
(`launch_keys.causal_conv1d_launches`) before either does anything: what the op takes, its four extents, and a refusal
by name of anything else. Pure: shapes and dtypes only, no tensor, no device."""
from __future__ import annotations

from typing import Optional, Tuple

from nbxtensor.dtype.nbx_dtype import NBXDtype

#: The input dtypes the causal convolution computes in: the depthwise kernel's float family.
CAUSAL_CONV1D_DTYPES = (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32)


def causal_conv1d_contract(x_shape, x: NBXDtype, w_shape, bias_shape=None, state_shape=None,
                           state: Optional[NBXDtype] = None) -> Tuple[int, int, int, int]:
    """(B, C, L, K) of a causal convolution, or a refusal that names what is wrong: an input
    [B, C, L] (L >= 1) of a float dtype, a depthwise weight [C, K] or [C, 1, K] of K >= 2 taps, a
    bias [C], and a state of exactly the last taps-1 input columns [B, C, K-1] at the input's dtype."""
    x_shape, w_shape = tuple(int(d) for d in x_shape), tuple(int(d) for d in w_shape)
    if len(x_shape) != 3:
        raise ValueError(f"causal_conv1d refuses an input of shape {x_shape}: it reads [batch, channels, length]")
    B, C, L = x_shape
    if L < 1:
        raise ValueError(f"causal_conv1d refuses an input of length {L}: it reads at least one column")
    if x not in CAUSAL_CONV1D_DTYPES:
        raise ValueError(f"causal_conv1d refuses an input of dtype {x.name}: it computes in "
                         f"{', '.join(d.name for d in CAUSAL_CONV1D_DTYPES)}")
    if len(w_shape) == 3 and w_shape[1] == 1:
        w_shape = (w_shape[0], w_shape[2])
    if len(w_shape) != 2 or w_shape[0] != C:
        raise ValueError(f"causal_conv1d refuses a weight of shape {w_shape}: it is depthwise, [{C}, taps] "
                         f"or [{C}, 1, taps] for an input of {C} channels")
    K = w_shape[1]
    if K < 2:
        raise ValueError(f"causal_conv1d refuses a weight of {K} taps: its state is taps-1 columns, at least one")
    if bias_shape is not None and tuple(int(d) for d in bias_shape) != (C,):
        raise ValueError(f"causal_conv1d refuses a bias of shape {tuple(bias_shape)}: it is one value per "
                         f"channel, [{C}]")
    if state_shape is not None:
        state_shape = tuple(int(d) for d in state_shape)
        if state_shape != (B, C, K - 1):
            raise ValueError(f"causal_conv1d refuses a state of shape {state_shape}: its state is the last "
                             f"taps-1 input columns, [{B}, {C}, {K - 1}] for {K} taps")
        if state != x:
            raise ValueError(f"causal_conv1d refuses a state of dtype {getattr(state, 'name', state)}: the state "
                             f"is kept at the input's dtype, {x.name}")
    return B, C, L, K
