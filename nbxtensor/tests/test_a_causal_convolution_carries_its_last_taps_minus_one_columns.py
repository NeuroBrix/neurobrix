"""`causal_conv1d_wrapper(x, weight, bias, state) -> (y, state_out)`: the causal depthwise convolution over the sequence
with its state, the last taps-1 INPUT columns [B, C, K-1] at the input's dtype. One pass over a length equals a short
prefill followed by one column at a time, from a zero and from a non-zero state; the state handed back is exactly the
last taps-1 columns of the state joined before the input; what the op does not take is refused by name.

The float64 reference (`oracles.fp64.causal_conv1d_oracle`) and the refusals (`ops._causal_conv1d.causal_conv1d_contract`,
read by the wrapper and by `causal_conv1d_launches`) run on any host; the kernel tests need a GPU device and skip by
name without one (`_no_device.absence`); the vendor witness (torch's conv1d, padded taps-1 and cut to the length)
needs torch and skips by name where it is absent. What this test would catch: a state K columns wide, a state taken
from the output instead of the input, a prefix shorter than the taps that drops the old state, a bias added twice, a
tap read in the wrong order, a state of another dtype silently cast."""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import launch_keys as LK
from nbxtensor.ops._causal_conv1d import causal_conv1d_contract
from nbxtensor.ops.oracles import fp64 as O

from _no_device import absence

F16, BF16, F32 = NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32


def _case(seed, B=2, C=6, L=23, K=4, state=False, bias=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, C, L))
    w = rng.standard_normal((C, K))
    b = rng.standard_normal(C) if bias else None
    s = rng.standard_normal((B, C, K - 1)) if state else None
    return x, w, b, s


def _composed(conv, x, w, b, s, prefill):
    """A prefill of `prefill` columns, then the rest one column at a time, the state carried between calls."""
    ys = []
    y, s = conv(x[:, :, :prefill], w, b, s)
    ys.append(y)
    for t in range(prefill, x.shape[2]):
        y, s = conv(x[:, :, t:t + 1], w, b, s)
        ys.append(y)
    return np.concatenate(ys, axis=2), s


# --- the float64 reference -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_the_reference_is_the_vendor_formula_from_a_zero_start(K, bias):
    x, w, b, _ = _case(1, K=K, bias=bias)
    y, _s = O.causal_conv1d_oracle(x, w, b)
    want = np.zeros_like(x)
    for bi in range(x.shape[0]):
        for c in range(x.shape[1]):
            want[bi, c] = np.convolve(x[bi, c], w[c, ::-1])[:x.shape[2]] + (b[c] if bias else 0.0)
    assert np.abs(y - want).max() < 1e-12


@pytest.mark.parametrize("start", ["zero", "non-zero"])
@pytest.mark.parametrize("prefill", [1, 2, 5])
def test_one_pass_equals_a_short_prefill_then_one_column_at_a_time(start, prefill):
    x, w, b, s = _case(2, state=(start == "non-zero"), bias=True)
    y1, s1 = O.causal_conv1d_oracle(x, w, b, s)
    y2, s2 = _composed(O.causal_conv1d_oracle, x, w, b, s, prefill)
    assert np.abs(y1 - y2).max() < 1e-12
    assert np.array_equal(s1, s2)


@pytest.mark.parametrize("L", [1, 2, 3, 23])
def test_the_state_is_the_last_taps_minus_one_input_columns(L):
    x, w, _, s = _case(3, L=L, K=4, state=True)
    _y, out = O.causal_conv1d_oracle(x, w, None, s)
    joined = np.concatenate([s, x], axis=2)
    assert out.shape == (2, 6, 3)
    assert np.array_equal(out, joined[:, :, -3:])      # shorter than the taps: the old state's tail stays in


def test_the_reference_reads_the_recorded_three_d_weight():
    x, w, _, s = _case(4, state=True)
    a = O.causal_conv1d_oracle(x, w, None, s)
    b = O.causal_conv1d_oracle(x, w[:, None, :], None, s)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# --- the refusals, by name -------------------------------------------------------------------------------------------

def test_a_state_as_wide_as_the_taps_is_refused_by_name():
    with pytest.raises(ValueError, match=r"refuses a state of shape \(2, 6, 4\).*last taps-1 input columns, \[2, 6, 3\]"):
        causal_conv1d_contract((2, 6, 23), F16, (6, 4), None, (2, 6, 4), F16)


def test_a_state_of_another_dtype_is_refused_by_name():
    with pytest.raises(ValueError, match=r"refuses a state of dtype float32: the state is kept at the input's dtype, float16"):
        causal_conv1d_contract((2, 6, 23), F16, (6, 1, 4), None, (2, 6, 3), F32)


@pytest.mark.parametrize("args, says", [
    (((2, 6, 23), NBXDtype.int32, (6, 4)), "refuses an input of dtype int32"),
    (((2, 6), F16, (6, 4)), r"refuses an input of shape \(2, 6\)"),
    (((2, 6, 0), F16, (6, 4)), "refuses an input of length 0"),
    (((2, 6, 23), F16, (6, 2, 4)), r"refuses a weight of shape \(6, 2, 4\): it is depthwise"),
    (((2, 6, 23), F16, (5, 4)), r"refuses a weight of shape \(5, 4\)"),
    (((2, 6, 23), F16, (6, 1)), "refuses a weight of 1 taps"),
    (((2, 6, 23), F16, (6, 4), (5,)), r"refuses a bias of shape \(5,\)"),
])
def test_what_the_op_does_not_take_is_refused_by_name(args, says):
    with pytest.raises(ValueError, match=says):
        causal_conv1d_contract(*args)


def test_the_launches_refuse_what_the_wrapper_refuses():
    with pytest.raises(ValueError, match="refuses an input of dtype int32"):
        LK.causal_conv1d_launches((1, 8, 5), (8, 4), NBXDtype.int32, NBXDtype.int32, None, 1 << 30)


@pytest.mark.parametrize("dt", [F16, BF16, F32])
@pytest.mark.parametrize("L", [1, 23])
def test_the_launches_are_one_depthwise_conv1d_over_the_joined_length(dt, L):
    got = LK.causal_conv1d_launches((1, 64, L), (64, 1, 4), dt, dt, None, 1 << 30)
    assert got == LK.conv_launches((1, 64, L + 3), (64, 1, 4), 1, 0, 1, False, 64, dt, dt, None, 1 << 30)
    assert got and all(k == LK.DEPTHWISE for k, _ in got)


# --- the vendor witness (torch in a test oracle only) ----------------------------------------------------------------

def test_the_reference_equals_the_vendors_padded_conv1d_cut_to_the_length():
    try:
        import torch
    except ImportError:
        pytest.skip("needs torch for the vendor witness: torch is absent on this host")
    x, w, b, _ = _case(5, bias=True)
    want = torch.nn.functional.conv1d(torch.from_numpy(x), torch.from_numpy(w[:, None, :]), torch.from_numpy(b),
                                      padding=w.shape[1] - 1, groups=x.shape[1])[..., :x.shape[2]].numpy()
    y, _s = O.causal_conv1d_oracle(x, w, b)
    assert np.abs(y - want).max() < 1e-12


# --- the kernel, on a device -----------------------------------------------------------------------------------------

_TOL = {F32: 1e-5, F16: 2e-3, BF16: 1e-2}


@pytest.fixture
def device():
    why = absence()
    if why is not None:
        pytest.skip(why)


def _dev(a, dt):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy_as(np.ascontiguousarray(a, dtype=np.float32), dt)


def _host(t):
    return np.asarray(O.host_values(t), dtype=np.float64)


def _run(x, w, b, s, _dt):
    from nbxtensor.ops.conv_wrappers import causal_conv1d_wrapper
    y, so = causal_conv1d_wrapper(x, w, b, s)
    return y, so


@pytest.mark.parametrize("dt", [F32, F16, BF16])
@pytest.mark.parametrize("L", [1, 3, 23, 150])
@pytest.mark.parametrize("start", ["zero", "non-zero"])
def test_the_kernel_matches_the_reference_and_hands_back_its_input_columns(device, dt, L, start):
    x, w, b, s = _case(6, B=2, C=64, L=L, K=4, state=(start == "non-zero"), bias=True)
    xd, wd, bd = _dev(x, dt), _dev(w, dt), _dev(b, dt)
    sd = None if s is None else _dev(s, dt)
    y, so = _run(xd, wd, bd, sd, dt)
    ref_y, ref_s = O.causal_conv1d_oracle(_host(xd), _host(wd), _host(bd), None if sd is None else _host(sd))
    assert tuple(y.shape) == (2, 64, L) and tuple(so.shape) == (2, 64, 3) and so.nbx_dtype == dt
    assert O.oracle_deviation(_host(y), ref_y) <= _TOL[dt]
    assert np.array_equal(_host(so), ref_s)            # a copy of input columns: exact


@pytest.mark.parametrize("dt", [F32, F16, BF16])
@pytest.mark.parametrize("start", ["zero", "non-zero"])
@pytest.mark.parametrize("prefill", [2, 17])
def test_on_the_kernel_one_pass_equals_a_short_prefill_then_one_column_at_a_time(device, dt, start, prefill):
    x, w, b, s = _case(7, B=1, C=64, L=40, K=4, state=(start == "non-zero"), bias=False)
    xd, wd = _dev(x, dt), _dev(w[:, None, :], dt)
    sd = None if s is None else _dev(s, dt)
    y1, s1 = _run(xd, wd, None, sd, dt)
    ys, st = [], sd
    y, st = _run(xd.narrow(2, 0, prefill).contiguous(), wd, None, st, dt)
    ys.append(_host(y))
    for t in range(prefill, 40):
        y, st = _run(xd.narrow(2, t, 1).contiguous(), wd, None, st, dt)
        ys.append(_host(y))
    ref_y, ref_s = O.causal_conv1d_oracle(_host(xd), _host(wd), None, None if sd is None else _host(sd))
    assert O.oracle_deviation(_host(y1), ref_y) <= _TOL[dt]
    assert O.oracle_deviation(np.concatenate(ys, axis=2), ref_y) <= _TOL[dt]
    assert np.array_equal(_host(s1), ref_s) and np.array_equal(_host(st), ref_s)


def test_on_the_kernel_a_state_as_wide_as_the_taps_is_refused_by_name(device):
    x, w, _, _ = _case(8, B=1, C=8, L=5, K=4)
    with pytest.raises(ValueError, match="last taps-1 input columns"):
        _run(_dev(x, F16), _dev(w, F16), None, _dev(np.zeros((1, 8, 4)), F16), F16)
