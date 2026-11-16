"""`gated_delta_rule_wrapper` on a device at one token: its output and its state against the float64 oracle
(`delta_rule_oracle`), from a zero AND a non-zero state, with Hv / Hk = 1 and > 1, inputs at float32, float16 and
bfloat16 (bfloat16 where the device has it natively), g and the state float32; q, k and v handed as views of one
projection (strided, as a layer splits them), never copied.

A card test: on a host with no device it skips by name (`_no_device.py`; the fixture below asks the one door). It runs
on a V100 card job and on the Mac (Metal), the same file.

THE BOUNDS, stated from what was measured (2026-11-16, this lot's proof folder, `measure_tolerances.out`): a float32
emulation of the kernel's order of operations at one token, against the oracle, worst over 8 draws: out 1.05e-06,
state 1.53e-07. The device's bound is four times that (out 4e-06, state 1e-06: the card's fused multiply-adds and its
own sqrt and exp round differently from numpy). An output stored at float16 or bfloat16 adds at most its rounding, half
an ulp relative to the largest value: 2^-11 (float16) and 2^-8 (bfloat16); the state is float32 at every input dtype.
Each test prints its measured deviation: the card's log is the measurement of these bounds.

What this test does if the code is wrong: a value head reading the wrong key head, the norm or the scale missing, the
read before the write, the state not read or narrowed, an output at the wrong dtype or layout, a dv tile's edge
dropped (dv = 96 is not a multiple of the larger tiles) - red."""
import numpy as np
import pytest

from _delta_rule_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.oracles import fp64 as O

SCALE = 128 ** -0.5
OUT_F32, STATE = 4e-6, 1e-6
ROUNDING = {NBXDtype.float32: 0.0, NBXDtype.float16: 2.0 ** -11, NBXDtype.bfloat16: 2.0 ** -8}
CASES = [(hk, hv, dk, dv, nz) for hk, hv in ((4, 4), (2, 6)) for dk, dv in ((128, 128), (128, 96)) for nz in (0, 1)]
IDS = [f"Hv{hv}-Hk{hk}-dk{dk}-dv{dv}-{'state' if nz else 'zeros'}" for hk, hv, dk, dv, nz in CASES]


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _to_device(a, dt):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device="cuda:0").to(dt)


def _host(t):
    return O.host_values(t.to(NBXDtype.float32)).astype(np.float64)


def _rounded(a, dt):
    """The numbers a float32 draw carries once stored at `dt` (the oracle reads what the kernel reads)."""
    return _host(_to_device(a, dt))


@pytest.mark.parametrize("dt", [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("hk,hv,dk,dv,nonzero", CASES, ids=IDS)
def test_one_token_is_the_oracle(dt, hk, hv, dk, dv, nonzero):
    from nbxtensor.ops.wrappers import has_native_bf16
    from nbxtensor.ops.gated_delta_rule import gated_delta_rule_wrapper
    if dt == NBXDtype.bfloat16 and not has_native_bf16():
        pytest.skip("bfloat16: this device has no native bfloat16")
    q, k, v, g, beta, s = draw(2, 1, hk, hv, dk, dv, 21, bool(nonzero))
    proj = _to_device(np.concatenate([q.reshape(2, 1, -1), k.reshape(2, 1, -1), v.reshape(2, 1, -1)], -1), dt)
    nq = hk * dk
    dq = proj[:, :, :nq].unflatten(2, (hk, dk))
    dk_ = proj[:, :, nq:2 * nq].unflatten(2, (hk, dk))
    dv_ = proj[:, :, 2 * nq:].unflatten(2, (hv, dv))
    db = _to_device(beta, dt)
    dg = _to_device(g, NBXDtype.float32)
    ds = None if s is None else _to_device(s, NBXDtype.float32)
    out, state = gated_delta_rule_wrapper(dq, dk_, dv_, dg, db, ds, qk_l2norm=True, eps=1e-6, scale=SCALE,
                                          compute_dtype=NBXDtype.float32)
    assert out.nbx_dtype == dt and tuple(out.shape) == (2, 1, hv, dv)
    assert state.nbx_dtype == NBXDtype.float32 and tuple(state.shape) == (2, hv, dk, dv)
    ref_o, ref_s = O.delta_rule_oracle(_rounded(q, dt), _rounded(k, dt), _rounded(v, dt), g, _rounded(beta, dt), s,
                                       qk_l2norm=True, eps=1e-6, scale=SCALE)()
    dev_o, dev_s = O.oracle_deviation(_host(out), ref_o), O.oracle_deviation(_host(state), ref_s)
    print(f"{dt.name} Hv/Hk={hv // hk} dk={dk} dv={dv} {'state' if nonzero else 'zeros'}: "
          f"out {dev_o:.2e} (bound {OUT_F32 + ROUNDING[dt]:.1e}) state {dev_s:.2e} (bound {STATE:.1e})")
    assert dev_o <= OUT_F32 + ROUNDING[dt]
    assert dev_s <= STATE


def test_the_state_handed_back_is_the_next_tokens_start():
    from nbxtensor.ops.gated_delta_rule import gated_delta_rule_wrapper
    q, k, v, g, beta, s = draw(1, 2, 2, 6, 128, 128, 33, True)
    F32 = NBXDtype.float32
    state = _to_device(s, F32)
    outs = []
    for t in range(2):
        o, state = gated_delta_rule_wrapper(*(_to_device(x[:, t:t + 1], F32) for x in (q, k, v, g, beta)), state,
                                            qk_l2norm=True, eps=1e-6, scale=SCALE, compute_dtype=F32)
        outs.append(_host(o))
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, qk_l2norm=True, eps=1e-6, scale=SCALE)()
    dev_o = O.oracle_deviation(np.concatenate(outs, axis=1), ref_o)
    dev_s = O.oracle_deviation(_host(state), ref_s)
    print(f"two tokens, the state handed over: out {dev_o:.2e} state {dev_s:.2e}")
    assert dev_o <= OUT_F32 and dev_s <= STATE
