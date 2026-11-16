"""`gated_delta_rule_wrapper` on a device over a symbolic length: its output and its state against the float64 oracle
(`delta_rule_oracle`) at L = 1, 23, 63, 64, 65 and 200 (both sides of the 64 a chunked form would cut at: the scan has
no chunk, the lengths prove it), from a non-zero state at every length and from zeros at L = 1 and 23, with Hv / Hk = 1
and > 1 read un-expanded, inputs at float32, float16 and bfloat16 (bfloat16 where the device has it natively), q, k
and v handed as views of one projection (strided, never copied). Beside it, on the device, the three identities of the
delta rule: a prompt cut anywhere, or a prefill followed by single tokens, is one pass; a zero-padded tail leaves the
state and the outputs exactly as they were; and above the oracle's cap the reference windows over value heads and
batch rows, every row of the length kept.

A card test: on a host with no device it skips by name (the fixture below asks the one door; `_no_device.py`). It runs
on a V100 card job and on the Mac (Metal), the same file.

THE BOUNDS, stated from what was measured (2026-11-16, this lot's proof folder, `measure_tolerances.out`): a float32
emulation of the kernel's order of operations over L = 1 to 200, against the oracle, worst over 8 draws a length: out
5.7e-07, state 8.9e-07. The device's bound is about four times that, out and state 4e-06 (the card's fused
multiply-adds and its own sqrt and exp round differently from numpy). An output stored at float16 or bfloat16 adds at
most its rounding, half an ulp relative to the largest value: 2^-11 and 2^-8; the state is float32 at every input
dtype. Each test prints its measured deviation: the card's log is the measurement of these bounds.

What this test does if the code is wrong: the loop stopped short of L or run past it, a token's stride misread, the
state not carried between tokens or narrowed, a value head reading the wrong key head, the state handed back not the
next call's start, a padded token that moves the state - red."""
import numpy as np
import pytest

from _delta_rule_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.oracles import fp64 as O

F32 = NBXDtype.float32
SCALE = 128 ** -0.5
ATTRS = dict(qk_l2norm=True, eps=1e-6, scale=SCALE)
OUT_F32, STATE = 4e-6, 4e-6
ROUNDING = {NBXDtype.float32: 0.0, NBXDtype.float16: 2.0 ** -11, NBXDtype.bfloat16: 2.0 ** -8}
CASES = ([(L, hk, hv, 1) for L in (1, 23, 63, 64, 65, 200) for hk, hv in ((4, 4), (2, 6))] +
         [(L, hk, hv, 0) for L in (1, 23) for hk, hv in ((4, 4), (2, 6))])
IDS = [f"L{L}-Hv{hv}-Hk{hk}-{'state' if nz else 'zeros'}" for L, hk, hv, nz in CASES]


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _to_device(a, dt):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device="cuda:0").to(dt)


def _host(t):
    return O.host_values(t.to(F32)).astype(np.float64)


def _rounded(a, dt):
    """The numbers a float32 draw carries once stored at `dt` (the oracle reads what the kernel reads)."""
    return _host(_to_device(a, dt))


def _operands(q, k, v, g, beta, s, dt):
    """q, k, v as views of ONE projection [B, L, (2 Hk dk + Hv dv)] at `dt` (strided, un-expanded), beta at `dt`, g and
    the state float32."""
    B, L, hk, dk = q.shape
    hv, dv = v.shape[2:]
    proj = _to_device(np.concatenate([q.reshape(B, L, -1), k.reshape(B, L, -1), v.reshape(B, L, -1)], -1), dt)
    n = hk * dk
    return (proj[:, :, :n].unflatten(2, (hk, dk)), proj[:, :, n:2 * n].unflatten(2, (hk, dk)),
            proj[:, :, 2 * n:].unflatten(2, (hv, dv)), _to_device(g, F32), _to_device(beta, dt),
            None if s is None else _to_device(s, F32))


def _run(ops):
    from nbxtensor.ops.gated_delta_rule import gated_delta_rule_wrapper
    return gated_delta_rule_wrapper(*ops, compute_dtype=F32, **ATTRS)


@pytest.mark.parametrize("dt", [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("L,hk,hv,nonzero", CASES, ids=IDS)
def test_the_scan_over_a_length_is_the_oracle(dt, L, hk, hv, nonzero):
    from nbxtensor.ops.wrappers import has_native_bf16
    if dt == NBXDtype.bfloat16 and not has_native_bf16():
        pytest.skip("bfloat16: this device has no native bfloat16")
    q, k, v, g, beta, s = draw(2, L, hk, hv, 128, 128, 200 + L, bool(nonzero), decay_max=0.1)
    out, state = _run(_operands(q, k, v, g, beta, s, dt))
    assert out.nbx_dtype == dt and tuple(out.shape) == (2, L, hv, 128)
    assert state.nbx_dtype == F32 and tuple(state.shape) == (2, hv, 128, 128)
    ref_o, ref_s = O.delta_rule_oracle(_rounded(q, dt), _rounded(k, dt), _rounded(v, dt), g, _rounded(beta, dt), s,
                                       **ATTRS)()
    dev_o, dev_s = O.oracle_deviation(_host(out), ref_o), O.oracle_deviation(_host(state), ref_s)
    print(f"{dt.name} L={L} Hv/Hk={hv // hk} {'state' if nonzero else 'zeros'}: "
          f"out {dev_o:.2e} (bound {OUT_F32 + ROUNDING[dt]:.1e}) state {dev_s:.2e} (bound {STATE:.1e})")
    assert dev_o <= OUT_F32 + ROUNDING[dt]
    assert dev_s <= STATE


@pytest.mark.parametrize("cuts", [[23], [64], [23, 64, 100], [100] + list(range(101, 150))],
                         ids=["cut23", "cut64", "cuts23-64-100", "prefill100-then-tokens"])
def test_a_prompt_in_several_calls_is_one_pass(cuts):
    q, k, v, g, beta, s = draw(1, 150, 2, 6, 128, 128, 300, True, decay_max=0.1)
    one_o, one_s = _run(_operands(q, k, v, g, beta, s, F32))
    state, outs = _to_device(s, F32), []
    for a, b in zip([0] + cuts, cuts + [150]):
        ops = _operands(*(x[:, a:b] for x in (q, k, v, g, beta)), None, F32)
        o, state = _run(ops[:5] + (state,))
        outs.append(_host(o))
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    parts_o = np.concatenate(outs, 1)
    dev = {"parts out": O.oracle_deviation(parts_o, ref_o), "parts state": O.oracle_deviation(_host(state), ref_s),
           "one out": O.oracle_deviation(_host(one_o), ref_o), "one state": O.oracle_deviation(_host(one_s), ref_s),
           "parts against one": O.oracle_deviation(parts_o, _host(one_o))}
    print(f"cuts {cuts[:4]}{'...' if len(cuts) > 4 else ''}: " + ", ".join(f"{n} {d:.2e}" for n, d in dev.items()))
    assert max(dev["parts out"], dev["one out"], dev["parts against one"]) <= OUT_F32
    assert max(dev["parts state"], dev["one state"]) <= STATE


@pytest.mark.parametrize("L", [23, 65])
def test_a_zero_padded_tail_leaves_the_state_and_the_outputs_exactly(L):
    q, k, v, g, beta, s = draw(2, L, 2, 6, 128, 128, 400 + L, True)
    pad = -L % 64
    padded = [np.concatenate([x, np.zeros((2, pad) + x.shape[2:], np.float32)], 1) for x in (q, k, v, g, beta)]
    o, st = _run(_operands(q, k, v, g, beta, s, F32))
    po, pst = _run(_operands(*padded, s, F32))
    np.testing.assert_array_equal(_host(pst), _host(st))
    np.testing.assert_array_equal(_host(po)[:, :L], _host(o))


def test_above_the_cap_the_oracle_windows_heads_and_batch_rows_and_keeps_every_row(monkeypatch):
    L = 200
    q, k, v, g, beta, s = draw(3, L, 2, 8, 128, 128, 500, True, decay_max=0.1)
    out, state = _run(_operands(q, k, v, g, beta, s, F32))
    monkeypatch.setattr(O, "ORACLE_MAX_MACS", 6 * L * 128 * 128 * 3)        # two heads a window
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    assert isinstance(ref_o, O.HeadWindowedOracle) and isinstance(ref_s, O.HeadWindowedOracle)
    assert all(ref.shape == (1, L, 2, 128) for _, ref in ref_o.blocks)
    dev_o, dev_s = O.oracle_deviation(_host(out), ref_o), O.oracle_deviation(_host(state), ref_s)
    print(f"windowed {ref_o.describe}: out {dev_o:.2e} state {dev_s:.2e}")
    assert dev_o <= OUT_F32 and dev_s <= STATE
