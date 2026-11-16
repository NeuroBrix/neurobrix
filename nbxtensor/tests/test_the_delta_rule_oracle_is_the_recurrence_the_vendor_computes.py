"""`nbxtensor.ops.oracles.fp64.delta_rule_oracle`: the float64 reference of the gated delta rule, numpy only, from
un-expanded q, k (Hk heads) and v (Hv heads), with its L2 norm flag, eps and query scale, from a zero or a given state.

Against an outside witness: the vendor's two functions (the recurrent one and the chunked one, the reference
implementations the layer calls), at L = 1 and L = 23, from a zero AND a non-zero state, with Hv / Hk = 1 and > 1. Those
tests read torch and transformers and skip BY NAME where either is absent; the library itself never imports torch. The
bounds are the measured deviations of the vendor's float32 arithmetic on these draws (worst 4.4e-07 recurrent, 9.8e-07
chunked, out or state, 2026-11-16) with a margin of about three: the oracle is the exact side.

Without torch: a prompt cut in two (the oracle's state handed from one call to the next) equals one call, the head
windows above the cap are the whole reference's cut, and value heads that are not a whole multiple of the key heads are
refused by name.

What this test does if the code is wrong: a value head reading the wrong key head (h % Hk instead of h // (Hv / Hk)),
the norm or the scale skipped or applied twice, the read taken BEFORE the write, the decay applied after the write, the
starting state ignored, a window over rows instead of heads - red."""
import numpy as np
import pytest

from _delta_rule_draw import draw
from nbxtensor.ops.oracles import fp64 as O

SCALE = 128 ** -0.5
ATTRS = dict(qk_l2norm=True, eps=1e-6, scale=SCALE)
CASES = [(L, hk, hv, nz) for L in (1, 23) for hk, hv in ((2, 2), (2, 6)) for nz in (False, True)]
IDS = [f"L{L}-Hv{hv}-Hk{hk}-{'state' if nz else 'zeros'}" for L, hk, hv, nz in CASES]
BOUND = {"recurrent": 1.5e-6, "chunked": 3e-6}


def _vendor():
    torch = pytest.importorskip("torch", reason="the vendor's functions need torch: not installed here")
    M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5",
                            reason="the vendor's functions need transformers with the gated delta rule: not installed")
    return torch, {"recurrent": M.torch_recurrent_gated_delta_rule, "chunked": M.torch_chunk_gated_delta_rule}


@pytest.mark.parametrize("L,hk,hv,nonzero", CASES, ids=IDS)
@pytest.mark.parametrize("fn", ["recurrent", "chunked"])
def test_the_oracle_is_the_vendors_function(fn, L, hk, hv, nonzero):
    torch, fns = _vendor()
    q, k, v, g, beta, s = draw(2, L, hk, hv, 128, 128, 11, nonzero)
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    tq, tk = (torch.from_numpy(x).repeat_interleave(hv // hk, dim=2) for x in (q, k))   # the layer expands before
    o, st = fns[fn](tq, tk, *(torch.from_numpy(x) for x in (v, g, beta)),
                    initial_state=None if s is None else torch.from_numpy(s), output_final_state=True,
                    use_qk_l2norm_in_kernel=True)
    dev_o, dev_s = O.oracle_deviation(o.numpy(), ref_o), O.oracle_deviation(st.numpy(), ref_s)
    print(f"{fn} L={L} Hv/Hk={hv // hk} {'state' if nonzero else 'zeros'}: out {dev_o:.2e} state {dev_s:.2e}")
    assert dev_o <= BOUND[fn] and dev_s <= BOUND[fn]


def test_a_starting_state_is_read():
    q, k, v, g, beta, s = draw(1, 4, 2, 4, 16, 8, 3, True)
    with_state = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    from_zeros = O.delta_rule_oracle(q, k, v, g, beta, None, **ATTRS)()
    assert not np.allclose(with_state[0], from_zeros[0]) and not np.allclose(with_state[1], from_zeros[1])


@pytest.mark.parametrize("cut", [1, 9, 22])
def test_a_prompt_cut_in_two_equals_one_pass(cut):
    q, k, v, g, beta, s = draw(2, 23, 2, 6, 32, 16, 5, True)
    whole_o, whole_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    head = O.delta_rule_oracle(*(x[:, :cut] for x in (q, k, v, g, beta)), s, **ATTRS)()
    tail = O.delta_rule_oracle(*(x[:, cut:] for x in (q, k, v, g, beta)), head[1], **ATTRS)()
    np.testing.assert_allclose(np.concatenate([head[0], tail[0]], axis=1), whole_o, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tail[1], whole_s, rtol=1e-12, atol=1e-12)


def test_value_head_h_reads_key_head_h_over_the_group():
    q, k, v, g, beta, s = draw(1, 3, 2, 6, 8, 4, 9, True)
    out, st = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    eo, es = O.delta_rule_oracle(np.repeat(q, 3, axis=2), np.repeat(k, 3, axis=2), v, g, beta, s, **ATTRS)()
    np.testing.assert_array_equal(out, eo)
    np.testing.assert_array_equal(st, es)


def test_above_the_cap_the_windows_are_heads_of_the_whole_reference(monkeypatch):
    q, k, v, g, beta, s = draw(3, 5, 2, 8, 16, 8, 13, True)
    whole_o, whole_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    per_head = 5 * 16 * 8 * 3
    monkeypatch.setattr(O, "ORACLE_MAX_MACS", 6 * per_head)        # two heads a window
    wo, ws = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    assert isinstance(wo, O.HeadWindowedOracle) and isinstance(ws, O.HeadWindowedOracle)
    assert [w for w, _ in wo.blocks] == [(0, 0, 2), (1, 3, 5), (2, 6, 8)]
    assert O.oracle_deviation(whole_o, wo) == 0.0 and O.oracle_deviation(whole_s, ws) == 0.0
    wrong = whole_s.copy()
    wrong[2, 7] += 1.0
    assert O.oracle_deviation(wrong, ws) > 0.0
    assert "batch 1 heads 3-5" in wo.describe


def test_value_heads_not_a_multiple_of_the_key_heads_are_refused_by_name():
    q, k, v, g, beta, _ = draw(1, 1, 2, 5, 8, 4, 1, False)
    with pytest.raises(ValueError, match="5 value heads are not a whole multiple of 2 key heads"):
        O.delta_rule_oracle(q, k, v, g, beta, **ATTRS)
