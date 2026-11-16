"""The gated delta rule over a symbolic length, with its state passed in and handed back: what is answered without a
device. `gated_delta_rule_wrapper` takes any length from one token (the L = 1 refusal of the token-step release is
gone), `gated_delta_rule_launches` answers one key whatever the length and batch, `_cost.gated_delta_rule` declares the
op's two outputs, its bytes and its flops, and refuses what the wrapper refuses. The three identities of the delta rule
(the chunked closed form IS the recurrence, the zero padding of a chunk is neutral, a prompt cut anywhere composes)
hold on the float64 reference at L = 23, 63, 64, 65, 150 and 200, and the reference windows over value heads and batch
rows, never over output rows.

Red before this lot: a length above one is refused by name ("this release runs the delta rule at the lengths (1,)")
and `_cost.gated_delta_rule` does not exist. What this test does if the code is wrong: a length in the launch key or a
chunk size in a signature, a cost whose outputs are at the wrong dtype or whose flops do not grow with L, a cost that
accepts what the wrapper refuses, a reference whose state depends on padding, or whose two calls differ from one - red.
No device needed."""
import inspect

import numpy as np
import pytest

from _delta_rule_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import _cost as C
from nbxtensor.ops import gated_delta_rule as G
from nbxtensor.ops import launch_keys as LK
from nbxtensor.ops.oracles import fp64 as O
from nbxtensor.tensor import NBXTensor

F16, BF16, F32 = NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32
SCALE = 128 ** -0.5
ATTRS = dict(qk_l2norm=True, eps=1e-6, scale=SCALE)
LENGTHS = (1, 23, 63, 64, 65, 200)


def _t(shape, dt=F16):
    return NBXTensor.empty_cpu(tuple(shape), dtype=dt)


def _specs(B=2, L=23, Hk=2, Hv=6, dk=128, dv=96, dt="float16", state=True):
    s = C.Spec.contiguous
    return [s((B, L, Hk, dk), dt), s((B, L, Hk, dk), dt), s((B, L, Hv, dv), dt), s((B, L, Hv), "float32"),
            s((B, L, Hv), dt), s((B, Hv, dk, dv), "float32") if state else None]


@pytest.mark.parametrize("L", LENGTHS)
def test_any_length_from_one_token_is_taken(L):
    ops = [_t((2, L, 2, 16)), _t((2, L, 2, 16)), _t((2, L, 6, 8)), _t((2, L, 6), F32), _t((2, L, 6)),
           _t((2, 6, 16, 8), F32)]
    assert G.gated_delta_rule_shapes(*ops) == (2, L, 2, 6, 16, 8)
    assert G.gated_delta_rule_dtypes(*(None if t is None else t.nbx_dtype for t in ops), F32) == F32


def test_neither_the_length_nor_a_chunk_size_is_an_argument_or_a_key():
    assert list(inspect.signature(LK.gated_delta_rule_launches).parameters) == ["dk", "dv", "q", "k", "v", "g", "beta"]
    for fn in (G.gated_delta_rule_wrapper, C.gated_delta_rule, O.delta_rule_oracle):
        assert not [p for p in inspect.signature(fn).parameters if "chunk" in p], fn.__name__
    key = LK.gated_delta_rule_launches(128, 128, BF16, BF16, BF16, F32, BF16)
    assert len(key) == 1 and key[0][0] == LK.GATED_DELTA_RULE
    assert key == LK.gated_delta_rule_launches(128, 128, BF16, BF16, BF16, F32, BF16)   # pure: same facts, same key


@pytest.mark.parametrize("dt", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("state", [True, False], ids=["state", "zeros"])
def test_the_cost_is_the_output_and_the_state_with_no_workspace(dt, state):
    B, L, Hk, Hv, dk, dv = 2, 23, 2, 6, 128, 96
    specs = _specs(B, L, Hk, Hv, dk, dv, dt, state)
    c = C.gated_delta_rule(*specs, qk_l2norm=True, compute_dtype=F32)
    size = {"float16": 2, "bfloat16": 2, "float32": 4}[dt]
    assert [(b.name, b.nbytes) for b in c.outputs] == [("output", B * L * Hv * dv * size),
                                                       ("state out", B * Hv * dk * dv * 4)]
    assert c.workspace == ()
    assert c.peak_bytes == B * L * Hv * dv * size + B * Hv * dk * dv * 4
    read = sum(s.nbytes for s in specs if s is not None)
    assert c.bytes_moved == read + B * L * Hv * dv * size + B * Hv * dk * dv * 4
    assert c.flops == B * L * Hv * (7 * dk * dv + 2 * dv + dk + 2 * (3 * dk + 1))


def test_the_cost_grows_with_the_length_and_the_state_in_is_read_once():
    at = {L: C.gated_delta_rule(*_specs(L=L), qk_l2norm=False, compute_dtype=F32) for L in (1, 200)}
    assert at[200].flops == 200 * at[1].flops
    assert at[1].flops == 2 * 6 * (7 * 128 * 96 + 2 * 96 + 128)
    zeros = C.gated_delta_rule(*_specs(L=1, state=False), qk_l2norm=False, compute_dtype=F32)
    assert at[1].bytes_moved - zeros.bytes_moved == 2 * 6 * 128 * 96 * 4


@pytest.mark.parametrize("at,dt,kw,match", [
    (None, None, dict(compute_dtype=F16), "compute_dtype float16 is refused"),
    (5, "float16", {}, "state is float16; it must be float32"),
    (3, "bfloat16", {}, "g is bfloat16; it must be float32"),
    (0, "float64", {}, "q is float64; it must be one of"),
])
def test_the_cost_refuses_by_name_what_the_wrapper_refuses(at, dt, kw, match):
    specs = _specs()
    if at is not None:
        specs[at] = C.Spec.contiguous(specs[at].shape, dt)
    with pytest.raises(ValueError, match=match):
        C.gated_delta_rule(*specs, **dict(dict(qk_l2norm=True, compute_dtype=F32), **kw))


def test_the_cost_refuses_a_state_of_the_wrong_shape_and_a_sequence_of_no_token_by_name():
    specs = _specs()
    specs[5] = C.Spec.contiguous((2, 6, 96, 128), "float32")
    with pytest.raises(ValueError, match="state has shape \\(2, 6, 96, 128\\); .* ask \\(2, 6, 128, 96\\)"):
        C.gated_delta_rule(*specs, qk_l2norm=True, compute_dtype=F32)
    with pytest.raises(ValueError, match="a sequence of length 0: the delta rule runs over at least 1 token"):
        C.gated_delta_rule(*_specs(L=0), qk_l2norm=True, compute_dtype=F32)


# --- the three identities of the delta rule, on the float64 reference -------------------------------------------

def _chunked(q, k, v, g, beta, state, chunk):
    """The closed form solved `chunk` positions at a time, in float64, the sequence padded with zeros to a whole number
    of chunks (k = v = 0, beta = 0, g = 0) and the padded outputs cut: the vendor's chunked algebra, written here from
    the recurrence's induction, never from the vendor's code."""
    B, L, Hk, dk = q.shape
    Hv, dv = v.shape[2:]
    rep = Hv // Hk
    q, k = (np.repeat(x.astype(np.float64), rep, axis=2) for x in (q, k))
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * SCALE
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    pad = -L % chunk
    q, k, v, g, beta = (np.concatenate([x.astype(np.float64), np.zeros((B, pad) + x.shape[2:])], 1)
                        for x in (q, k, v, g, beta))
    S = np.zeros((B, Hv, dk, dv)) if state is None else state.astype(np.float64).copy()
    out = np.empty(v.shape)
    for c0 in range(0, L + pad, chunk):
        sl = slice(c0, c0 + chunk)
        for b in range(B):
            for h in range(Hv):
                qc, kc, vc = q[b, sl, h], k[b, sl, h], v[b, sl, h]
                bc, cs = beta[b, sl, h], np.cumsum(g[b, sl, h])
                D = np.tril(np.exp(cs[:, None] - cs[None, :]))
                A = np.tril(bc[:, None] * (kc @ kc.T) * D, -1)
                rhs = bc[:, None] * vc - (bc * np.exp(cs))[:, None] * (kc @ S[b, h])
                U = np.linalg.solve(np.eye(chunk) + A, rhs)
                out[b, sl, h] = np.exp(cs)[:, None] * (qc @ S[b, h]) + ((qc @ kc.T) * D) @ U
                S[b, h] = np.exp(cs[-1]) * S[b, h] + (np.exp(cs[-1] - cs)[:, None] * kc).T @ U
    return out[:, :L], S


@pytest.mark.parametrize("L", [23, 63, 64, 65, 200])
def test_the_chunked_closed_form_is_the_recurrence(L):
    q, k, v, g, beta, s = draw(1, L, 2, 4, 32, 16, 40 + L, True, decay_max=0.1)
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    co, cs = _chunked(q, k, v, g, beta, s, 64)
    assert O.oracle_deviation(co, ref_o) < 1e-10 and O.oracle_deviation(cs, ref_s) < 1e-10


@pytest.mark.parametrize("L", [23, 65])
def test_zero_padding_leaves_the_state_and_the_outputs_unchanged(L):
    q, k, v, g, beta, s = draw(2, L, 2, 6, 16, 8, 50 + L, True)
    pad = -L % 64
    padded = [np.concatenate([x, np.zeros((2, pad) + x.shape[2:], np.float32)], 1) for x in (q, k, v, g, beta)]
    ro, rs = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    po, ps = O.delta_rule_oracle(*padded, s, **ATTRS)()
    np.testing.assert_array_equal(ps, rs)
    np.testing.assert_array_equal(po[:, :L], ro)


def test_a_prompt_cut_anywhere_and_a_prefill_then_single_tokens_compose_into_one_pass():
    q, k, v, g, beta, s = draw(1, 150, 2, 4, 32, 16, 60, True, decay_max=0.1)
    whole_o, whole_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    for cuts in ([1], [23], [64], [100], [23, 64, 100], [100] + list(range(101, 150))):
        state, outs = s, []
        for a, b in zip([0] + cuts, cuts + [150]):
            o, state = O.delta_rule_oracle(*(x[:, a:b] for x in (q, k, v, g, beta)), state, **ATTRS)()
            outs.append(o)
        np.testing.assert_allclose(np.concatenate(outs, 1), whole_o, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(state, whole_s, rtol=1e-12, atol=1e-12)


def test_at_length_the_reference_windows_over_heads_and_batch_rows_never_over_rows(monkeypatch):
    L = 200
    q, k, v, g, beta, s = draw(3, L, 2, 8, 16, 8, 70, True)
    whole_o, whole_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    monkeypatch.setattr(O, "ORACLE_MAX_MACS", 6 * L * 16 * 8 * 3)        # two heads a window
    wo, ws = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    assert isinstance(wo, O.HeadWindowedOracle) and isinstance(ws, O.HeadWindowedOracle)
    assert [w for w, _ in wo.blocks] == [(0, 0, 2), (1, 3, 5), (2, 6, 8)]
    assert all(ref.shape == (1, L, 2, 8) for _, ref in wo.blocks)          # every row of the length kept
    assert O.oracle_deviation(whole_o, wo) == 0.0 and O.oracle_deviation(whole_s, ws) == 0.0


# --- an outside witness: the vendor's two functions at the lengths around a chunk (torch, tests only) ------------

VENDOR_BOUND = {"recurrent": 1.5e-6, "chunked": 2e-6}   # measured worst 5.9e-07 and 5.6e-07 (2026-11-16), a margin of about three


@pytest.mark.parametrize("L", [63, 64, 65, 200])
@pytest.mark.parametrize("fn", ["recurrent", "chunked"])
def test_the_reference_over_a_length_is_the_vendors_function(fn, L):
    torch = pytest.importorskip("torch", reason="the vendor's functions need torch: not installed here")
    M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5",
                            reason="the vendor's functions need transformers with the gated delta rule: not installed")
    vendor = {"recurrent": M.torch_recurrent_gated_delta_rule, "chunked": M.torch_chunk_gated_delta_rule}[fn]
    q, k, v, g, beta, s = draw(1, L, 2, 6, 128, 128, 80 + L, True, decay_max=0.1)
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    tq, tk = (torch.from_numpy(x).repeat_interleave(3, dim=2) for x in (q, k))   # the layer expands before
    o, st = vendor(tq, tk, *(torch.from_numpy(x) for x in (v, g, beta)), initial_state=torch.from_numpy(s),
                   output_final_state=True, use_qk_l2norm_in_kernel=True)
    dev_o, dev_s = O.oracle_deviation(o.numpy(), ref_o), O.oracle_deviation(st.numpy(), ref_s)
    print(f"{fn} L={L}: out {dev_o:.2e} state {dev_s:.2e}")
    assert dev_o <= VENDOR_BOUND[fn] and dev_s <= VENDOR_BOUND[fn]
