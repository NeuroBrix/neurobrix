"""The expert store keeps a component's routed experts on the host and serves the ones a call
routes to from device slots, in both engines, byte for byte the resident computation.

Three parts:
* the residency POLICY (`core/runtime/expert_residency.py`, no tensor library, no card): one LRU
  over (layer, expert), the call's experts pinned, a call wider than the slots refused by name;
* the triton engine (CARD REQUIRED — skipped without one): `execute_moe_fused` with the experts on
  pinned host memory and an `NBXExpertStore` of fewer slots than the call routes to (the grouped
  pass, a prefill) and of more (the direct pass, decode-like reuse), against the same call on
  resident experts — BYTE equality, since the kernels, launch constants and reduce order are the same;
* the compiled engine's store (CARD REQUIRED): every slot it hands out is the host expert's bytes,
  after evictions.

What these tests would do if the code were wrong: a slot handed to the wrong expert, a run cut at
a block that is not an expert boundary, a copy not ordered before the GEMM that reads it, or an
LRU that evicts a pinned expert changes the output bytes (red) — the injection test below hands
the triton store's experts one slot off and asserts the comparison sees it.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from neurobrix.core.runtime.expert_residency import ExpertResidency, routed_layers


# --------------------------------------------------------------------------- the policy (no card)

def test_hits_misses_and_least_recent_eviction():
    r = ExpertResidency(3)
    slots, missed = r.assign("L0", [2, 0])
    assert missed == [0, 2] and slots == {0: 0, 2: 1}
    slots, missed = r.assign("L0", [2])
    assert missed == [] and slots == {2: 1} and (r.hits, r.misses) == (1, 2)
    slots, missed = r.assign("L1", [5])                    # the free slot
    assert missed == [5] and slots == {5: 2}
    slots, missed = r.assign("L1", [7])                    # evicts (L0, 0), the least recent
    assert missed == [7] and slots == {7: 0} and r.slot_of("L0", 0) is None
    assert r.slot_of("L0", 2) == 1 and r.slot_of("L1", 5) == 2


def test_a_calls_own_experts_never_evict_one_another():
    r = ExpertResidency(2)
    r.assign("L", [0, 1])
    slots, missed = r.assign("L", [1, 3])                  # 0 is the only evictable key
    assert missed == [3] and slots == {1: 1, 3: 0}
    assert len(set(slots.values())) == len(slots)


def test_a_call_wider_than_the_slots_is_refused_by_name():
    r = ExpertResidency(2)
    with pytest.raises(RuntimeError, match="routes to 3 distinct experts"):
        r.assign("L", [0, 1, 2])
    with pytest.raises(RuntimeError, match="at least one slot"):
        ExpertResidency(0)


def _fused(uid, ids, stacked=False):
    a = {"num_experts": len(ids), "top_k": 2,
         "expert_gate_weight_ids": [f"{i}.g" for i in ids], "expert_up_weight_ids": [f"{i}.u" for i in ids],
         "expert_down_weight_ids": [f"{i}.d" for i in ids]}
    if stacked:
        a["stacked_experts"] = {"input_linear_tid": "in"}
    return {"op_type": "custom::moe_fused", "attributes": a}


def test_the_layers_are_keyed_by_their_first_gate_weight_and_a_slab_is_refused():
    dag = {"ops": {"a": _fused("a", ["l0e0", "l0e1"]), "b": {"op_type": "aten::mm"},
                   "c": _fused("c", ["l1e0", "l1e1"])}, "execution_order": ["a", "b", "c"]}
    layers = routed_layers(dag)
    assert list(layers) == ["l0e0.g", "l1e0.g"] and layers["l1e0.g"][2] == ["l1e0.d", "l1e1.d"]
    with pytest.raises(RuntimeError, match="stacked slab"):
        routed_layers({"ops": {"s": _fused("s", ["x"], stacked=True)}})


# --------------------------------------------------------------------------- triton (card)

E, TOP_K, H, F = 8, 2, 64, 32


def _card():
    pytest.importorskip("triton")
    from neurobrix.kernels.nbx_tensor import NBXTensor
    try:
        NBXTensor.from_numpy(np.zeros(1, np.float16))
    except Exception as exc:                                # no card: the op-level probe is skipped
        pytest.skip(f"no card: {exc}")


def _host(arr):
    from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor
    t = NBXTensor.empty_cpu(arr.shape, NBXDtype.float16)
    ctypes.memmove(t.data_ptr(), np.ascontiguousarray(arr).ctypes.data, arr.nbytes)
    return t


def _problem(tokens, seed):
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((tokens, H)) * 0.5).astype(np.float16)
    logits = rng.standard_normal((tokens, E)).astype(np.float32)
    # nn.Linear layout [out, in], as the per-expert matrices are stored
    g = [(rng.standard_normal((F, H)) * 0.1).astype(np.float16) for _ in range(E)]
    u = [(rng.standard_normal((F, H)) * 0.1).astype(np.float16) for _ in range(E)]
    d = [(rng.standard_normal((H, F)) * 0.1).astype(np.float16) for _ in range(E)]
    return h, logits, g, u, d


def _run(h, logits, g, u, d, store=None, key=""):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.triton.moe import execute_moe_fused
    mk = _host if store is not None else NBXTensor.from_numpy
    out = execute_moe_fused(NBXTensor.from_numpy(logits), NBXTensor.from_numpy(h),
                            [mk(x) for x in g], [mk(x) for x in u], [mk(x) for x in d],
                            top_k=TOP_K, num_experts=E, norm_topk_prob=False, cache_key=key,
                            expert_store=store, expert_layer="L" if store is not None else None)
    return np.asarray(out.numpy())


def _store(n_slots, d0):
    from neurobrix.triton.expert_store import NBXExpertStore
    like = (_host(np.zeros((F, H), np.float16)), _host(np.zeros((F, H), np.float16)), _host(d0))
    return NBXExpertStore(ExpertResidency(n_slots), 0, like)


@pytest.mark.parametrize("n_slots", [2, 3, 5, 8])
def test_the_triton_store_is_byte_for_byte_the_resident_moe(n_slots):
    _card()
    h, logits, g, u, d = _problem(tokens=24, seed=11)
    ref = _run(h, logits, g, u, d, key=f"resident-{n_slots}")
    store = _store(n_slots, np.zeros((H, F), np.float16))
    try:
        got = _run(h, logits, g, u, d, store=store, key="store")            # 24 tokens: every expert routed
        again = _run(h, logits, g, u, d, store=store, key="store")          # the second call reuses slots
        h1, l1 = h[:1], logits[:1]
        one = _run(h1, l1, g, u, d, store=store, key="store")               # a decode token
        one_ref = _run(h1, l1, g, u, d, key=f"resident1-{n_slots}")
    finally:
        store.release()
    assert np.array_equal(got.view(np.uint16), ref.view(np.uint16)), f"n_slots={n_slots}: bytes differ"
    assert np.array_equal(again.view(np.uint16), ref.view(np.uint16))
    assert np.array_equal(one.view(np.uint16), one_ref.view(np.uint16))
    assert store.residency.misses > 0


def test_the_comparison_sees_an_expert_served_from_the_wrong_slot(monkeypatch):
    """The injection: the store hands every expert its neighbour's slot. Seen red, or the byte
    equality above proves nothing."""
    _card()
    from neurobrix.triton.expert_store import NBXExpertStore
    h, logits, g, u, d = _problem(tokens=24, seed=11)
    ref = _run(h, logits, g, u, d, key="resident-inject")
    real = NBXExpertStore.acquire

    def shifted(self, layer, routed, n_experts, host):
        gl, ul, dl = real(self, layer, routed, n_experts, host)
        rot = lambda xs: xs[1:] + xs[:1]
        return rot(gl), rot(ul), rot(dl)
    monkeypatch.setattr(NBXExpertStore, "acquire", shifted)
    store = _store(3, np.zeros((H, F), np.float16))
    try:
        got = _run(h, logits, g, u, d, store=store, key="store-inject")
    finally:
        store.release()
    assert not np.array_equal(got.view(np.uint16), ref.view(np.uint16)), "the injection went unseen"


def test_the_triton_store_refuses_a_foreign_geometry():
    _card()
    h, logits, g, u, d = _problem(tokens=4, seed=2)
    store = _store(4, np.zeros((H, F + 1), np.float16))
    try:
        with pytest.raises(RuntimeError, match="one geometry per component"):
            _run(h, logits, g, u, d, store=store, key="store-geometry")
    finally:
        store.release()


# --------------------------------------------------------------------------- compiled (card)

def test_the_compiled_store_hands_out_the_host_bytes_after_evictions():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no card")
    from neurobrix.core.runtime.graph.expert_store import TorchExpertStore
    gen = torch.Generator().manual_seed(5)
    host = {e: tuple(torch.randn(s, generator=gen).half().pin_memory() for s in ((F, H), (F, H), (H, F)))
            for e in range(E)}
    store = TorchExpertStore(ExpertResidency(3), torch.device("cuda:0"), host[0])
    try:
        for routed in ([0, 1, 2], [2, 5], [6, 7, 1], [0, 3], [5, 6, 7]):
            on_card, events = store.acquire("L", routed, host.__getitem__)
            assert sorted(on_card) == sorted(set(routed))
            for e, trip in on_card.items():
                if e in events:
                    torch.cuda.current_stream().wait_event(events[e])
                for dev_t, host_t in zip(trip, host[e]):
                    assert torch.equal(dev_t.cpu(), host_t), f"expert {e} of {routed}: wrong bytes"
    finally:
        store.release()
    assert store.residency.hits > 0 and store.residency.misses > 0
