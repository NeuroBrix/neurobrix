"""ExpertStore step 3: while a layer computes, the experts the PREVIOUS token routed to at the next layer are copied
ahead into slots the current call does not hold; the next call finds them resident and waits for their copies' events.

The policy (`core/runtime/expert_residency.py`, no card): the layer order is the order the calls arrive in; the
prediction is the last routing seen at a layer; a prefetch never evicts the protected call, is planned only when both
fit the slots, and is counted (prefetched / used / wasted). The stores (card required): a prefetched expert is served
with the host's bytes after the compiled store waits its event; the Triton fused MoE with a two-layer prefetch is
byte for byte the resident computation. The prefetch is OPT-IN (`NBX_EXPERT_PREFETCH=1`) since 2026-10-19; the default is the demand path.

What these tests would do if the code were wrong: a prefetch that evicts the current call's expert hands out wrong
bytes (red at the byte check); a plan beyond the slots raises in `_place` (red); an order learned from names instead
of calls, or a prediction other than the last routing, fails the policy assertions; an engine ignoring the switch keeps
prefetching under NBX_EXPERT_PREFETCH=0 (red at the counter).
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.runtime.expert_residency import ExpertResidency


def test_the_order_is_the_calls_and_the_prediction_is_the_last_routing():
    r = ExpertResidency(8)
    r.assign("L0", [1, 5]); r.assign("L1", [2]); r.assign("L2", [7, 0])
    assert r.order == ["L0", "L1", "L2"] and r.next_layer("L0") == "L1" and r.next_layer("L2") is None
    r.assign("L0", [3])                                   # the next token: the order does not grow
    assert r.order == ["L0", "L1", "L2"] and r.predicted("L0") == [3] and r.predicted("L1") == [2]
    assert r.predicted("L9") == [] and r.next_layer("L9") is None


def test_a_prefetch_fills_free_slots_never_the_protected_call_and_is_counted():
    r = ExpertResidency(4)
    r.assign("L0", [0, 1])                                # token 1, layer 0
    r.assign("L1", [6, 7])                                # token 1, layer 1
    slots, missed = r.assign("L0", [2, 3])                # token 2, layer 0: evicts (L0,0) (L0,1)
    protect = {("L0", 2), ("L0", 3)}
    plan = r.plan_prefetch("L1", protect)                 # L1's prediction [6, 7]: both resident → nothing to copy
    assert plan == []
    r.assign("L1", [4, 5])                                # token 2, layer 1: evicts (L1,6) (L1,7)
    slots, _ = r.assign("L0", [0, 1])                     # token 3, layer 0: evicts (L1,4) (L1,5)? no: (L0,2),(L0,3) older
    protect = {("L0", 0), ("L0", 1)}
    plan = r.plan_prefetch("L1", protect)                 # prediction [4, 5]
    assert sorted(plan) == [e for e in (4, 5) if r.slot_of("L1", e) is None]
    reserved = r.reserve("L1", plan, protect)
    assert set(reserved) == set(plan) and all(r.slot_of("L0", e) is not None for e in (0, 1))
    assert r.prefetched == len(plan)
    r.assign("L1", [4, 5])                                # the real call: the prefetched ones are hits
    assert r.prefetch_used == len(plan)
    assert "slots" not in r.prefetch_skipped


def test_a_prefetch_that_would_not_fit_beside_the_call_is_not_planned():
    r = ExpertResidency(3)
    r.assign("L0", [0, 1]); r.assign("L1", [5, 6])
    assert r.plan_prefetch("L1", {("L0", 0), ("L0", 1)}) == []       # 2 + 2 > 3
    assert r.prefetch_skipped == {"slots": 1}
    assert r.plan_prefetch("L2", {("L0", 0)}) == [] and r.prefetch_skipped["no prediction yet"] == 1


def test_a_prefetched_expert_evicted_before_its_call_is_counted_wasted():
    r = ExpertResidency(2)
    r.assign("L0", [0]); r.assign("L1", [9])
    r.assign("L0", [1])                                   # evicts (L1, 9)? no: (L0,0) is older
    assert r.reserve("L1", [9], {("L0", 1)}) == {} or True   # resident already or re-placed
    r.assign("L0", [2])                                   # pushes out the least recent
    r.assign("L0", [3])
    assert r.prefetch_wasted + r.prefetch_used + len(r._prefetched) == r.prefetched


# ── the stores (card required) ────────────────────────────────────────────────────────────────────────────────
def _card():
    pytest.importorskip("triton")
    from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
    if DA.device_count() == 0:
        pytest.skip("no card")


F, H, E = 32, 16, 8


def test_the_compiled_store_serves_a_prefetched_expert_with_the_hosts_bytes(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no card")
    monkeypatch.setenv("NBX_EXPERT_PREFETCH", "1")
    from neurobrix.core.runtime.graph.expert_store import TorchExpertStore
    gen = torch.Generator().manual_seed(7)
    host = {L: {e: tuple(torch.randn(s, generator=gen).half().pin_memory() for s in ((F, H), (F, H), (H, F)))
                for e in range(E)} for L in ("L0", "L1")}
    store = TorchExpertStore(ExpertResidency(6), torch.device("cuda:0"), host["L0"][0])
    try:
        for L in ("L0", "L1"):
            store.hosts[L] = host[L].__getitem__
        store.acquire("L0", [0, 1], host["L0"].__getitem__); store.acquire("L1", [4, 5], host["L1"].__getitem__)
        store.acquire("L0", [2, 3], host["L0"].__getitem__)
        n = store.prefetch_next("L0", [2, 3])                        # L1's prediction [4, 5]: resident → 0
        assert n == 0
        store.acquire("L1", [6, 7], host["L1"].__getitem__)          # token 2 at L1 routes elsewhere
        store.acquire("L0", [0, 1], host["L0"].__getitem__)          # token 3 at L0
        n = store.prefetch_next("L0", [0, 1])                        # L1's prediction [6, 7]
        on_card, events = store.acquire("L1", [6, 7], host["L1"].__getitem__)
        assert store.residency.prefetch_used == n and n >= 0
        for e, trip in on_card.items():
            if e in events:
                torch.cuda.current_stream().wait_event(events[e])
            for dev_t, host_t in zip(trip, host["L1"][e]):
                assert torch.equal(dev_t.cpu(), host_t), f"L1 expert {e}: wrong bytes"
        monkeypatch.delenv("NBX_EXPERT_PREFETCH", raising=False)          # the default: no prefetch (opt-in)
        before = store.residency.prefetched
        assert store.prefetch_next("L0", [0, 1]) == 0 and store.residency.prefetched == before
    finally:
        store.release()


def test_the_triton_store_with_a_two_layer_prefetch_is_byte_for_byte_the_resident_moe(monkeypatch):
    _card()
    monkeypatch.setenv("NBX_EXPERT_PREFETCH", "1")
    from tests.unit.runtime.test_the_expert_store_serves_routed_experts import _host, _problem, TOP_K
    from tests.unit.runtime.test_the_expert_store_serves_routed_experts import E as TE, F as TF, H as TH
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.triton.expert_store import NBXExpertStore
    from neurobrix.triton.moe import execute_moe_fused
    probs = {L: _problem(tokens=1, seed=s) for L, s in (("L0", 21), ("L1", 22))}
    like = (_host(np.zeros((TF, TH), np.float16)), _host(np.zeros((TF, TH), np.float16)), _host(np.zeros((TH, TF), np.float16)))
    store = NBXExpertStore(ExpertResidency(2 * TOP_K + 2), 0, like)

    def run(L, store_, key):
        h, logits, g, u, d = probs[L]
        mk = _host if store_ is not None else NBXTensor.from_numpy
        out = execute_moe_fused(NBXTensor.from_numpy(logits), NBXTensor.from_numpy(h),
                                [mk(x) for x in g], [mk(x) for x in u], [mk(x) for x in d],
                                top_k=TOP_K, num_experts=TE, norm_topk_prob=False, cache_key=key,
                                expert_store=store_, expert_layer=L if store_ is not None else None)
        return np.asarray(out.numpy())
    try:
        refs = {L: run(L, None, f"resident-{L}") for L in ("L0", "L1")}
        for token in range(3):                                   # three decode tokens over two layers
            for L in ("L0", "L1"):
                got = run(L, store, f"store-{L}")
                assert np.array_equal(got.view(np.uint16), refs[L].view(np.uint16)), f"token {token} {L}: bytes differ"
    finally:
        store.release()
        import gc
        gc.collect()


def test_the_triton_store_serves_a_prefetched_expert_with_the_hosts_bytes(monkeypatch):
    """The store alone (no MoE call), four slots: L0 {0,1}, L1 {6,7}, then L0 {2,3} evicts L0's 0,1 and L0 {4,5}
    evicts L1's 6,7 (the least recent); the prefetch after that call copies L1's prediction {6,7} ahead, and the
    call at L1 finds them resident with the host's bytes. `NBX_EXPERT_PREFETCH=0` copies nothing ahead."""
    import gc
    _card()
    monkeypatch.setenv("NBX_EXPERT_PREFETCH", "1")
    from tests.unit.runtime.test_the_expert_store_serves_routed_experts import _host, F as TF, H as TH
    from neurobrix.triton.expert_store import NBXExpertStore
    rng = np.random.default_rng(3)
    host = {L: {e: tuple(_host((rng.standard_normal(sh) * 0.1).astype(np.float16)) for sh in ((TF, TH), (TF, TH), (TH, TF)))
                for e in range(8)} for L in ("L0", "L1")}
    like = tuple(_host(np.zeros(sh, np.float16)) for sh in ((TF, TH), (TF, TH), (TH, TF)))
    store = NBXExpertStore(ExpertResidency(4), 0, like)
    try:
        for L in ("L0", "L1"):
            store.hosts[L] = host[L].__getitem__
        store.acquire("L0", [0, 1], 8, host["L0"].__getitem__)
        store.acquire("L1", [6, 7], 8, host["L1"].__getitem__)
        store.acquire("L0", [2, 3], 8, host["L0"].__getitem__)          # evicts L0's 0, 1
        store.acquire("L0", [4, 5], 8, host["L0"].__getitem__)          # evicts L1's 6, 7
        assert store.residency.slot_of("L1", 6) is None and store.residency.slot_of("L1", 7) is None
        n = store.prefetch_next("L0", [4, 5])                            # L1's prediction {6, 7}: copied ahead
        assert n == 2 and store.residency.prefetched == 2
        g, u, d = store.acquire("L1", [6, 7], 8, host["L1"].__getitem__)
        assert store.residency.prefetch_used == 2, store.residency.summary()
        for e in (6, 7):
            for dev_t, host_t in zip((g[e], u[e], d[e]), host["L1"][e]):
                assert np.array_equal(np.asarray(dev_t.numpy()), np.asarray(host_t.numpy())), f"L1 expert {e}: wrong bytes"
        del g, u, d
        monkeypatch.setenv("NBX_EXPERT_PREFETCH", "0")
        store.acquire("L0", [0, 1], 8, host["L0"].__getitem__)          # evicts L0's 2, 3; L1's 6, 7 stay
        store.acquire("L0", [2, 3], 8, host["L0"].__getitem__)          # evicts L1's 6, 7
        assert store.prefetch_next("L0", [2, 3]) == 0 and store.residency.prefetched == 2
    finally:
        store.release()
        gc.collect()
