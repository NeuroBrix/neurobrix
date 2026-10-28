"""The expert store's residency on the card (`moe_device_residency`, S3 2026-10-28) is an LRU of the store's slots: over
a run of tokens across layers, every routed expert lands in a slot holding exactly its bytes, a resident expert is not
copied again, a miss takes a free slot first and then the least recently used slot the token does not read, and the
slot map always names what each slot holds. Checked against a Python model of that LRU. Needs a device."""
import numpy as np
import pytest


def _has_gpu():
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="needs a device")
def test_slots_follow_the_lru_and_hold_the_routed_experts():
    import ctypes
    import triton
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from nbxtensor.ops.moe_device_residency import expert_slot_assign_kernel, expert_miss_copy_kernel
    rng = np.random.default_rng(3)
    L, E, S, K, n = 3, 16, 10, 4, 1000                     # n elements per projection
    dev = "cuda:0"
    def pinned(a):
        t = NBXTensor.empty_cpu((n,), NBXDtype.float16, pinned=True)
        np.frombuffer((ctypes.c_uint16 * n).from_address(t.data_ptr()), dtype=np.float16)[:] = a
        return t
    host = [[[pinned(rng.standard_normal(n).astype(np.float16)) for _p in range(3)] for _e in range(E)]
            for _l in range(L)]
    slots = [[NBXTensor.empty((n,), dtype=NBXDtype.float16, device=dev) for _p in range(3)] for _s in range(S)]
    i64 = lambda a: NBXTensor.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)   # noqa: E731
    host_tabs = [i64([host[l][e][p].data_ptr() for l in range(L) for e in range(E)]) for p in range(3)]
    slot_tabs = [i64([slots[s][p].data_ptr() for s in range(S)]) for p in range(3)]
    slot_of = NBXTensor.from_numpy(np.full(L * E, -1, dtype=np.int32)).to(dev)
    owner = NBXTensor.from_numpy(np.full(S, -1, dtype=np.int32)).to(dev)
    stamp = i64(np.full(S, -1))
    clock = i64([0])
    model_owner, model_stamp, clk = [-1] * S, [-1] * S, 0
    for step in range(40):
        layer = step % L
        ids = rng.choice(E, size=K, replace=False)
        so, mo = NBXTensor.empty((K,), dtype=NBXDtype.int64, device=dev), NBXTensor.empty((K,), dtype=NBXDtype.int64,
                                                                                           device=dev)
        expert_slot_assign_kernel[(1,)](i64(ids), slot_of, owner, stamp, clock, so, mo, layer * E, S, TOP_K=K,
                                        BLOCK_S=triton.next_power_of_2(S))
        expert_miss_copy_kernel[(K, 3, triton.cdiv(n, 256))](mo, so, *host_tabs, *slot_tabs, layer * E, n, n, n,
                                                           BLOCK=256, W_BF16=False)
        # the model: hits stamped, then each miss in order takes the oldest stamp (free = -1 first, lowest index)
        want_slot, want_miss = [None] * K, [-1] * K
        for k, e in enumerate(ids):
            key = layer * E + int(e)
            if key in model_owner:
                s = model_owner.index(key)
                model_stamp[s] = clk
                want_slot[k] = s
        for k, e in enumerate(ids):
            if want_slot[k] is None:
                s = int(np.argmin(model_stamp))
                model_owner[s], model_stamp[s] = layer * E + int(e), clk
                want_slot[k], want_miss[k] = s, int(e)
        clk += 1
        assert so.numpy().tolist() == want_slot and mo.numpy().tolist() == want_miss, (step, ids)
        for k, e in enumerate(ids):
            for p in range(3):
                h = np.frombuffer((ctypes.c_uint16 * n).from_address(host[layer][int(e)][p].data_ptr()), dtype=np.float16)
                assert np.array_equal(slots[want_slot[k]][p].numpy(), h), (step, k, p)
    assert owner.numpy().tolist() == model_owner
    m = slot_of.numpy()
    assert all((m[o] == s) for s, o in enumerate(model_owner) if o >= 0) and int((m >= 0).sum()) == sum(
        1 for o in model_owner if o >= 0)
