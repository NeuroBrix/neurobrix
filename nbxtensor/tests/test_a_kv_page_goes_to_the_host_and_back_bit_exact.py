"""A KV page offloaded to the host tier and restored comes back bit-exact, every layer's K and V, into the same pool
addresses; the other pages are untouched. On the machine's accelerator (Metal or CUDA), pageable and pinned host pages."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.kv_host_pages import HostPages, offload, restore
from nbxtensor.tensor import NBXTensor

LAYERS, PAGES, HEADS, TOKENS, DK, DV = 3, 6, 2, 16, 64, 32


def _accelerator():
    try:
        DeviceAllocator.set_device(0)
        NBXTensor.zeros((1,), dtype=NBXDtype.float16, device="cuda:0")
    except Exception as e:                                  # no accelerator: a card-less machine
        pytest.skip(f"no accelerator: {e}")


def _pools(rng, d):
    return [NBXTensor.from_numpy(rng.standard_normal((PAGES, HEADS, TOKENS, d)).astype(np.float16)) for _ in range(LAYERS)]


@pytest.mark.parametrize("pinned", [False, True])
def test_a_page_round_trips_bit_exact_into_the_same_addresses(pinned):
    _accelerator()
    rng = np.random.default_rng(7)
    k, v = _pools(rng, DK), _pools(rng, DV)
    before_k, before_v = [p.numpy().copy() for p in k], [p.numpy().copy() for p in v]
    addrs = [p.data_ptr() for p in (*k, *v)]
    host = HostPages(capacity=2, num_layers=LAYERS, k_page_bytes=HEADS * TOKENS * DK * 2,
                     v_page_bytes=HEADS * TOKENS * DV * 2, pinned=pinned)
    try:
        offload(k, v, [4, 1], host, [1, 0])
        for p in (*k, *v):                                   # the device pages overwritten (another conversation)
            p_np = np.zeros(p.shape, dtype=np.float16)
            DeviceAllocator.memcpy(p.data_ptr(), NBXTensor.from_numpy(p_np).data_ptr(), p_np.nbytes, 3)
        restore(k, v, host, [1, 0], [2, 5])                  # back into OTHER device pages
        DeviceAllocator.device_synchronize(0)
        for i in range(LAYERS):
            kn, vn = k[i].numpy(), v[i].numpy()
            assert np.array_equal(kn[2], before_k[i][4]) and np.array_equal(kn[5], before_k[i][1])
            assert np.array_equal(vn[2], before_v[i][4]) and np.array_equal(vn[5], before_v[i][1])
            assert not kn[[0, 1, 3, 4]].any()                # nothing written elsewhere
        assert [p.data_ptr() for p in (*k, *v)] == addrs     # the pools never moved
    finally:
        host.close()


def test_a_page_size_that_does_not_match_the_pool_is_refused():
    _accelerator()
    rng = np.random.default_rng(1)
    k, v = _pools(rng, DK), _pools(rng, DV)
    host = HostPages(1, LAYERS, HEADS * TOKENS * DK * 2, HEADS * TOKENS * DK * 2, pinned=False)   # V sized as K
    with pytest.raises(ValueError, match="v page"):
        offload(k, v, [0], host, [0])
