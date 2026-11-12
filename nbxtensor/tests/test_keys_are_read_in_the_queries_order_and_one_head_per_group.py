"""`memory.paged_kv.keys_seq_major` and `one_head_per_group`: the keys an attention reads put in the queries'
[B, H, S, D] order (the graph's recorded layout first, the shape only without one, the square case refused), and keys
or values whose heads were repeated for grouped-query attention read back as their own heads. The engine's KV
interceptor called its own copies (triton/kv_cache.py) until 2026-11-12.

What this test would do if the code were wrong: a recorded layout overridden by the shape, a square K guessed, a K
already in order transposed, a group read at another head than its first or a head of another group — RED."""
import numpy as np
import pytest

from nbxtensor.memory.paged_kv import keys_seq_major, one_head_per_group
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _t(shape):
    return NBXTensor.from_numpy(np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape))


def test_the_recorded_layout_decides_and_the_shape_only_without_one():
    _accelerator()
    q, kt = _t((1, 2, 5, 3)), _t((1, 2, 3, 5))                      # K stored [B, H, D, S]
    out = keys_seq_major(q, kt)
    assert out.shape == (1, 2, 5, 3) and np.array_equal(out.numpy(), kt.numpy().transpose(0, 1, 3, 2))
    assert keys_seq_major(q, kt, False) is kt                       # recorded "not transposed": left as it is
    k = _t((1, 2, 5, 3))
    assert keys_seq_major(q, k) is k                                # already in order
    assert np.array_equal(keys_seq_major(q, k, True).numpy(), k.numpy().transpose(0, 1, 3, 2))
    flat = _t((2, 5, 3))
    assert keys_seq_major(q, flat) is flat


def test_a_square_key_with_no_record_is_refused():
    _accelerator()
    q, k = _t((1, 2, 4, 4)), _t((1, 2, 4, 4))
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        keys_seq_major(q, k)
    assert keys_seq_major(q, k, False) is k


def test_repeated_heads_are_read_back_one_per_group():
    _accelerator()
    kv = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)          # 2 KV heads
    expanded = NBXTensor.from_numpy(np.repeat(kv, 3, axis=1))                    # 6 query heads, group 3
    heads = one_head_per_group(expanded, 2, 3)
    assert heads.shape == (1, 2, 3, 4) and np.array_equal(heads.numpy(), kv)
    assert heads.storage_root() is expanded                                      # a view, no copy
