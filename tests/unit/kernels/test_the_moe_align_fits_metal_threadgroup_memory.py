"""The device MoE align runs for a model with 128 experts on Metal, and sorts as the host sort does.

Qwen3-Coder-30B-A3B-Instruct, Triton certified-only on the Mac (VALIDATE 49, 2026-10-04 20:06, tree 1e94a1dc):
`moe_align_block_size` stopped at stage 1 with "out of resource: MSL threadgroup memory, Required: 65536,
Hardware limit: 32768". Stage 1 is one program holding [BE, BT] and [BE, BE] int32 matrices; with 128 experts
(BE 128, BT 128) each is 64 KB, twice Metal's 32 KB. Granite's 40 experts (BE 64) fit, so only stage 3 had been
narrowed for Metal. Before the fix the 128-expert cases fail on Metal; a chunk wider than the expert block
(8 experts: BF 128 > BE 16, a [128, 128] block) failed the same way. Every case checks the host sort's order.
"""
import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor


def _host_align(ids, bs, E):
    """The removed host implementation: tokens grouped by expert in token order, each group padded to bs with
    the sentinel n; one expert id per block, -1 past the true total."""
    n = ids.size
    counts = np.bincount(ids, minlength=E)
    padded = (counts + bs - 1) // bs * bs
    total = int(padded.sum())
    sorted_ids, expert_ids = [], []
    for e in range(E):
        own = np.flatnonzero(ids == e)
        sorted_ids += list(own) + [n] * int(padded[e] - own.size)
        expert_ids += [e] * int(padded[e] // bs)
    return np.array(sorted_ids, dtype=np.int64), np.array(expert_ids, dtype=np.int64), total


@pytest.mark.parametrize("E,top_k,tokens,bs", [(8, 2, 5, 16), (40, 8, 37, 16), (128, 8, 37, 16), (128, 8, 300, 64), (128, 2, 5, 16)])
def test_the_device_align_sorts_as_the_host_sort(E, top_k, tokens, bs):
    from neurobrix.triton.moe import moe_align_block_size
    ids = np.random.default_rng(E + tokens).integers(0, E, size=tokens * top_k).astype(np.int64)
    s, x, t = moe_align_block_size(NBXTensor.from_numpy(ids), bs, E, 0)
    want_s, want_x, want_t = _host_align(ids, bs, E)
    got_t = int(t.numpy()[0])
    assert got_t == want_t
    np.testing.assert_array_equal(s.numpy()[:want_t], want_s)
    nb = want_t // bs
    np.testing.assert_array_equal(x.numpy()[:nb], want_x)
    assert (x.numpy()[nb:] == -1).all()
