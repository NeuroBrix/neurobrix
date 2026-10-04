"""A bucketize over an empty boundary vector answers all zeros, as torch.bucketize does, and launches nothing.

Qwen3-VL-30B-A3B-Thinking, Triton certified-only on the Mac (VALIDATE 49, 2026-10-04 20:19, tree 1e94a1dc): the
vision tower buckets its positions over `cu_seqlens[1:-1]`, the inner frame boundaries. Traced on a 5-frame video
they are 4; the one-image request has none, a (0,) tensor whose address is 0, and the triton-ext driver refused the
null pointer ("Failed at aten.bucketize::0 ... handed a null pointer parameter"). Before the fix the empty cases
fail on Metal; every case checks numpy's searchsorted, which bucketize is.
"""
import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("nb", [0, 1, 4])
def test_bucketize_answers_as_searchsorted(nb, right):
    from neurobrix.kernels.wrappers import bucketize_wrapper
    x = np.arange(784, dtype=np.int64)
    b = np.sort(np.random.default_rng(nb).choice(784, size=nb, replace=False)).astype(np.int64)
    got = bucketize_wrapper(NBXTensor.from_numpy(x), NBXTensor.from_numpy(b), right=right).numpy()
    np.testing.assert_array_equal(got, np.searchsorted(b, x, side="right" if right else "left"))
