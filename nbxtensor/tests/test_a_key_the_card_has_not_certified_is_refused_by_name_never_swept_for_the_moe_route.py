"""On a card, in a confirmation run (NBX_AUTOTUNE_CERTIFIED_ONLY=1), `moe_route` at a key the certified directory does
not serve is REFUSED BY NAME (`KeyNotCertified`, naming the kernel and the key) and never sweeps: the kernel's tuner
cache holds no new entry afterwards and no candidate ran. The refusal is the library's own door for every autotuned
kernel (`_configs.run_with_notice` -> `autotune_certified.refuse_missing`); this test proves the routing kernel goes
through it. The key used (E 72, H 88, k 3) is one no certification covers: the card job certifies the keys of the
device test, not this one.

A card test: with no device it skips by name."""
import math

import numpy as np
import pytest

from _moe_route_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype


def test_an_uncertified_key_is_refused_by_name_and_never_swept(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")
    from nbxtensor.ops import moe_route as R
    from nbxtensor.ops.autotune_certified import KeyNotCertified
    from nbxtensor.tensor import NBXTensor
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    x, w, b = draw(1, 72, 88, 5, 0.0)
    dev = lambda a: NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device="cuda:0")
    cache = R.moe_route_kernel.cache
    before = len(cache)
    with pytest.raises(KeyNotCertified, match=r"CERTIFIED-ONLY: no certified setting for moe_route_kernel.*never sweeps"):
        R.moe_route(dev(x), dev(w), choice_bias=dev(b), score_fn="sigmoid", top_k=3, n_group=1, topk_group=1,
                    group_score="top2_sum", fill=-math.inf, renorm_eps=1e-20, scale=1.0)
    assert len(cache) == before, "the refused launch left an entry in the tuner's cache: it swept"
