"""Every element-wise launch reads its tile and warps from ONE certified key: its element count's class
(`ops/elementwise_launch.ew_meta`, the shared copy's autotuner), the key the census derives for every class a
component can launch (`launch_keys.elementwise_classes`).

What this test would do if the code were wrong: a wrapper launching at a fixed tile never asks the shared copy —
red at `_Launched`; a census that misses the class of some count up to the largest tensor — red at the coverage
assertion; a class floor that does not form its class — red at the inverse assertion."""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.ops import launch_keys as LK
from nbxtensor.ops import autotune_bucket as AB

_LADDERS = {
    "measured": [{"up_to": 65536, "step": 65536}, {"up_to": 1048576, "step": 1048576},
                 {"up_to": 16777216, "step": 16777216}, {"up_to": None, "step": 68719476736}],
    "fine": [{"up_to": 64, "step": 1}, {"up_to": 256, "step": 16}, {"up_to": 1024, "step": 32},
             {"up_to": 8192, "step": 128}, {"up_to": None, "step": 512}],
}


@pytest.fixture(params=sorted(_LADDERS))
def ladder(request, monkeypatch):
    prof = {"architecture": "test", "autotune": {"buckets": {"elements": _LADDERS[request.param]}}}
    real = AB.ladder_for
    monkeypatch.setattr(AB, "ladder_for", lambda dim, profile=None: real(dim, prof))
    return request.param


@pytest.mark.parametrize("largest", [1, 63, 64, 65, 300, 65536, 65537, 2_000_000, 50_000_000])
def test_the_census_names_the_class_of_every_count_up_to_the_largest(ladder, largest):
    classes = {key[0] for _, key in LK.elementwise_classes(largest)}
    probes = sorted({1, largest, max(1, largest // 2), min(largest, largest // 3 + 1)}
                    | set(np.random.default_rng(largest).integers(1, largest + 1, 50).tolist()))
    for n in probes:
        assert AB.bucket_of("elements", n) in classes, (ladder, largest, n, sorted(classes)[:8])
    assert all(AB.class_floor("elements", top) <= largest for top in classes)


@pytest.mark.parametrize("n", [1, 7, 64, 65, 255, 4097, 65536, 65537, 1048577, 99_999_999])
def test_a_class_floor_forms_its_class_and_one_less_does_not(ladder, n):
    top = AB.bucket_of("elements", n)
    floor = AB.class_floor("elements", top)
    assert AB.bucket_of("elements", floor) == top and floor <= n
    assert floor == 1 or AB.bucket_of("elements", floor - 1) < top


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("n", [5, 70000, 3_000_000])
def test_a_wrapper_launch_asks_the_shared_copy_for_its_class(monkeypatch, n):
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import elementwise_wrappers as _nbx_elementwise_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops import elementwise_launch as E
    from nbxtensor.ops.autotune_cache import key_of

    class _Launched(Exception):
        pass
    seen = []

    def run(*args, **kwargs):
        seen.append(key_of(E.ew_launch_kernel, args, kwargs))
        raise _Launched
    monkeypatch.setattr(E.ew_launch_kernel, "run", run)
    monkeypatch.setattr(E, "_MEMO", {})
    x = NBXTensor.from_numpy(np.ones(n, dtype=np.float32)).to("cuda")
    with pytest.raises(_Launched):
        _nbx_elementwise_wrappers.silu(x)
    assert seen == [(E.ew_bucket(n), "fp32", "fp32")]
    assert seen[0] in [k for _, k in LK.elementwise_classes(n)]
