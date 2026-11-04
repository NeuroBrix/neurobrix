"""The fixed-order reduce after both split decodes (`flash_decode_reduce_kernel`, through
`attention_wrappers.decode_reduce`) runs its certified configuration: the key it forms at the launch
(`autotune_cache.key_of`, the launcher's own) IS the key `launch_keys` computes from shapes and dtypes — the
second launch of `decode_vec_launches` and `paged_decode_launches`, the functions the derived census calls; and the
certifier's synthesis forms that same key and matches its fp64 oracle. Its warps were a literal 4 before 2026-11-04.

What this test would do if the code were wrong: the census keyed on another split rule than the wrapper's (the bucket
instead of the splits, the paged rule for the vector decode), the reduce's output dtype taken from Q where the
wrapper writes the cache's, or the reduce left out of the launch lists — red at the key; a synthesis that forms
another key, or an oracle that is not the log-sum-exp combine — red at the certification. Needs a device."""
from __future__ import annotations

import numpy as np
import pytest


def _card():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.device_count() >= 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _card(), reason="needs a device")


def _dtypes():
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    return NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32


class _Launched(Exception):
    """The reduce reached: its key is read, the sweep never runs."""


def _capture(monkeypatch):
    from nbxtensor.ops.autotune_cache import key_of
    from nbxtensor.ops.flash_decode import flash_decode_reduce_kernel as K
    seen = []

    def run(*args, **kwargs):
        seen.append(tuple(key_of(K, args, kwargs)))
        raise _Launched
    monkeypatch.setattr(K, "run", run)
    return seen


def _t(shape, dt):
    from nbxtensor.tensor import NBXTensor
    x = NBXTensor.from_numpy((np.random.default_rng(0).standard_normal(shape) * 0.1).astype(np.float32)).to("cuda")
    return x if dt == _dtypes()[2] else x.to(dt)


@pytest.mark.parametrize("Tk", [1, 255, 257, 4000, 70000])
@pytest.mark.parametrize("q_name,kv_name,rounded", [("f16", "f16", False), ("f32", "f16", True), ("bf16", "bf16", False)])
def test_the_vector_decode_reduce_is_its_key(monkeypatch, Tk, q_name, kv_name, rounded):
    from nbxtensor.ops import attention_wrappers as AW
    from nbxtensor.ops import launch_keys as LK
    f16, bf16, f32 = _dtypes()
    q_dt, kv_dt = ({"f16": f16, "bf16": bf16, "f32": f32}[n] for n in (q_name, kv_name))
    H, Hk, D = 8, 2, 64
    seen = _capture(monkeypatch)
    q, k, v = _t((1, H, 1, D), q_dt), _t((1, Hk, Tk, D), kv_dt), _t((1, Hk, Tk, D), kv_dt)
    with pytest.raises(_Launched):
        AW._decode_attn_vec(q, k, v, None, 0.125, 1, H, Hk, Tk, D, q_round=kv_dt if rounded else None)
    want = LK.decode_vec_launches(H, Hk, Tk, D, D, q_dt, kv_dt, kv_dt, None, q_round=kv_dt if rounded else None)
    assert want[1][0] == LK.DECODE_REDUCE and seen == [want[1][1]], (seen, want)


@pytest.mark.parametrize("Tk", [1, 256, 257, 5000])
@pytest.mark.parametrize("q_name", ["f16", "f32"])
def test_the_paged_decode_reduce_is_its_key(monkeypatch, Tk, q_name):
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops import attention_wrappers as AW
    from nbxtensor.ops import launch_keys as LK
    f16, _bf16, f32 = _dtypes()
    q_dt = {"f16": f16, "f32": f32}[q_name]
    H, Hk, D, pt = 8, 2, 64, 64
    pages = -(-Tk // pt)
    seen = _capture(monkeypatch)
    q, kp, vp = _t((1, H, 1, D), q_dt), _t((pages, Hk, pt, D), f16), _t((pages, Hk, pt, D), f16)
    bt = NBXTensor.from_numpy(np.arange(pages, dtype=np.int32).reshape(1, pages)).to("cuda")
    lens = NBXTensor.from_numpy(np.array([Tk], dtype=np.int32)).to("cuda")
    with pytest.raises(_Launched):
        AW.paged_decode_attention(q, kp, vp, bt, lens, 0.125, Tk, q_round=f16)
    want = LK.paged_decode_launches(H, Hk, Tk, D, D, q_dt, f16, pt)
    assert want[1][0] == LK.DECODE_REDUCE and seen == [want[1][1]], (seen, want)


@pytest.mark.parametrize("splits,dv,out", [(1, 64, "fp16"), (17, 128, "bf16"), (64, 80, "fp32")])
def test_the_certifier_certifies_the_reduce_key(splits, dv, out):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.ops.flash_decode import flash_decode_reduce_kernel as K
    from neurobrix.kernels.autotune_certify import _bind_hardware_profile, certify_key
    _bind_hardware_profile()
    f16, bf16, f32 = _dtypes()
    _q, key = LK.decode_reduce_launch(splits, dv, {"fp16": f16, "bf16": bf16, "fp32": f32}[out])
    entry = certify_key(LK.DECODE_REDUCE, K, key, 1e-2, np.random.default_rng(1),
                        bench=lambda fn: (fn(), 1.0)[1])
    assert entry["excluded"] == [] and entry["proof"]["accepted"] == entry["proof"]["candidates"] == 4, entry
    assert entry["proof"]["shape"] == list(key) and entry["proof"]["deviation"] <= 1e-2
