"""One decode token through a routed layer whose experts are stored in half precision runs the SIMT decode band
(`moe_decode_vec` half kernels, S3 2026-10-28), and its output is the routed SwiGLU sum computed in fp64 by numpy.
Before: the grouped GEMM ran it (one real row per 16-row tile, ~70 GB/s on V100 for Qwen3-30B-A3B's experts). Needs a
device."""
import numpy as np
import pytest


def _has_gpu():
    from nbxtensor.tensor import DeviceAllocator
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _oracle(x, gates, ups, downs, ids, wts):
    x = x.astype(np.float64)
    out = np.zeros(downs[0].shape[0], dtype=np.float64)
    for e, w in zip(ids, wts):
        g = gates[e].astype(np.float64) @ x
        u = ups[e].astype(np.float64) @ x
        h = g / (1.0 + np.exp(-g)) * u
        out += float(w) * (downs[e].astype(np.float64) @ h)
    return out


@pytest.mark.skipif(not _has_gpu(), reason="needs a device")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_the_half_band_is_the_routed_swiglu_sum(dtype, monkeypatch):
    from nbxtensor.tensor import NBXTensor, NBXDtype
    from neurobrix.triton import moe
    nbx = {"float16": NBXDtype.float16, "bfloat16": NBXDtype.bfloat16}[dtype]
    rng = np.random.default_rng(7)
    E, K, N, top_k = 6, 512, 192, 4
    gates = [rng.standard_normal((N, K)).astype(np.float32) * 0.05 for _ in range(E)]
    ups = [rng.standard_normal((N, K)).astype(np.float32) * 0.05 for _ in range(E)]
    downs = [rng.standard_normal((K, N)).astype(np.float32) * 0.05 for _ in range(E)]
    x = rng.standard_normal((1, K)).astype(np.float32)
    ids = np.array([[4, 1, 5, 2]], dtype=np.int64)
    wts = np.array([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)

    def dev(a, dt):
        return NBXTensor.from_numpy(a).to(dt).to("cuda:0")
    half = lambda a: dev(a, nbx)                                        # noqa: E731
    rounded = lambda ws: [half(w).to(NBXDtype.float32).numpy() for w in ws]  # noqa: E731
    used = []
    real = moe._moe_decode_vec_half_pass

    def spy(*a, **k):
        used.append(True)
        return real(*a, **k)
    monkeypatch.setattr(moe, "_moe_decode_vec_half_pass", spy)
    out = moe.execute_moe_fused(
        None, half(x), [half(g) for g in gates], [half(u) for u in ups], [half(d) for d in downs],
        top_k=top_k, num_experts=E, cache_key=f"half_band_{dtype}",
        topk_indices=dev(ids, NBXDtype.int64), topk_weights=dev(wts, NBXDtype.float32))
    assert used, "the decode token did not take the half SIMT band"
    ref = _oracle(half(x).to(NBXDtype.float32).numpy()[0], rounded(gates), rounded(ups), rounded(downs),
                  ids[0], half(wts).to(NBXDtype.float32).numpy()[0])
    got = out.to(NBXDtype.float32).numpy().reshape(-1).astype(np.float64)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < (2e-3 if dtype == "float16" else 1.6e-2), err
