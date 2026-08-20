"""Unit test — sort_wrapper values AND indices (contaminated-oracle class).

`sort_wrapper` is the second of the four wrappers in `wrappers.py` that
return a (values, indices) pair, and until this file it had no test that
called it. The class is named in the top-k post-mortem: a byte gate
proves two runs AGREE, never that either is RIGHT, so a kernel wrong in
its indices while plausible in its values passes every gate the project
has. `topk_wrapper` shipped exactly that way in 0.5.1.

It is reached in production through the `aten::sort` dispatch
(`kernels/dispatch.py`), which is how Ming's windowed-ViT uses it — not
through the sampler, where the triton `TopPSampler` is defined but never
instantiated and `CombinedSampler` sorts on the host.

=== WHAT IS CHECKED, SEPARATELY ===

  values      the returned values are the input's values in order
  indices     the indices are a permutation of range(n), per row
  coherence   x[index] == value, elementwise

Index EQUALITY against numpy's argsort is NOT asserted: equal values may
legitimately be ordered either way by a non-stable sort, and asserting it
would manufacture failures out of correct behaviour. Coherence plus the
permutation property catch every real corruption.

=== WHERE A RADIX SORT ACTUALLY BREAKS ===

The implementation sorts integers, reaching floats through an
order-preserving bit transform (`convert_to_uint_preserve_order`). The
failure modes of that transform are specific and are not exercised by
drawing ordinary normal samples: the sign bit, negative zero, and the
infinities. Those get their own cases below.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_sort_values_and_indices.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_sort_values_and_indices.py
"""
from __future__ import annotations

import ctypes

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:  # pragma: no cover - shim
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore[assignment]

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor

_NP = {
    NBXDtype.float32: (np.float32, 4),
    NBXDtype.float16: (np.float16, 2),
    NBXDtype.int64: (np.int64, 8),
    NBXDtype.int32: (np.int32, 4),
}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy().reshape(t.shape)


def _check(x: np.ndarray, descending: bool, label: str) -> None:
    v, i = W.sort_wrapper(NBXTensor.from_numpy(x), dim=-1, descending=descending)
    got_v = _d2h(v).astype(np.float32)
    got_i = _d2h(i)
    ref = x.astype(np.float32)
    n = x.shape[-1]

    want = np.sort(ref, axis=-1)
    if descending:
        want = want[..., ::-1]

    # 1. VALUES — the sorted sequence itself.
    assert np.array_equal(got_v, want), (
        f"{label}: wrong VALUES\n  got  {got_v.ravel()[:8]}\n"
        f"  want {want.ravel()[:8]}")

    # 2. INDICES — each row must be a permutation of range(n). This is
    #    what catches a pad or sentinel slot leaking into the output,
    #    which is how the sibling top-k defect presented.
    flat_i = got_i.reshape(-1, n)
    for r, row in enumerate(flat_i):
        assert np.array_equal(np.sort(row), np.arange(n)), (
            f"{label}: row {r} indices are not a permutation "
            f"(min={row.min()} max={row.max()} unique={len(np.unique(row))})")

    # 3. COHERENCE — every index must carry the value it was returned
    #    with. A kernel can sort the values correctly and still permute
    #    the indices independently; only this catches that.
    assert np.array_equal(np.take_along_axis(ref, got_i, axis=-1), got_v), (
        f"{label}: index/value incoherent")


@pytest.mark.parametrize("n", [16, 64, 100, 255, 256, 1000, 1024, 4096])
@pytest.mark.parametrize("descending", [True, False])
def test_sort_random(n: int, descending: bool) -> None:
    """Powers of two and deliberately awkward lengths between them."""
    if not _has_gpu():
        pytest.skip("no GPU")
    x = np.random.default_rng(n).standard_normal((1, n)).astype(np.float32)
    _check(x, descending, f"n={n} desc={descending}")


@pytest.mark.parametrize("shape", [(4, 128), (2, 1000), (8, 64)])
def test_sort_batched(shape) -> None:
    """Several rows at once: each row sorts independently, and an index
    from one row must never appear in another."""
    if not _has_gpu():
        pytest.skip("no GPU")
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    _check(x, True, f"shape={shape}")


@pytest.mark.parametrize("descending", [True, False])
def test_sort_signed_and_zero(descending: bool) -> None:
    """The sign bit and negative zero — where an order-preserving bit
    transform fails if it flips the wrong bits."""
    if not _has_gpu():
        pytest.skip("no GPU")
    x = np.array([[0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 3.5, -3.5,
                   1e30, -1e30, 2.0, -2.0, 0.5, -0.5, 7.0, -7.0]],
                 dtype=np.float32)
    _check(x, descending, f"signed/zero desc={descending}")


@pytest.mark.parametrize("descending", [True, False])
def test_sort_infinities(descending: bool) -> None:
    """+-inf must sort to the extremes, not wrap around.

    The sibling top-k defect was exactly a non-finite value entering a
    permutation primitive that could not carry it.
    """
    if not _has_gpu():
        pytest.skip("no GPU")
    x = np.array([[np.inf, -np.inf, 0.0, 1.0, -1.0, np.inf, -np.inf, 2.0]],
                 dtype=np.float32)
    _check(x, descending, f"infinities desc={descending}")


def test_sort_all_equal() -> None:
    """Every value identical: any order is correct, but the indices must
    still be a permutation and must still carry their value."""
    if not _has_gpu():
        pytest.skip("no GPU")
    _check(np.full((1, 512), 1.5, dtype=np.float32), True, "all-equal")


def test_sort_already_sorted() -> None:
    """Sorted and reverse-sorted input — the degenerate cases a
    partitioning bug survives on random data."""
    if not _has_gpu():
        pytest.skip("no GPU")
    asc = np.arange(1024, dtype=np.float32)[None, :]
    _check(asc, True, "already ascending")
    _check(asc[:, ::-1].copy(), True, "already descending")


def test_sort_fp16() -> None:
    """fp16 input: the bit transform is width-dependent."""
    if not _has_gpu():
        pytest.skip("no GPU")
    x = np.random.default_rng(1).standard_normal((1, 512)).astype(np.float16)
    _check(x, True, "fp16")
    _check(x, False, "fp16 ascending")


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    cases = []
    for n in (16, 64, 100, 255, 256, 1000, 1024, 4096):
        for d in (True, False):
            cases.append((f"random n={n} desc={d}",
                          lambda n=n, d=d: test_sort_random(n, d)))
    for shape in ((4, 128), (2, 1000), (8, 64)):
        cases.append((f"batched {shape}",
                      lambda s=shape: test_sort_batched(s)))
    for d in (True, False):
        cases.append((f"signed/zero desc={d}",
                      lambda d=d: test_sort_signed_and_zero(d)))
        cases.append((f"infinities desc={d}",
                      lambda d=d: test_sort_infinities(d)))
    cases += [("all-equal", test_sort_all_equal),
              ("already-sorted", test_sort_already_sorted),
              ("fp16", test_sort_fp16)]
    for label, fn in cases:
        try:
            fn()
            print(f"  PASS  {label}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL  {label}\n        {str(exc).splitlines()[0]}")
        except Exception as exc:  # a crash is a finding too
            fails += 1
            print(f"  ERROR {label}\n        {type(exc).__name__}: {exc}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
