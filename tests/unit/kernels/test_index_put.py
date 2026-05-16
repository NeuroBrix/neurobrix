"""Unit test for the Triton index_put dispatch path (D2).

Regression guard for P-CORRECTNESS-SILENT-FAILURES D2:
`dispatch.py` mapped `aten::index_put` / `index_put_` to identity
lambdas (`lambda x, indices, values, acc=False: x`) — every scatter
write was silently DROPPED in `--triton` / `--triton-sequential`
(reachable via `dispatch()` in both triton/sequence.py:1511 and
triton/sequential.py:170). This test drives the real dispatch path
and compares against `torch.index_put_`.

It MUST fail on the pre-fix code (identity lambda returns the input
unchanged != torch result) and pass once the kernel is wired.

Runnable two ways:
  - pytest:  PYTHONPATH=src /usr/bin/python3 -m pytest tests/unit/kernels/test_index_put.py -v
  - script:  PYTHONPATH=src <gpu-venv>/bin/python tests/unit/kernels/test_index_put.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU runtime venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

        @staticmethod
        def raises(*a, **k):
            raise AssertionError("pytest.raises unavailable in script mode")

    pytest = _NoPytest()  # type: ignore

import numpy as np
import torch

from neurobrix.kernels.dispatch import dispatch
from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch

_NP = {"float32": np.float32, "float16": np.float16}
_TT = {"float32": torch.float32, "float16": torch.float16}


def _nbx(arr):
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


def _ref(x_np, idx_np, val_np, tt, accumulate):
    x = torch.from_numpy(x_np.astype(np.float32)).to(tt).cuda()
    idx = torch.from_numpy(idx_np).long().cuda()
    val = torch.from_numpy(val_np.astype(np.float32)).to(tt).cuda()
    x.index_put_((idx,), val, accumulate=accumulate)
    return x


def _got(x_np, idx_np, val_np, npdt, accumulate):
    fn = dispatch("aten::index_put")
    out = fn(_nbx(x_np.astype(npdt)), [_nbx(idx_np)],
             _nbx(val_np.astype(npdt)), accumulate)
    return nbx_to_torch(out)


# (config-id, x.shape, idx (int64), values.shape, accumulate)
_CONFIGS = [
    ("simple_1d",    (8,),   np.array([2, 5, 0], dtype=np.int64),     (3,),    False),
    ("fancy_multid", (6, 4), np.array([[1, 3, 0], [5, 2, 4]], np.int64), (2, 3, 4), False),
    ("accumulate",   (5, 3), np.array([1, 1, 4], dtype=np.int64),     (3, 3),  True),
    ("overwrite",    (5, 3), np.array([3, 0, 4], dtype=np.int64),     (3, 3),  False),
]


@pytest.mark.parametrize("dt", ["float32", "float16"])
@pytest.mark.parametrize("cid,xshape,idx,vshape,acc", _CONFIGS)
def test_index_put_matches_torch(cid, xshape, idx, vshape, acc, dt):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(xshape).astype(np.float32)
    v_np = rng.standard_normal(vshape).astype(np.float32)

    got = _got(x_np, idx, v_np, _NP[dt], acc).reshape(-1).float().cpu()
    ref = _ref(x_np, idx, v_np, _TT[dt], acc).reshape(-1).float().cpu()

    if acc:
        # accumulate uses atomic_add — order differs from torch's
        # sequential add (and torch CUDA also uses atomicAdd, so even
        # torch is not bit-stable here). Tolerance scaled to dtype.
        atol = 1e-3 if dt == "float32" else 1e-2
        assert torch.allclose(got, ref, rtol=0.0, atol=atol), (
            f"{cid}/{dt}: max|d|={(got-ref).abs().max().item()}")
    else:
        # pure scatter-copy — bit-exact in both fp32 and fp16.
        assert torch.equal(got, ref), (
            f"{cid}/{dt} not bit-exact: max|d|="
            f"{(got-ref).abs().max().item()}\n got={got[:8].tolist()}\n"
            f" ref={ref[:8].tolist()}")


def test_index_put_failfast_multi_index():
    """k>=2 advanced indices must raise, not silently mis-scatter."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fn = dispatch("aten::index_put")
    x = _nbx(np.zeros((4, 4), np.float32))
    i0 = _nbx(np.array([0, 1], np.int64))
    i1 = _nbx(np.array([2, 3], np.int64))
    v = _nbx(np.ones((2,), np.float32))
    with pytest.raises(NotImplementedError):
        fn(x, [i0, i1], v, False)


def test_index_put_failfast_bool_mask():
    """Boolean-mask index must raise (no nonzero kernel in catalogue)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fn = dispatch("aten::index_put")
    x = _nbx(np.zeros((4,), np.float32))
    mask = _nbx(np.array([True, False, True, False], dtype=np.bool_))
    v = _nbx(np.ones((2,), np.float32))
    with pytest.raises(NotImplementedError):
        fn(x, [mask], v, False)


if __name__ == "__main__":  # production-torch fidelity run (no pytest dep)
    fails = 0
    for dt in ("float32", "float16"):
        for cid, xs, ix, vs, ac in _CONFIGS:
            rng = np.random.default_rng(0)
            xn = rng.standard_normal(xs).astype(np.float32)
            vn = rng.standard_normal(vs).astype(np.float32)
            g = _got(xn, ix, vn, _NP[dt], ac).reshape(-1).float().cpu()
            r = _ref(xn, ix, vn, _TT[dt], ac).reshape(-1).float().cpu()
            ok = (torch.allclose(g, r, rtol=0.0,
                                 atol=(1e-3 if dt == "float32" else 1e-2))
                  if ac else torch.equal(g, r))
            fails += 0 if ok else 1
            print(f"[{'OK ' if ok else 'FAIL'}] {cid}/{dt} "
                  f"max|d|={(g-r).abs().max().item():.3e}")
    # fail-fast paths
    _f = dispatch("aten::index_put")
    for label, idxs, xs in (
        ("k>=2", [_nbx(np.array([0, 1], np.int64)),
                  _nbx(np.array([2, 3], np.int64))], (4, 4)),
        ("bool", [_nbx(np.array([True, False, True, False], np.bool_))], (4,)),
    ):
        try:
            _f(_nbx(np.zeros(xs, np.float32)), idxs,
               _nbx(np.ones((2,), np.float32)), False)
            print(f"[FAIL] {label} did not raise"); fails += 1
        except NotImplementedError:
            print(f"[OK ] {label} raised NotImplementedError")
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    raise SystemExit(1 if fails else 0)
