"""Unit test for the Triton linspace dispatch path (D1).

Regression guard for P-CORRECTNESS-SILENT-FAILURES D1:
`neurobrix.kernels.dispatch._create_linspace` used to return an
UNINITIALISED `NBXTensor.empty` (it imported `fill_kernel` but never
called any kernel), so every `aten::linspace` in a `--triton` /
`--triton-sequential` DAG silently produced random memory. This test
compares the wired kernel against `torch.linspace` bit-exactly.

It MUST fail on the pre-fix code (garbage memory != torch.linspace)
and pass once the kernel is wired.

Runnable two ways:
  - pytest:  PYTHONPATH=src /usr/bin/python3 -m pytest tests/unit/kernels/test_linspace.py -v
  - script:  PYTHONPATH=src <gpu-venv>/bin/python tests/unit/kernels/test_linspace.py
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
        def approx(x, **k):
            return x

    pytest = _NoPytest()  # type: ignore

import torch

from neurobrix.kernels.dispatch import _create_linspace
from neurobrix.kernels.nbx_tensor import NBXDtype, nbx_to_torch

# (NBXDtype, torch dtype, exact?) — fp32 is bit-exact vs torch; fp16/bf16
# use a 1-ULP tolerance because the bidirectional FlagGems kernel and
# torch.linspace round interior points on different accumulation paths
# (endpoints stay exact in both — asserted separately below).
_DTYPES = [
    (NBXDtype.float32, torch.float32, True),
    (NBXDtype.float16, torch.float16, False),
    (NBXDtype.bfloat16, torch.bfloat16, False),
]
_SIZES = [1, 2, 50, 10000]
# (start, end): an increasing and a decreasing range that straddles 0.
_RANGES = [(0.0, 1.0), (-3.0, 7.0)]

# 1 ULP at the magnitude of the range, per dtype.
_ULP = {torch.float16: 2 ** -10, torch.bfloat16: 2 ** -7, torch.float32: 0.0}


def _run(start: float, end: float, steps: int,
         nbx_dt: NBXDtype, torch_dt: torch.dtype):
    out = _create_linspace(start, end, steps, dtype=nbx_dt, device="cuda")
    got = nbx_to_torch(out).reshape(-1)
    ref = torch.linspace(start, end, steps, dtype=torch_dt, device="cuda")
    return got, ref


@pytest.mark.parametrize("nbx_dt,torch_dt,exact", _DTYPES)
@pytest.mark.parametrize("steps", _SIZES)
@pytest.mark.parametrize("start,end", _RANGES)
def test_linspace_matches_torch(start, end, steps, nbx_dt, torch_dt, exact):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    got, ref = _run(start, end, steps, nbx_dt, torch_dt)

    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"

    if exact:
        # fp32: strict bit-exactness vs torch.linspace.
        assert torch.equal(got, ref), (
            f"fp32 linspace({start},{end},{steps}) not bit-exact:\n"
            f" got[:5]={got[:5].tolist()} ref[:5]={ref[:5].tolist()}\n"
            f" max|d|={(got.float()-ref.float()).abs().max().item()}"
        )
    else:
        tol = _ULP[torch_dt] * max(1.0, abs(end - start))
        assert torch.allclose(got.float(), ref.float(), rtol=0.0, atol=tol), (
            f"{torch_dt} linspace({start},{end},{steps}) exceeds 1 ULP:\n"
            f" max|d|={(got.float()-ref.float()).abs().max().item()} tol={tol}"
        )

    # Endpoint exactness is a hard invariant of the bidirectional kernel
    # (forward from start, backward from end) regardless of dtype.
    if steps == 1:
        assert got[0].item() == pytest.approx(start, abs=_ULP[torch_dt]), (
            f"steps==1 must yield [start]={start}, got {got[0].item()}"
        )
    else:
        assert got[0].item() == ref[0].item(), "start endpoint not exact"
        assert got[-1].item() == ref[-1].item(), "end endpoint not exact"


if __name__ == "__main__":  # production-torch fidelity run (no pytest dep)
    fails = 0
    for st, en in _RANGES:
        for s in _SIZES:
            for ndt, tdt, ex in _DTYPES:
                g, r = _run(st, en, s, ndt, tdt)
                ok = (torch.equal(g, r) if ex else torch.allclose(
                    g.float(), r.float(), rtol=0.0,
                    atol=_ULP[tdt] * max(1.0, abs(en - st))))
                tag = "OK " if ok else "FAIL"
                if not ok:
                    fails += 1
                print(f"[{tag}] linspace({st},{en},{s}) {tdt} "
                      f"max|d|={(g.float()-r.float()).abs().max().item():.3e}")
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    raise SystemExit(1 if fails else 0)
