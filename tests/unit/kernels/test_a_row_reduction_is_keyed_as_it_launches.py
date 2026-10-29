"""Sum, mean and amax run their certified configuration (chantier 1): each wrapper launches its kernel with the key
`launch_keys.row_reduce_launches` computes — the reduced elements a row (one dim, several, or all), every other
element a row's index, the input after the `dtype` keyword's cast — and the values are the reductions'.

What this test would do if the code were wrong: a fixed launch never asks the tuner — red at `_Launched`; rows and
row length swapped, or the cast's dtype ignored — red at the comparison; a tile that drops elements — red at the
values."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from nbxtensor.ops import launch_keys as LK  # noqa: E402
from nbxtensor.ops import wrappers as W  # noqa: E402
from nbxtensor.tensor import NBXTensor, NBXDtype  # noqa: E402
from nbxtensor.ops.autotune_cache import key_of  # noqa: E402

F32, F16 = NBXDtype.float32, NBXDtype.float16
_FN = {"mean": "mean_wrapper", "sum": "sum_wrapper", "amax": "amax_wrapper"}


class _Launched(Exception):
    pass


def _t(a, dt):
    x = NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("cuda")
    return x if dt == F32 else x.to(dt)


@pytest.mark.parametrize("red", sorted(_FN))
@pytest.mark.parametrize("shape,dim", [((4, 77, 768), -1), ((2, 64, 9, 9), 1), ((3, 5, 7), None)])
@pytest.mark.parametrize("dt", [F32, F16])
def test_the_launch_is_its_key(monkeypatch, red, shape, dim, dt):
    import importlib
    K_ = getattr(importlib.import_module(f"nbxtensor.ops.{red}"), f"{red}_forward_kernel")
    seen = []

    def run(*args, **kwargs):
        seen.append(key_of(K_, args, kwargs))
        raise _Launched
    monkeypatch.setattr(K_, "run", run)
    x = _t(np.random.default_rng(0).standard_normal(shape), dt)
    with pytest.raises(_Launched):
        getattr(W, _FN[red])(x, dim)
    feat = int(np.prod(shape)) if dim is None else shape[dim]
    want = LK.row_reduce_launches(red, int(np.prod(shape)) // feat, feat, dt)
    assert want[0][0] == LK.ROW_REDUCE[red] and seen == [want[0][1]], (seen, want)


@pytest.mark.parametrize("red", sorted(_FN))
@pytest.mark.parametrize("shape,dim", [((3, 100000), -1), ((300, 33), -1), ((2, 4, 6), 1)])
def test_the_values_are_the_reduction(red, shape, dim):
    a = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    got = getattr(W, _FN[red])(_t(a, F32), dim).to("cpu").numpy()
    want = {"mean": np.mean, "sum": np.sum, "amax": np.max}[red](a.astype(np.float64), axis=dim)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-3)


def test_a_candidate_block_stays_within_one_block_s_registers(monkeypatch):
    """A 128 x 4 096 block spilled and ptxas took minutes per candidate (2026-10-23): the candidates of a long row
    stay within `memory.max_registers_per_block` fp32 elements (V100's 65 536). Red if the bound is not applied."""
    from nbxtensor.ops import _configs as K
    from nbxtensor.ops import wrappers as W
    regs = 65536
    real = W._arch_param
    monkeypatch.setattr(W, "_arch_param", lambda sec, key, default: regs if key == "max_registers_per_block"
                        else real(sec, key, default))
    keep = K.prune_row_reduce(K.row_reduce_configs(), {"batch_dim": 100000, "feat_dim": 65536})
    assert keep and all(c.kwargs["BLOCK_SIZE_BATCH"] * c.kwargs["BLOCK_SIZE_FEAT"] <= regs for c in keep)
