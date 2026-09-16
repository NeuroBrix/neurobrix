"""repeat_interleave's per-element loop bound is 32-bit, and it matches torch.

Kokoro reached aten.repeat_interleave after weight_norm was fixed, and was
refused: "scf.for with 64-bit loop bounds: the induction lowering would not
terminate (hang). Use 32-bit bounds." `repeats` arrives i64 by torch default;
the loop `range(0, repeats, BLOCK)` then has a 64-bit bound.

Casting `repeats` to i32 is the refusal's own remedy, safe because a
per-element count of 2**31 needs an 8 GB output for one element (the allocator
refuses first) and numerically inert on CUDA. The output must still match
torch.repeat_interleave.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_repeat_interleave_32bit_bound.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


def test_the_loop_bound_is_cast_to_int32():
    """Structural: the value feeding the range() is narrowed. Asserted on the
    source so it holds for the shapes no runtime test exercises."""
    import neurobrix.kernels.ops.repeat_interleave as R

    src = Path(R.__file__).read_text()
    assert ".to(tl.int32)" in src, (
        "the loop bound `repeats` must be cast to i32; a 64-bit scf.for bound "
        "hangs the Metal induction lowering")


@pytest.mark.skipif(sys.platform != "darwin", reason="the Metal path")
def test_repeat_interleave_matches_torch():
    import numpy as np
    torch = pytest.importorskip("torch")
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype

    repeats = np.array([3, 1, 4, 1, 5, 2], dtype=np.int64)
    # the tensor wrapper returns the INDEX map (0,0,0,1,2,2,2,2,...)
    got = W.repeat_interleave_tensor_wrapper(
        NBXTensor.from_numpy(repeats)).to_cpu().numpy().ravel()
    want = torch.repeat_interleave(
        torch.arange(len(repeats)), torch.from_numpy(repeats)).numpy()
    assert np.array_equal(got, want), (
        f"repeat_interleave index map differs from torch:\n got={got}\n want={want}")
