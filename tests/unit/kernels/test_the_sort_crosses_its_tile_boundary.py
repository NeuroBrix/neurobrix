"""The radix sort is right on BOTH sides of its tile boundary, and loses nothing.

The suite's `test_sort_random` parametrises 16 … 4096 and caught this only at
4096. Bisected 2026-09-17 on triton-ext, the boundary is exact:

    n <= 2048   correct
    n == 2049   WRONG, and not merely misordered — 9 of 2049 values ABSENT
    n == 4096   WRONG, 1726 absent

2048 is `SWEEP_TILE` in `sort_wrapper`. The scatter kernel carried a DECOUPLED
LOOKBACK: each tile published its per-bin count with `cache_modifier=".cg"` and
then spun on its predecessors' flags (`volatile=True`) to learn how many of its
bin came earlier. That needs forward progress across CTAs and cross-CTA
visibility for those cache hints — Triton promises neither inside a kernel, and
Metal does not make threadgroups co-resident. The spin read stale flags rather
than blocking, two tiles computed the same destinations, and elements were
overwritten.

It is now three launches: per-tile counts, an exclusive scan along the tile
axis, then the scatter. That is the shape `cumsum_wrapper` in this tree already
uses (`scan_part_sum_kernel`, then `add_base_sum_kernel` when there is more than
one part), measured correct at eight tiles on the same backend the same day.
Triton orders kernel launches; it orders nothing between CTAs.

Two things this file pins that the parametrised test does not:

  * the BOUNDARY itself — one element either side of a tile;
  * PERMUTATION, which is the property that broke. "Sorted" alone would pass a
    result that dropped a thousand values and sorted what was left.
"""
from __future__ import annotations

import numpy as np
import pytest


def _sorted_desc(host):
    from neurobrix.kernels import launcher as L
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import sort_wrapper

    L.install()
    got = sort_wrapper(NBXTensor.from_numpy(host.copy()), descending=True)
    out = got[0] if isinstance(got, (tuple, list)) else got
    return out.to_cpu().numpy()


def _tile() -> int:
    """`SWEEP_TILE`, read from the wrapper rather than restated here."""
    import inspect
    import re

    from neurobrix.kernels import wrappers

    src = inspect.getsource(wrappers.sort_wrapper)
    m = re.search(r"^\s*SWEEP_TILE\s*=\s*(\d+)", src, re.M)
    assert m, "sort_wrapper no longer names SWEEP_TILE; re-point this test"
    return int(m.group(1))


@pytest.mark.parametrize("delta", [-1, 0, 1, 2])
def test_nothing_is_lost_across_the_tile_boundary(delta):
    n = _tile() + delta
    rng = np.random.default_rng(7)
    host = rng.standard_normal(n).astype(np.float32)

    out = _sorted_desc(host)

    assert np.array_equal(np.sort(out), np.sort(host)), (
        f"n={n}: the result is not a PERMUTATION of the input — "
        f"{len(set(map(float, host)) - set(map(float, out)))} value(s) absent. "
        f"Two tiles wrote the same destinations.")
    assert np.all(out[:-1] >= out[1:]), f"n={n}: not descending"
    assert np.array_equal(out, np.sort(host)[::-1]), f"n={n}: wrong order"


def test_the_scatter_does_not_spin_on_other_programs():
    """Structural, and read from the AST so the docstring above cannot fail it.

    A lookback is not a thing to be tuned back in: it is a cross-CTA dependency
    inside one kernel, which Triton does not order. If it returns, this says so
    at the seam rather than at n=2049 in somebody's model.
    """
    import ast
    import inspect

    from neurobrix.kernels.ops import sort_op

    tree = ast.parse(inspect.getsource(sort_op))
    scatter = next((f for f in ast.walk(tree)
                    if isinstance(f, ast.FunctionDef)
                    and f.name == "radix_sort_scatter_kernel"), None)
    assert scatter is not None, "radix_sort_scatter_kernel is gone; re-point this test"

    whiles = [w for w in ast.walk(scatter) if isinstance(w, ast.While)]
    assert not whiles, (
        "the scatter kernel spins again — a `while` inside it is a cross-CTA "
        "wait, and the cross-tile scan belongs in its own launch")
