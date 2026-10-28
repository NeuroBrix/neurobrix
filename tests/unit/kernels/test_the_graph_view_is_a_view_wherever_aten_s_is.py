"""The graph's aten.view is a VIEW wherever ATen's is one, a strided input included (`dispatch._view_strides`, ATen's
`computeStride`): the graph was traced under ATen, where it copies nothing, and the Triton engine copied every
non-contiguous input — mochi-1-preview's VAE held a 3 GB copy per permute -> view the graph never priced (2026-10-28).
ATen is the oracle here (a test, never the engine).

What this test would do if the code were wrong: a stride differing from ATen's view, or a view where ATen refuses (red)."""
import itertools
import random

import pytest

torch = pytest.importorskip("torch")

from neurobrix.kernels.dispatch import _view_strides


def _candidates(shape):
    n = 1
    for d in shape:
        n *= d
    out = {tuple(shape), (n,), (1, n)}
    for i in range(1, len(shape)):
        left, right = 1, 1
        for d in shape[:i]:
            left *= d
        for d in shape[i:]:
            right *= d
        out.add((left, right))
    for i in range(len(shape)):
        d = shape[i]
        for f in (2, 3):
            if d % f == 0:
                out.add(tuple(shape[:i]) + (f, d // f) + tuple(shape[i + 1:]))
    return out


def test_view_strides_match_aten_on_permuted_layouts():
    rng = random.Random(0)
    checked = 0
    for _ in range(300):
        nd = rng.randint(1, 4)
        base = [rng.choice([1, 2, 3, 4, 6]) for _ in range(nd)]
        t = torch.empty(base)
        for perm in itertools.permutations(range(nd)):
            v = t.permute(*perm)
            for new in _candidates(list(v.shape)):
                try:
                    want = tuple(v.view(new).stride())
                except RuntimeError:
                    want = None
                got = _view_strides(tuple(v.shape), tuple(v.stride()), tuple(new))
                got = tuple(got) if got is not None else None
                if want is not None and got is None:
                    pytest.fail(f"ATen views {tuple(v.shape)}{tuple(v.stride())} as {new}; refused here")
                if got is not None:
                    assert want is not None, f"a view of {tuple(v.shape)}{tuple(v.stride())} as {new} ATen refuses"
                    # strides of size-1 dims carry no layout: compare where the dim is real
                    assert all(g == w for g, w, s in zip(got, want, new) if s != 1), (v.shape, v.stride(), new, got, want)
                checked += 1
    assert checked > 1000
