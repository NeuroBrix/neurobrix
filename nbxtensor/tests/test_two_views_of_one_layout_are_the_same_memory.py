"""`NBXTensor.is_same_memory`: two view objects of one layout are the same elements — a copy between them would write
nothing, and the engine's replay asks before it launches one.

A served model's weights are bound as views (a transposed weight); after a request's fresh bind the view OBJECT is
rebuilt over the same card memory, and the replay's scan — which compares objects — copied each of them onto itself
with a strided-copy launch: ~100 launches a token (Qwen3-Coder-30B-A3B served, 2026-11-11: 18 of 50 stacks of a
decode in that copy). No card: host tensors.

What this test would do if the code were wrong: `is_same_memory` true for a transposed view, a narrower view or
another dtype — a copy skipped that had to be made — RED; false for a rebuilt view of the same layout — the launch
kept."""
import numpy as np

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor


def _host(shape):
    t = NBXTensor.empty_cpu(shape, NBXDtype.float32)
    t.numpy()[...] = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return t


def test_a_rebuilt_view_of_one_layout_is_the_same_memory():
    w = _host((4, 6))
    assert w.t().is_same_memory(w.t()) and w.t() is not w.t()
    assert w.is_same_memory(w)


def test_any_other_layout_is_not():
    w = _host((4, 6))
    assert not w.t().is_same_memory(w)                     # same address, other strides and shape
    assert not w.narrow(0, 0, 2).is_same_memory(w)         # same address, fewer rows
    assert not w.narrow(0, 1, 2).is_same_memory(w.narrow(0, 0, 2))   # other address
    assert not _host((4, 6)).is_same_memory(w)             # other memory, equal values
