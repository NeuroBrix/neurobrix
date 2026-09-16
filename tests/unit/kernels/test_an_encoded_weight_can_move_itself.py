"""An encoded weight answers the same movers a dense one does.

Every placement path moves a weight by calling a mover ON THE TENSOR
(`strategies/base.py`, `zero3.py`, `strategies/triton/base.py`). On 2026-09-16
the vitrine's first LLM request died at `'QuantizedTensor' object has no
attribute 'to_cuda_async'`: Qwen3-Coder-30B-A3B-Instruct-int4g128 under a plan
that streams. The handle carried the read-only surface of a weight and none of
its movers.

No card needed: the three parts are stubs that record what they were asked,
which is exactly what the wrapper must delegate — and the metadata must ride
unchanged, since where the bytes live says nothing about the logical shape or
the view flag.

Injection: `to_cuda_async` removed again → the first test raised
AttributeError, which is the defect verbatim."""
import pytest

from neurobrix.kernels.quantized_tensor import QuantizedTensor


class _Part:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def to_cuda(self, device_idx=0):
        self.calls.append(("to_cuda", device_idx)); return _moved(self, "cuda", device_idx)

    def to_cuda_async(self, device_idx=0, stream=0):
        self.calls.append(("to_cuda_async", device_idx, stream)); return _moved(self, "cuda", device_idx)

    def to_cpu(self, pinned=False):
        self.calls.append(("to_cpu", pinned)); return _moved(self, "cpu", None)


def _moved(src, where, idx):
    m = _Part(src.name + f"@{where}{idx if idx is not None else ''}")
    return m


def _triplet():
    return QuantizedTensor(_Part("qweight"), _Part("scales"), _Part("qmins"), (4096, 1024), transposed=True)


def test_the_streaming_mover_moves_all_three_parts_on_the_stream():
    q = _triplet()
    out = q.to_cuda_async(2, stream=7)
    assert isinstance(out, QuantizedTensor)
    for part in (q.qweight, q.scales, q.qmins):
        assert part.calls == [("to_cuda_async", 2, 7)], part.name
    assert out.logical_shape == (4096, 1024) and out.transposed is True


def test_the_synchronous_movers_do_the_same_and_keep_the_metadata():
    q = _triplet()
    g = q.to_cuda(1)
    assert [p.calls[0][:2] for p in (q.qweight, q.scales, q.qmins)] == [("to_cuda", 1)] * 3
    assert g.logical_shape == (4096, 1024) and g.transposed is True
    h = _triplet().to_cpu(pinned=True)
    assert h.logical_shape == (4096, 1024) and h.transposed is True


def test_every_mover_a_dense_weight_answers_is_answered_here():
    """The placement paths call these three by name; a missing one is a defect
    that only appears under the plan that needs it."""
    for name in ("to_cuda", "to_cuda_async", "to_cpu", "pin_host", "t"):
        assert callable(getattr(QuantizedTensor, name, None)), name
