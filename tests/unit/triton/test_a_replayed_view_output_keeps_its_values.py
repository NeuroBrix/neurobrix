"""A graph output that is a VIEW (a permute, a transpose) keeps its values through the replay's hand-overs, and the
verify pass compares values, not storage bytes.

Allegro-TI2V's encoder ends on a permute to frames-first. The replay handed its outputs over by copying the view's
STORAGE bytes into a contiguous tensor of the view's shape — the values in the wrong order — and the verify hashed
storage bytes, so the scrambled copy matched the view it came from and the plan was VERIFIED: every replayed tile of the
I2V conditioning came back scrambled, the 88-frame video a mosaic of tiles (2026-11-04; judged coherent on 2026-10-25,
before the replay was the default).

What this test would do if the code were wrong: a storage copy for the hand-over — red at the values; a storage hash
for the verify — red at the hash of a view against its scrambled copy. One card."""
import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor
from neurobrix.triton import replay

pytestmark = pytest.mark.usefixtures("needs_device")


def _view():
    base = NBXTensor.from_numpy(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)).to("cuda:0")
    v = base.permute(1, 0, 2)                       # [3, 2, 4], not contiguous
    assert not v.is_contiguous()
    return base, v


def test_a_view_is_handed_over_with_its_values_in_order():
    base, v = _view()
    want = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)
    got = replay._heap_copy(v)
    assert got.data_ptr() != base.data_ptr()
    assert np.array_equal(got.to_cpu().numpy(), want)


def test_a_view_is_copied_into_a_recorded_buffer_by_its_values():
    _base, v = _view()
    frozen = NBXTensor.empty((3, 2, 4), v.nbx_dtype, "cuda:0")
    replay._copy_into(frozen, v)
    assert np.array_equal(frozen.to_cpu().numpy(), np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2))


def test_the_verify_hash_tells_a_view_from_its_scrambled_copy():
    from types import SimpleNamespace
    base, v = _view()
    scrambled = NBXTensor.empty((3, 2, 4), v.nbx_dtype, "cuda:0")
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.memcpy(scrambled.data_ptr(), v.data_ptr(), v._nbytes, 3)     # the former hand-over
    good = replay._heap_copy(v)
    seq = lambda t: SimpleNamespace(dag={"output_tensor_ids": ["o"]}, _tid_to_slot={"o": 0}, _arena=[t])  # noqa: E731
    assert replay._output_hashes(seq(v)) == replay._output_hashes(seq(good))
    assert replay._output_hashes(seq(v)) != replay._output_hashes(seq(scrambled))
