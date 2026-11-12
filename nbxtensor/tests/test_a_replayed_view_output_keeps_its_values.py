"""A graph output that is a VIEW (a permute, a transpose) keeps its values through the replay's copies
(`nbxtensor.backend.frozen_plan._heap_copy` / `_copy_into`, moved from the engine 2026-11-12).

Allegro-TI2V's encoder ends on a permute to frames-first. The replay handed its outputs over by copying the view's
STORAGE bytes into a contiguous tensor of the view's shape — the values in the wrong order: every replayed tile of the
I2V conditioning came back scrambled, the 88-frame video a mosaic of tiles (2026-11-04).

What this test would do if the code were wrong: a storage copy for the hand-over, or into a recorded buffer — red at
the values. One card (skipped, said, without one)."""
import numpy as np
import pytest

from nbxtensor.backend import frozen_plan as replay
from nbxtensor.tensor import NBXTensor


@pytest.fixture(autouse=True)
def _card():
    try:
        NBXTensor.from_numpy(np.zeros(1, dtype=np.float32)).to("cuda:0")
    except Exception as e:                      # no card here
        pytest.skip(f"no accelerator: {e}")


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
