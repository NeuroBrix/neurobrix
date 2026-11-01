"""A recording whose slab proved too small still ran the step, and the caller keeps its outputs: they are handed over as
heap copies, so the retired slab is freed once the run's own tensors die (Allegro triton on a V100-16GB, 2026-11-01: a
VAE chunk's output pinned a 7.5 GB undersized slab through the next chunks and the 2.5 GB malloc that followed was
refused; with the hand-over the run completed at 11 418 MB, its judged sha).

What this test would do if the code were wrong: the output in the arena still points inside the slab (red). One card."""
from types import SimpleNamespace

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor
from neurobrix.triton import replay

pytestmark = pytest.mark.usefixtures("needs_device")


def test_an_output_carved_from_the_slab_is_replaced_by_a_heap_copy_with_its_bytes():
    carved = NBXTensor.from_numpy(np.arange(16, dtype=np.float32)).to("cuda:0")
    kept = NBXTensor.from_numpy(np.ones(4, dtype=np.float32)).to("cuda:0")
    slab = SimpleNamespace(base=carved.data_ptr(), size=carved._nbytes)
    seq = SimpleNamespace(dag={"output_tensor_ids": ["out", "other"]}, _tid_to_slot={"out": 0, "other": 1},
                          _arena=[carved, kept])
    assert replay._outputs_off_the_slab(seq, slab) == 1
    assert not (slab.base <= seq._arena[0].data_ptr() < slab.base + slab.size)
    assert seq._arena[1] is kept
    assert np.array_equal(seq._arena[0].to_cpu().numpy(), np.arange(16, dtype=np.float32))
