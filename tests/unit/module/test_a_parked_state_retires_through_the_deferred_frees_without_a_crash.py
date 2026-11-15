"""A temporal stream's parked state is retired by an engine's deferred frees (`nbxtensor` DeferredFrees.add reads
`nbytes()`): it holds no device bytes, and the batch releases without a card.

No card. Run: python -m pytest tests/unit/module/test_a_parked_state_retires_through_the_deferred_frees_without_a_crash.py
"""
from nbxtensor.memory.deferred_frees import DeferredFrees

from neurobrix.core.module.temporal_stream_runner import ParkedTensor


def test_a_parked_state_retires_through_the_deferred_frees_without_a_crash():
    parked = ParkedTensor.__new__(ParkedTensor)          # a state already on the host: no tensor library needed
    frees = DeferredFrees(cliff=1 << 30, count=8)
    frees.add(parked)
    assert len(frees) == 1 and frees.nbytes == 0
    assert frees.release(sync=False) == 0
