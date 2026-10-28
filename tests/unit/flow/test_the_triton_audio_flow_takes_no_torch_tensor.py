"""The Triton audio flow takes NBXTensor (and the host's numpy arrays) end to end — R33 at execution, supervisor
2026-10-28 20:10: `triton/flow/audio.py` converted any tensor-like value it was handed with `.detach().cpu().numpy()`
(`_torch_to_nbx`), so a torch tensor reaching the Triton branch ran on silently. A value that is neither is refused by
name; nothing converts a foreign tensor."""
import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.triton.flow import audio as A


class _ForeignTensor:
    """What a torch tensor looks like to a duck test: shape, dtype, device, detach/cpu/numpy."""
    shape = (1, 4)
    dtype = "int64"
    device = "cpu"

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((1, 4), dtype=np.int64)


def test_a_foreign_tensor_is_refused_by_name():
    with pytest.raises(TypeError, match="R33"):
        A._nbx(_ForeignTensor(), "global.input_ids")


def test_numpy_and_nbx_pass():
    arr = np.arange(4, dtype=np.int64)[None, :]
    t = A._nbx(arr, "x")
    assert isinstance(t, NBXTensor) and tuple(t.shape) == (1, 4)
    assert A._nbx(t, "x") is t
    assert A._is_tensor(t) and not A._is_tensor(arr) and not A._is_tensor(None)


def test_a_foreign_tensor_met_among_the_resolved_values_is_refused_not_skipped():
    with pytest.raises(TypeError, match="R33"):
        A._is_tensor(_ForeignTensor())
