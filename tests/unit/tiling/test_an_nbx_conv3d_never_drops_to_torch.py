"""A chunked rank-5 convolution of an NBXTensor runs the Triton branch's wrapper or fails by name: an import that fails
there is raised, never answered by the torch path below it (`tiling_engine.conv3d_chunked` caught ImportError and
fell through to F.conv3d on the NBXTensor — a silent fallback, and torch on the Triton branch).

What this test would do if the code were wrong: the ImportError swallowed — the torch path runs (or fails with a
torch error) instead of the import error (red). No card: the NBXTensor stays on the host."""
import sys

import numpy as np
import pytest


def test_a_failing_triton_import_is_raised_not_answered_by_torch(monkeypatch):
    from nbxtensor.tensor import NBXTensor
    from neurobrix.core.module import tiling_engine as TE
    x = NBXTensor.from_numpy(np.zeros((1, 2, 3, 4, 4), dtype=np.float32))
    w = NBXTensor.from_numpy(np.zeros((2, 2, 1, 1, 1), dtype=np.float32))
    monkeypatch.setitem(sys.modules, "nbxtensor.ops.conv_wrappers", None)      # its import now fails
    with pytest.raises(ImportError):
        TE.conv3d_chunked(x, w)
