"""Activation proof of the TRITON post-loop finite gate (R30 mirror of the
compiled gate): an NBXTensor decoder output holding NaN is refused; a finite
one passes. GPU test — the isfinite/all wrappers are Triton kernels."""
import math

import numpy as np
import pytest
import torch

from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.triton.flow.iterative_process import _gate_component_outputs_finite

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")


def _nbx(values):
    return NBXTensor.from_numpy(np.ascontiguousarray(np.array(values, dtype=np.float32)))


def test_triton_gate_passes_finite_and_refuses_nan():
    ok = {"vae.output_0": _nbx([[1.0, 2.0], [3.0, 4.0]])}
    _gate_component_outputs_finite(ok, "vae")
    bad = {"vae.output_0": _nbx([[1.0, math.nan], [math.inf, 0.0]])}
    with pytest.raises(RuntimeError, match="non-finite output of post-loop component 'vae'"):
        _gate_component_outputs_finite(bad, "vae")
    # only the named component is inspected
    _gate_component_outputs_finite({"transformer.output_0": _nbx([[math.nan]])}, "vae")
