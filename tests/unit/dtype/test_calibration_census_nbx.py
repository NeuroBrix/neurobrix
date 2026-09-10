"""The precision census observes the Triton engine's NBXTensor outputs
through the engine's own kernels (R33: no torch), and a record's measured
arm preference is honoured by the resolver.

Pins:
  GPU: observe() on an NBXTensor records its largest finite |x| and flags a
       non-finite value, merged with the ATen accumulators by finalize().
  CPU: a record whose `prefer[arch]` is "conservative" makes resolve()
       return the conservative triple, with the timing it was measured with.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.core.dtype import calibration as cal


def _gpu():
    try:
        from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor
        NBXTensor.empty((1,), NBXDtype.float32, "cuda:0")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="needs a GPU")
def test_census_observes_nbx_outputs():
    from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
    c = cal.RangeCensus()
    a = np.array([[1.0, -7.5], [3.0, 0.5]], dtype=np.float32)
    c.observe("aten.add::0", NBXTensor.from_numpy(a))
    b = np.array([2.0, np.inf, -1.0], dtype=np.float32)
    c.observe("aten.mul::1", NBXTensor.from_numpy(b))
    c.observe("aten.mul::1", NBXTensor.from_numpy(np.array([4.0], dtype=np.float16)))
    DeviceAllocator.sync_device()
    got = c.finalize()
    assert got["aten.add::0"] == 7.5
    assert got["aten.mul::1"] == 4.0                    # the finite maximum, across two observations
    assert c.non_finite_ops() == ["aten.mul::1"]
    c.observe("aten.arange::0", NBXTensor.from_numpy(np.arange(3, dtype=np.int64)))
    assert "aten.arange::0" not in c.finalize()         # integers are not recorded


def test_measured_preference_keeps_the_conservative_path(monkeypatch, tmp_path):
    from neurobrix.core.runtime import precision_contract as pc
    dag = {"ops": {"aten.mm::0": {"op_type": "aten.mm", "inputs": [], "outputs": ["t0"]}}}
    rec = cal.CalibrationRecord.build("m", "model", dag, {"aten.mm::0": 1.0}, stimulus={}, passes=1,
                                      reference="conservative")
    rec.timing["cuda-70"] = {"conservative_s": 1.0, "calibrated_s": 1.2, "identical": True, "reps": 3}
    rec.prefer["cuda-70"] = "conservative"
    monkeypatch.setattr(pc, "load_calibration", lambda *a, **k: rec)
    from neurobrix.triton import autotune_cache as atc
    monkeypatch.setattr(atc, "_arch_fingerprint", lambda: "cuda-70")
    monkeypatch.delenv(pc.FLAG_ENV, raising=False)
    safe, pins, narrow = pc.resolve(str(tmp_path), "model", dag, compute_dtype="float16")
    assert (safe, pins, narrow) == (False, frozenset(), frozenset())
    rec.prefer.clear()
    safe, pins, narrow = pc.resolve(str(tmp_path), "model", dag, compute_dtype="float16")
    assert safe is True                                  # without the preference the record applies
    again = cal.CalibrationRecord.from_dict(rec.to_dict())
    assert again.timing["cuda-70"]["calibrated_s"] == 1.2   # the fields round-trip
