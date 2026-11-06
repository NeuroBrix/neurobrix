"""The launcher hands out a driver only for a backend it implements, and refuses every other by name.

`_resolve_driver` answered the CUDA driver for any backend but Metal and for a backend it could not read: on a HIP
machine an AMD card failed inside libcuda at its first launch, and an unreadable backend became CUDA silently (the
Mac, 2026-10-30 00:40; ZERO FALLBACK). What would this file do on that code? The hip cell gets a CudaDriver (RED),
the unreadable cell gets a CudaDriver instead of the reading's own error (RED). Since 2026-11-06 hip has its driver
(`HipDriver`, prepared, not yet run on an AMD card): the refusal stands for a backend that has none."""
import pytest

from nbxtensor.backend import launcher as L
from nbxtensor.backend import runtime as R


@pytest.fixture(autouse=True)
def _cuda_sentinel(monkeypatch):
    """The CUDA driver as a sentinel: which driver is chosen is the question, card-less (its instance calls cuInit)."""
    monkeypatch.setattr(L.CudaDriver, "instance", classmethod(lambda cls: "the CUDA driver"))
    monkeypatch.setattr(L.HipDriver, "instance", classmethod(lambda cls: "the HIP driver"))


def test_a_backend_without_a_driver_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(R, "gpu_backend", lambda: "level_zero")
    with pytest.raises(RuntimeError, match="no launch driver for the 'level_zero' backend"):
        L._resolve_driver()


def test_hip_is_served_its_driver_never_cuda(monkeypatch):
    monkeypatch.setattr(R, "gpu_backend", lambda: "hip")
    assert L._resolve_driver() == "the HIP driver"


def test_an_unreadable_backend_is_its_own_error_never_cuda(monkeypatch):
    def unreadable():
        raise RuntimeError("no GPU runtime could be loaded")
    monkeypatch.setattr(R, "gpu_backend", unreadable)
    with pytest.raises(RuntimeError, match="no GPU runtime could be loaded"):
        L._resolve_driver()


def test_cuda_is_served_its_driver(monkeypatch):
    monkeypatch.setattr(R, "gpu_backend", lambda: "cuda")
    assert L._resolve_driver() == "the CUDA driver"
