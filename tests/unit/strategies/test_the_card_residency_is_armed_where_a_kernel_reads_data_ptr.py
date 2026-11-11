"""The expert store's residency ON THE CARD (`NBXExpertStore.enable_device_decode`) launches its kernels over tables of
`data_ptr()` addresses. That is a property of the backend: where a tensor's `data_ptr()` is the address a kernel
dereferences (CUDA, HIP) it holds; on Metal `data_ptr()` is the CPU mapping (a kernel reads a pinned wrap's address,
through the launcher) and on Level Zero a kernel loads no address from memory at all. The strategy armed it on "Triton,
not in place, half precision" with no word of the backend, and the residency's own device test ran — and failed — on a
Mac (2026-11-11). No Apple run reached the code (a unified device's store is in place); one that did would have
stopped at its first decode token on the launcher's TypeError.

What these tests would do if the code were wrong: the strategy arms the card residency on a backend whose kernels
cannot read those tables; a backend without a stated answer is guessed instead of refused.
"""
from types import SimpleNamespace

import pytest

from nbxtensor.backend import capabilities as cap
from nbxtensor.dtype.nbx_dtype import NBXDtype
from neurobrix.core.strategies import expert_store as strategy


def _store(in_place=False, trace=None):
    return SimpleNamespace(in_place=in_place, trace=trace)


def _like(dtype=NBXDtype.float16):
    return (SimpleNamespace(_dtype=dtype),)


@pytest.mark.parametrize("backend, answer", [("cuda", True), ("hip", True), ("xpu", False), ("metal", False)])
def test_every_backend_states_whether_a_kernel_reads_data_ptr(monkeypatch, backend, answer):
    monkeypatch.setattr(cap._rt, "gpu_backend", lambda: backend)
    assert cap.backend_kernels_read_data_ptr() is answer


def test_a_backend_without_a_row_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(cap._rt, "gpu_backend", lambda: "newcard")
    with pytest.raises(RuntimeError, match="newcard.*_BACKEND_KERNELS_READ_DATA_PTR"):
        cap.backend_kernels_read_data_ptr()


@pytest.mark.parametrize("backend, armed", [("cuda", True), ("hip", True), ("xpu", False), ("metal", False)])
def test_a_copying_half_precision_triton_store_arms_the_card_residency_only_where_kernels_read_data_ptr(
        monkeypatch, backend, armed):
    monkeypatch.setattr(cap._rt, "gpu_backend", lambda: backend)
    assert strategy.card_residency_applies(True, _store(), _like()) is armed


@pytest.mark.parametrize("is_triton, store, like", [
    (False, _store(), _like()),                              # the PyTorch engine
    (True, _store(in_place=True), _like()),                  # nothing is copied: no slot to keep on the card
    (True, _store(trace=object()), _like()),                 # the routing's reuse curve is still being recorded
    (True, _store(), _like(NBXDtype.float32)),               # the decode band is the half-precision one
])
def test_the_other_conditions_stand_as_they_were(monkeypatch, is_triton, store, like):
    monkeypatch.setattr(cap._rt, "gpu_backend", lambda: "cuda")
    assert strategy.card_residency_applies(is_triton, store, like) is False
