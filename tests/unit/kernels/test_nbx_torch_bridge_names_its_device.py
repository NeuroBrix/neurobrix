"""The NBX->torch bridge must name the device it is actually on.

It named `cuda:{idx}` on every machine. On Apple that is
`AssertionError: Torch not compiled with CUDA enabled`, raised at the
boundary before any number is compared — which is why 161 oracle tests in
this directory failed here without ever running their kernel.

Fixing the name then exposed a second defect the first one hid: the copy
wrote into the torch tensor's `data_ptr()`, which is valid only when torch's
allocator and ours sit on the same device heap. On Metal it is a SIGSEGV.
Both are settled by declared tables, not by asking which vendor we are on.
"""
import numpy as np
import pytest

from neurobrix.kernels import nbx_tensor as nt


def test_every_backend_has_a_torch_name_and_a_heap_answer():
    """The two tables must cover the same backends — a backend named but not
    answered (or the reverse) is the shape of a silent guess."""
    assert set(nt._TORCH_DEVICE_BY_BACKEND) == set(nt._TORCH_SHARES_DEVICE_HEAP)


def test_the_internal_token_is_not_what_torch_is_told():
    """`NBXTensor._device` is the string "cuda" on every backend by design.
    The bridge must not pass that token through."""
    assert nt._TORCH_DEVICE_BY_BACKEND["metal"] == "mps"
    # torch's ROCm build answers to "cuda"; only Apple differs
    assert nt._TORCH_DEVICE_BY_BACKEND["hip"] == "cuda"


def test_only_a_shared_heap_may_be_written_into():
    assert nt._TORCH_SHARES_DEVICE_HEAP["metal"] is False, (
        "writing into a torch MPS tensor's data_ptr from our allocator "
        "segfaults; measured rc=139")
    assert nt._TORCH_SHARES_DEVICE_HEAP["cuda"] is True
    assert nt._TORCH_SHARES_DEVICE_HEAP["hip"] is True


def test_an_unknown_backend_refuses_by_name(monkeypatch):
    monkeypatch.setattr(nt, "_detect_gpu_backend", lambda: "quantum")
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK.*'quantum'"):
        nt.torch_device_str(0)


def test_the_bridge_carries_the_values_on_this_machine():
    """The real call, on whatever backend this machine has."""
    a = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
    t = nt.NBXTensor.from_numpy(np.ascontiguousarray(a))
    out = nt.nbx_to_torch(t)
    assert out.flatten().tolist() == a.ravel().tolist()
    assert str(out.device).startswith(
        nt._TORCH_DEVICE_BY_BACKEND[nt._detect_gpu_backend()])


def test_an_empty_tensor_still_crosses():
    t = nt.NBXTensor.from_numpy(np.zeros((0, 4), dtype=np.float32))
    out = nt.nbx_to_torch(t)
    assert tuple(out.shape) == (0, 4)


def test_every_dtype_crosses_including_the_one_numpy_cannot_name():
    """The host path goes through raw bytes, not `.numpy()`.

    `_DTYPE_TYPESTR` maps bfloat16 to '<V2' — an opaque 2-byte void, because
    numpy has no bfloat16 — and `numpy()` falls back to '<f4' for anything
    absent from that table. Either way a bf16 tensor would not arrive, and
    bf16 is what most of the hub runs in.
    """
    from neurobrix.kernels.nbx_tensor import NBXDtype

    a = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
    src = nt.NBXTensor.from_numpy(np.ascontiguousarray(a))
    expected = a.ravel().tolist()

    for nbx_dt in (NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16):
        out = nt.nbx_to_torch(src.to(nbx_dt))
        assert out.float().flatten().tolist() == expected, f"{nbx_dt} lost its values"
        assert out.dtype == nt.nbx_dtype_to_torch(nbx_dt)


def test_bool_survives_the_crossing():
    b = nt.NBXTensor.from_numpy(np.array([[True, False], [False, True]]))
    assert nt.nbx_to_torch(b).flatten().tolist() == [True, False, False, True]
