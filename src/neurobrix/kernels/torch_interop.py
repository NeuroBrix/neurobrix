"""The engine's torch boundary for NBXTensor: dtype and device names handed to torch, the device made current for torch,
and the copy of an NBXTensor into a torch tensor (D2D where torch shares the device heap, through the host where it
does not). These are the engine's, not the tensor library's: nbxtensor never imports torch (R-NBXT 6); exchange with
another tensor library goes through DLPack. Moved here from kernels/nbx_tensor.py when nbx_tensor.py entered nbxtensor
(M1, 2026-10-29), unchanged.
"""
from __future__ import annotations

from nbxtensor import tensor as _nt
import nbxtensor.backend.runtime as _nbx_runtime
from nbxtensor.memory.allocator import DeviceAllocator  # noqa: F401
from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size
from nbxtensor.tensor import NBXTensor


# ============================================================================
# TORCH DTYPE BOUNDARY — converts NBXDtype ↔ torch.dtype at system edges
# ============================================================================

def nbx_dtype_to_torch(d: NBXDtype):
    """Convert NBXDtype to torch.dtype. Used at NBXTensor↔torch boundaries."""
    import torch
    _MAP = {
        NBXDtype.float16: torch.float16, NBXDtype.bfloat16: torch.bfloat16,
        NBXDtype.float32: torch.float32, NBXDtype.float64: torch.float64,
        NBXDtype.int8: torch.int8, NBXDtype.int16: torch.int16,
        NBXDtype.int32: torch.int32, NBXDtype.int64: torch.int64,
        NBXDtype.uint8: torch.uint8, NBXDtype.bool_: torch.bool,
        NBXDtype.complex64: torch.complex64, NBXDtype.complex128: torch.complex128,
    }
    return _MAP[d]


# How each detected backend is named to TORCH. The engine's internal token
# for device memory is the string "cuda" on every backend (see
# `device_label`), and that token is deliberate — but it is not a name to
# hand to torch, which resolves it against the build it was compiled with.
# Torch's ROCm build also answers to "cuda"; only Apple differs.
_TORCH_DEVICE_BY_BACKEND = {"cuda": "cuda", "hip": "cuda", "metal": "mps"}

# Whether a torch tensor's `data_ptr()` is a pointer OUR allocator may write
# into. On CUDA and ROCm both allocators sit on the same driver heap, so a
# device-to-device copy straight into the torch tensor is valid and is what
# this bridge has always done. On Metal, torch owns its own MTLBuffers and
# ours are not those: writing there is not slow, it is a SIGSEGV (measured
# 2026-09-10, rc=139 at the first `nbx_to_torch` on this machine).
#
# This is a capability, not a vendor test — adding a backend is adding a row.
_TORCH_SHARES_DEVICE_HEAP = {"cuda": True, "hip": True, "metal": False}





def torch_device_str(device_idx: int) -> str:
    """The torch device string for the backend this process actually detected.

    Was `f"cuda:{idx}"`, written at the one place where the internal token
    crosses into torch. On Apple that asked torch for a CUDA device and got
    `AssertionError: Torch not compiled with CUDA enabled` — which is why
    every oracle test in tests/unit/kernels (161 of them, 13 files) failed
    here without ever comparing a number.
    """
    backend = _nbx_runtime.gpu_backend()
    name = _TORCH_DEVICE_BY_BACKEND.get(backend)
    if name is None:
        raise RuntimeError(
            f"ZERO FALLBACK: backend {backend!r} has no torch device name in "
            f"_TORCH_DEVICE_BY_BACKEND ({sorted(_TORCH_DEVICE_BY_BACKEND)}). "
            f"Adding a backend is adding a row, not guessing a name.")
    return f"{name}:{device_idx}"


#: Whether torch offers a device-SELECTION call for this backend. CUDA and
#: ROCm number their devices and require one to be current before a launch.
#: Apple's torch build exposes no `torch.mps.set_device` at all, because MPS is
#: a single device (`torch.mps.device_count()` is 1) — so there is nothing to
#: select and nothing to be current.
#:
#: This is a capability, not a vendor test — adding a backend is adding a row.
_TORCH_HAS_DEVICE_SELECTION = {"cuda": True, "hip": True, "metal": False}


def set_torch_device(device) -> None:
    """Make `device` current for torch, where that means anything.

    Takes an index or a torch device, as `torch.cuda.set_device` does.

    Was `torch.cuda.set_device(idx)`, written on the compiled engine's
    multi-device path. On Apple that raises
    `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'` —
    the same shape as the `f"cuda:{idx}"` that `torch_device_str` replaced: the
    engine's internal token for device memory handed to torch, which resolves
    it against the build it was compiled with.

    Measured 2026-09-11 on `CogVideoX-2b`: the compiled arm died in 14.7 s at
    `compiled_sequence.py:4136`, and no model measured before it had ever taken
    the multi-device path, so the line had never been reached on this machine.

    Doing nothing on a single-device backend is not a fallback: there is no
    selection to make, and a backend that HAS one and is missing from the table
    is refused rather than guessed.
    """
    backend = _nbx_runtime.gpu_backend()
    selects = _TORCH_HAS_DEVICE_SELECTION.get(backend)
    if selects is None:
        raise RuntimeError(
            f"ZERO FALLBACK: backend {backend!r} does not say whether torch "
            f"selects a device on it "
            f"({sorted(_TORCH_HAS_DEVICE_SELECTION)}). Adding a backend is "
            f"adding a row, not guessing.")
    if not selects:
        return
    import torch
    torch.cuda.set_device(device)


def nbx_to_torch(tensor: 'NBXTensor'):
    """Convert NBXTensor to torch.Tensor via D2D copy.

    Used at the boundary between triton execution and torch-based pipeline.
    """
    import torch
    backend = _nbx_runtime.gpu_backend()
    device = torch_device_str(tensor._device_idx)
    shares_heap = _TORCH_SHARES_DEVICE_HEAP.get(backend)
    if shares_heap is None:
        raise RuntimeError(
            f"ZERO FALLBACK: backend {backend!r} does not say whether torch "
            f"shares its device heap "
            f"({sorted(_TORCH_SHARES_DEVICE_HEAP)}). Copying into a foreign "
            f"allocator's pointer on a guess is how this crashed.")

    if shares_heap:
        t = torch.empty(tensor.shape, dtype=nbx_dtype_to_torch(tensor._dtype),
                        device=device)
        if tensor._nbytes > 0:
            DeviceAllocator.memcpy(t.data_ptr(), tensor.data_ptr(),
                                   tensor._nbytes)
        return t

    # Separate heaps: the values travel through the host. On unified memory
    # that is a RAM copy, not a bus transfer.
    #
    # Through RAW BYTES, not through `.numpy()`: numpy has no bfloat16, so
    # `_DTYPE_TYPESTR` maps it to '<V2' (opaque 2-byte void) and a bf16
    # tensor would arrive as something torch cannot take — on the dtype most
    # of the hub runs in. `torch.frombuffer` knows bfloat16, so the bytes
    # are typed once, correctly, by the same table that named the device.
    import ctypes
    torch_dtype = nbx_dtype_to_torch(tensor._dtype)
    if tensor._nbytes == 0:
        return torch.empty(tuple(tensor.shape), dtype=torch_dtype,
                           device=device)
    host = tensor.contiguous()
    if host._device != 'cpu':
        host = host.to_cpu()
    raw = bytearray(ctypes.string_at(host.data_ptr(),
                                     host.numel() * dtype_size(host._dtype)))
    flat = torch.frombuffer(raw, dtype=torch_dtype)
    return flat.reshape(tuple(tensor.shape)).to(device=device)


