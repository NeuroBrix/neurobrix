"""A torch tensor reaching the Triton branch arrives as the values it shows, whatever its dtype, strides or device,
through ONE helper (`kernels/torch_interop.torch_to_nbx`, over `nbxtensor.interop.from_dlpack`).

Before (audit 2026-11-12, TENSOR_IN_ENGINE_A rows 93/95): four copies of the conversion (cfg/engine `_ensure_nbx`,
flow/iterative_process `_to_nbx`, flow/audio `_torch_to_nbx`, flow/forward_pass `_ensure_nbx_tensor`). Measured on the
M4 Pro against main's four: a CPU bfloat16 tensor fails in all four ("Got unsupported ScalarType BFloat16": numpy has no
bfloat16); an MPS tensor fails in two (taken for a CPU one, `.numpy()` refuses it); a CPU channels-last view is right
in all four (numpy copies by strides). On a CUDA card `cfg/engine` wrapped a non-contiguous tensor's pointer with
contiguous strides, reading it in buffer order (the audit; the Dell's proof).

What this test does if the code is wrong: a bfloat16 tensor fails or changes bits, a permuted view arrives in buffer
order, an MPS tensor is not read correctly, a tensor needing grad is refused, a pass-through value is converted, or one
of the four flows grows a conversion of its own again (a fifth copy) — red.

Run: python -m pytest tests/unit/triton/test_a_torch_tensor_enters_the_triton_branch_as_the_values_it_shows.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _device():
    from nbxtensor.memory.allocator import DeviceAllocator
    try:
        DeviceAllocator.set_device(0)
    except Exception as e:                                      # pragma: no cover
        pytest.skip(f"no device: {e}")


def test_a_bfloat16_tensor_arrives_bit_for_bit():
    from neurobrix.kernels.torch_interop import torch_to_nbx
    _device()
    t = torch.linspace(-3, 3, 64, dtype=torch.float32).to(torch.bfloat16).reshape(8, 8)
    got = torch_to_nbx(t)
    assert got.nbx_dtype.name == "bfloat16"
    assert np.array_equal(got.to("float32").numpy(), t.float().numpy())


def test_a_channels_last_view_arrives_in_the_order_it_shows():
    from neurobrix.kernels.torch_interop import torch_to_nbx
    _device()
    hwc = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    chw = hwc.permute(2, 0, 1)                                  # what run.py's --input-image hands a VAE encoder
    assert not chw.is_contiguous()
    assert np.array_equal(torch_to_nbx(chw).numpy(), chw.numpy())


def test_an_mps_tensor_arrives_through_the_host():
    from neurobrix.kernels.torch_interop import torch_to_nbx
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    _device()
    t = torch.arange(16, dtype=torch.float16, device="mps").reshape(4, 4) * 0.5
    assert np.array_equal(torch_to_nbx(t).numpy(), t.cpu().numpy())


def test_a_tensor_that_needs_grad_is_taken_as_its_values():
    from neurobrix.kernels.torch_interop import torch_to_nbx
    _device()
    t = torch.ones(3, requires_grad=True) * 2
    assert np.array_equal(torch_to_nbx(t).numpy(), np.full(3, 2.0, dtype=np.float32))


def test_what_is_not_a_foreign_tensor_passes_through():
    from neurobrix.kernels.torch_interop import torch_to_nbx
    from nbxtensor.tensor import NBXTensor
    _device()
    n = NBXTensor.from_numpy(np.ones(2, dtype=np.float32))
    assert torch_to_nbx(None) is None and torch_to_nbx(n) is n and torch_to_nbx(7) == 7


def test_no_flow_carries_a_conversion_of_its_own():
    """The four flows call the one helper; a converter of their own growing back is a fifth copy."""
    from neurobrix.triton.cfg import engine
    from neurobrix.triton.flow import audio, forward_pass, iterative_process
    for module, name in ((engine, "_ensure_nbx"), (audio, "_torch_to_nbx"), (forward_pass, "_ensure_nbx_tensor"),
                         (iterative_process, "_to_nbx")):
        assert not hasattr(module, name), f"{module.__name__}.{name} is back"

def test_a_strided_view_leaves_for_torch_as_the_values_it_shows():
    """The reverse crossing (`nbx_to_torch`): a transposed NBXTensor view arrives in torch in the order it shows. On a
    backend whose torch shares the device heap it was copied from its buffer as-is (the audit, row 93)."""
    from neurobrix.kernels.torch_interop import nbx_to_torch
    from nbxtensor.tensor import NBXTensor
    _device()
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = NBXTensor.from_numpy(a).t()
    assert np.array_equal(nbx_to_torch(view).cpu().numpy(), a.T)
