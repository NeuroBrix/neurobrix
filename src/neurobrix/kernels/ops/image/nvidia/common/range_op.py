"""
Range - Create tensor with evenly spaced values
ATen: aten::arange
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _range_kernel(
    out_ptr,
    start,
    delta,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = start + offsets.to(tl.float32) * delta
    tl.store(out_ptr + offsets, values, mask=mask)

def _range_impl(
    start: float,
    end: float,
    step: float = 1.0,
    dtype: torch.dtype = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Range implementation."""
    if dtype is None:
        dtype = torch.float32
    return torch.arange(start, end, step, dtype=dtype, device=device)

def range_kernel(
    start: float = 0,
    end: float = None,
    step: float = 1.0,
    dtype: torch.dtype = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Range/Arange operation.
    """
    return _range_impl(start, end, step, dtype, device)
