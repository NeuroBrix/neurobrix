import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    softmax_out = numerator / denominator[:, None]
    return softmax_out

def benchmark(func: callable, 
              input_tensor: torch.Tensor, 
              iterations: int = 100
              ) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warm-up to avoid cold start
    func(input_tensor)
    
    start.record()
    for _ in range(iterations):
        func(input_tensor)
    end.record()
    
    # Wait for all events to complete
    torch.cuda.synchronize()
    
    elapsed_time_ms = start.elapsed_time(end) / iterations
    print(f"Average execution time over {iterations} iterations: {elapsed_time_ms:.3f} ms")
    return elapsed_time_ms


@triton.jit
def _softmax_fwd_kernel(
    output_ptr: tl.tensor,
    stride_output_row: tl.int32,
    input_ptr: tl.tensor,
    stride_input_row: tl.int32,
    num_cols: tl.int32,
    block_size: tl.constexpr,
) -> None:
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + (row_idx * stride_input_row) 
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols

    row = tl.load(input_pointers, mask= row_mask, other=float("-inf"))

    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    output_ptr_row = output_ptr + (row_idx * stride_output_row)
    output_pointers = output_ptr_row + col_offsets
    tl.store(output_pointers, softmax_out, mask=row_mask)

    
def softmax(x: torch.Tensor) -> torch.Tensor:
    """ triton implementation of softmax
    """
    rows, cols = x.shape
    assert x.dim() == 2, f"check tensors are 2D"    
    block_size = triton.next_power_of_2(cols)
    num_warps = 4
    if block_size > 2**11 -1:
        num_warps = 2 * num_warps
    elif block_size > 4095:
        num_warps = 4 * num_warps

    grid = (rows, )
    softmax_out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        softmax_out,
        softmax_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size = block_size,
        num_warps = num_warps
    )
    return softmax_out



if __name__ == "__main__":
    # Run correctness check with smaller sample
    sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype=torch.float32, device="cuda")
    ref_out = F.softmax(sample, dim=1)
    eager_out = naive_softmax(sample)
    triton_out = softmax(sample)

    print(f"{ref_out=}")
    print(f"{eager_out=}")
    print(f"{triton_out=}")
    assert torch.allclose(ref_out, triton_out, atol=1e-6), "The naive softmax does not match PyTorch's softmax!"

    large_sample = torch.randn(10000, 1000, dtype=torch.float32, device="cuda")  # 10k samples, each with 1k features

    # Benchmark PyTorch's softmax and naive softmax
    print("Benchmarking PyTorch's softmax:")
    benchmark(lambda x: F.softmax(x, dim=1), large_sample)

    print("Benchmarking naive softmax:")
    benchmark(naive_softmax, large_sample)

    print("Benchmarking Triton softmax:")
    benchmark(softmax, large_sample)
