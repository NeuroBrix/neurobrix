"""The concat and stack kernels select one input's metadata per program in an
if/elif chain; Triton specialises a Python int argument as int32 when it fits and
int64 when it does not, so ONE input past 2^31 elements beside a small one made the
branches disagree and the kernel refused to compile:

    Mismatched type for total_elements between then block (int64) and else block (int32)

Measured 2026-09-25 on Open-Sora-v2's VAE decode at 129 frames 256x256
(validation_outputs/rebuilds_2026_09_25/Open-Sora-v2/f129_256x256.log, aten.cat::8).
The int32 index wrap past two billion elements (register 58/59) is the same family.

What this test does if the code is wrong: it compiles each kernel through Triton's
warmup with one count above 2^31 and the others below, on a CUDA device with only
1-element tensors — the compile itself fails (seen red before the cast, 2026-09-25).

Run: python -m pytest tests/unit/kernels/test_a_concat_branch_agrees_on_its_count_type.py
"""
import pytest
import torch

triton = pytest.importorskip("triton")

from neurobrix.kernels.ops.cat_op import cat_copy_kernel_4
from neurobrix.kernels.ops.stack_op import stack_copy_kernel

BIG = 2 ** 31 + 4096
SMALL = 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device compiles the kernel")
@pytest.mark.parametrize("kernel", [cat_copy_kernel_4, stack_copy_kernel])
def test_one_input_past_two_billion_elements_still_compiles(kernel):
    t = [torch.zeros(1, device="cuda") for _ in range(5)]
    # out, a, b, c, d, dim_size_in x4, dim_size_out, dim_prod_post, dim_offset x4, total_elements x4
    args = [t[0], t[1], t[2], t[3], t[4], 8, 8, 8, 8, 32, 1024, 0, 8, 16, 24, BIG, SMALL, SMALL, SMALL]
    kernel.warmup(*args, BLOCK_X=1024, grid=(1, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device runs the kernel")
def test_one_element_inputs_take_the_constexpr_form_and_still_run():
    """Triton specialises a scalar equal to 1 as a constexpr Python int: a cast
    placed inside the branches broke on it ('int' object has no attribute 'to',
    Flex.1-alpha's new arm, 2026-09-25). The selection must accept every form."""
    import numpy as np
    from neurobrix.kernels.nbx_tensor import NBXTensor
    g = lambda x: NBXTensor.from_numpy(x).to("cuda")
    one, two = np.array([1.5], dtype=np.float32), np.array([-2.0], dtype=np.float32)
    assert np.array_equal(NBXTensor.cat([g(one), g(two)], dim=0).to("cpu").numpy(), np.concatenate([one, two]))
    assert np.array_equal(NBXTensor.stack([g(one), g(two)], dim=0).to("cpu").numpy(), np.stack([one, two], 0))
