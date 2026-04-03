"""NLL Loss forward kernel — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.

NLL Loss: l_n = -w_{y_n} * x_{n, y_n}

Reduction modes:
  0 = none: per-sample loss
  1 = mean: weighted mean
  2 = sum:  weighted sum

The kernel handles ignore_index to skip certain targets.
"""

import triton
import triton.language as tl


@triton.jit
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """NLL Loss forward for 1D/2D input [N, C].

    Args:
        inp_ptr: Log-probability input [N, C].
        tgt_ptr: Target class indices [N].
        wgt_ptr: Per-class weights [C] or None.
        out_ptr: Output. Shape depends on reduction mode.
        ignore_index: Target index to ignore.
        N: Batch size.
        C: Number of classes.
        reduction: 0=none, 1=mean, 2=sum.
        BLOCK_N: Block size for batch dimension.
    """
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = (tgt != ignore_index) & mask_n

    # Load per-class weight for target, or 1.0 if no weights
    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # Gather input at target index: inp[n, tgt[n]]
    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")
        tl.atomic_add(out_ptr + 1, total_wgt, sem="relaxed")
        tl.atomic_add(out_ptr + 2, 1, sem="release")
        counter = tl.load(out_ptr + 2)
        if counter == tl.num_programs(0):
            total_out = tl.load(out_ptr)
            total_wgt = tl.load(out_ptr + 1)
            tl.store(out_ptr + 3, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")


@triton.jit
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """NLL Loss backward for 1D/2D input [N, C].

    Computes gradient w.r.t. input: grad_input[n, tgt[n]] = -w * grad_output / total_weight.
    """
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = (tgt != ignore_index) & mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad = tl.load(out_grad_ptr + offsets_n, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)
