"""
CrossEntropy Kernel - NVIDIA
Fused LogSoftmax + NLLLoss

Source: Liger-Kernel (adapted)
Tier: common
"""
import torch
import triton
import triton.language as tl

from neurobrix.kernels.registry import register_kernel


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    out_ptr,
    stride_logits_row,
    stride_logits_col,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes CrossEntropy loss for a batch of logits and targets.
    """
    row_idx = tl.program_id(0)
    
    # Pointers
    logits_row_ptr = logits_ptr + row_idx * stride_logits_row
    
    # Load row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    logits = tl.load(logits_row_ptr + col_offsets * stride_logits_col, mask=mask, other=-float('inf')).to(tl.float32)
    
    # 1. Max subtraction (stability)
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    
    # 2. LogSumExp
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp)
    
    # 3. Get target class logit
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Only load the specific target logit if valid
    # Since we can't index easily with variable, we recompute or load
    # Here we assume target_idx is valid
    
    # Ideally we want: target_logit = logits[target_idx]
    # But in Triton block load, we loaded everything.
    # We can mask-select:
    
    target_mask = col_offsets == target_idx
    # tl.sum with mask to extract single value
    target_logit = tl.sum(tl.where(target_mask, logits, 0.0), axis=0)
    
    # 4. CrossEntropy = log_sum_exp - target_logit
    # (Since we subtracted max_val from both, it cancels out:
    #  log(sum(exp(x-m))) - (x[t]-m) = log(sum(exp(x))*exp(-m)) - x[t] + m
    #  = log(sum(exp(x))) - m - x[t] + m = log(sum(exp(x))) - x[t])
    
    loss = log_sum_exp - target_logit
    
    # Store
    tl.store(out_ptr + row_idx, loss)


# =============================================================================
# WRAPPER
# =============================================================================

def _cross_entropy_impl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Logits: [batch, num_classes]
    # Targets: [batch]
    assert logits.ndim == 2, "Logits must be 2D"
    assert targets.ndim == 1, "Targets must be 1D"
    assert logits.shape[0] == targets.shape[0], "Batch size mismatch"
    
    n_rows, n_cols = logits.shape
    
    # Output: [batch] (reduction='none') - we'll mean it later if needed
    # But usually PyTorch F.cross_entropy returns scalar mean by default
    # Let's return the vector of losses for flexibility
    losses = torch.empty((n_rows,), dtype=torch.float32, device=logits.device)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    grid = (n_rows,)
    
    _cross_entropy_kernel[grid](
        logits,
        targets,
        losses,
        logits.stride(0),
        logits.stride(1),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Default reduction: mean
    return losses.mean()


# =============================================================================
# REGISTRATION
# =============================================================================

@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="cross_entropy")
def cross_entropy_kernel(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropy kernel compatible with all NVIDIA architectures."""
    return _cross_entropy_impl(logits, targets)
