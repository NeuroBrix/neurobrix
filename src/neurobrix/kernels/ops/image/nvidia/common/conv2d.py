# Conv2d - 2D Convolution (Pure Triton)
# Type: Triton Kernel
# NeuroBrix - NVIDIA Common (All architectures)

import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_direct_kernel(
    x_ptr, w_ptr, out_ptr,
    bias_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr,  # Batch size block
    BLOCK_H: tl.constexpr,  # Height block
    BLOCK_W: tl.constexpr,  # Width block
):
    """
    Direct Convolution Kernel.
    Grid: (C_out, N, H_out_blocks * W_out_blocks)
    One program instance computes a tile of output pixels for one channel.
    """
    pid_c = tl.program_id(0) # Output channel index
    pid_n = tl.program_id(1) # Batch index
    pid_hw = tl.program_id(2) # Spatial block index
    
    # Spatial blocking
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    pid_h = pid_hw // num_w_blocks
    pid_w = pid_hw % num_w_blocks
    
    # Offsets for this block
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Masks
    mask_h = off_h < H_out
    mask_w = off_w < W_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    
    # Accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels
    for c in range(C_in):
        # Iterate over kernel dimensions
        for kh in range(K_H):
            for kw in range(K_W):
                # Input coordinates
                h_in = off_h[:, None] * stride_h - pad_h + kh * dil_h
                w_in = off_w[None, :] * stride_w - pad_w + kw * dil_w
                
                # Check bounds
                valid_h = (h_in >= 0) & (h_in < H_in)
                valid_w = (w_in >= 0) & (w_in < W_in)
                valid_hw = valid_h & valid_w & mask_hw
                
                # Pointers
                # Input: x[n, c, h_in, w_in]
                x_off = (pid_n * stride_xn + 
                         c * stride_xc + 
                         h_in * stride_xh + 
                         w_in * stride_xw)
                         
                # Weight: w[c_out, c_in, kh, kw]
                w_off = (pid_c * stride_wco + 
                         c * stride_wci + 
                         kh * stride_wkh + 
                         kw * stride_wkw)
                
                # Load values
                val_x = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)
                val_w = tl.load(w_ptr + w_off) # Weight is scalar for this kernel tap
                
                acc += val_x * val_w
                
    # Add bias if present
    if bias_ptr is not None:
        b = tl.load(bias_ptr + pid_c)
        acc += b
        
    # Store output
    # Output: out[n, c_out, h_out, w_out]
    out_off = (pid_n * stride_on + 
               pid_c * stride_oc + 
               off_h[:, None] * stride_oh + 
               off_w[None, :] * stride_ow)
               
    tl.store(out_ptr + out_off, acc, mask=mask_hw)

# Registration for discovery (Adaptor V2 uses module introspection)
