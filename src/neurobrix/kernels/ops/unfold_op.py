"""unfold_backward (overlap-add) — pure @triton.jit kernel.

aten::unfold_backward is the gradient of `Tensor.unfold`: it scatter-adds the
unfolded frames back into the original signal, so overlapping positions are
summed. This is exactly the iSTFT overlap-add (frames [.., N_frames, size] with
hop `step` → signal [.., L]).

For output position i: out[i] = Σ_{f} grad[f, j] where j = i - f*step and
0 <= j < size. The covering frames are f = i//step - fo for a small fo range
(~size/step frames overlap a given position).
"""

import triton
import triton.language as tl


@triton.jit
def unfold_backward_1d_kernel(
    grad_ptr,        # (B, N_frames, size) contiguous
    out_ptr,         # (B, L) contiguous
    N_frames, size, step, L,
    MAX_OVERLAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    offs = pid_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # output positions
    mask = offs < L
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    base = pid_b * (N_frames * size)
    for fo in range(MAX_OVERLAP):
        f = (offs // step) - fo
        j = offs - f * step
        valid = (f >= 0) & (f < N_frames) & (j >= 0) & (j < size) & mask
        val = tl.load(grad_ptr + base + f * size + j, mask=valid, other=0.0)
        acc += tl.where(valid, val, 0.0)
    tl.store(out_ptr + pid_b * L + offs, acc, mask=mask)
