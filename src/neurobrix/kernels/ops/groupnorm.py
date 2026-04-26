"""Group normalization — pure @triton.jit kernel.

Two-pass loop over hidden_size (= group_size * HW) in BLOCK_SIZE chunks,
so the kernel scales to arbitrary spatial dimensions. Replaces the
original FlagGems extraction whose single-tile design crashed with
`numel exceeds triton maximum tensor numel (1048576)` once HW reached
1024×1024 (PixArt VAE 1024px output).

Algorithm: standard sum-of-squares variance per (batch, group) program.
For each program (one (batch, group) pair):
  1. Loop over hidden_size in BLOCK_SIZE chunks → accumulate sum and
     sum-of-squares → mean = s / N, var = ss/N - mean².
  2. rstd = 1 / sqrt(var + eps).
  3. Loop a second time → normalize → multiply by per-channel weight,
     add per-channel bias, store.

Forward only (inference). The wrapper guarantees BLOCK_SIZE ≤
triton.next_power_of_2(hidden_size_per_chunk_target) and ≤ 16384, well
under Triton's 2^20 numel ceiling on every architecture.
"""

import triton
import triton.language as tl


@triton.jit
def group_norm_forward_kernel(
    X,                              # input pointer, shape (N, C, HW)
    Y,                              # output pointer, shape (N, C, HW)
    W,                              # weight pointer, shape (C,)
    B,                              # bias pointer, shape (C,)
    Mean,                           # mean output, shape (N * num_groups,)
    Rstd,                           # rstd output, shape (N * num_groups,)
    group_size,                     # channels per group
    C,                              # total channels
    HW,                             # spatial size = H * W (or H * W * D)
    num_groups,                     # total groups
    eps,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,       # chunk size along hidden_size, ≤ 16384
):
    """One program handles one (batch, group) pair. Spatial dimension is
    walked in BLOCK_SIZE chunks so the per-program tile stays bounded
    regardless of input HW."""
    pid = tl.program_id(0)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups

    # Channel range of this group (each channel has HW elements contiguous
    # in memory; the group's elements are group_size*HW elements starting
    # at base_offset).
    chan_start = group_idx * group_size
    base_offset = batch_idx * C * HW + chan_start * HW
    hidden = group_size * HW

    block_range = tl.arange(0, BLOCK_SIZE)

    # ── Pass 1: sum and sum-of-squares over the full hidden dim ─────────
    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, hidden, BLOCK_SIZE):
        idx = off + block_range
        mask = idx < hidden
        x = tl.load(X + base_offset + idx, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x)
        ss += tl.sum(x * x)

    inv_n = 1.0 / hidden
    mean = s * inv_n
    var = ss * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

    # ── Pass 2: normalize → optional affine → store ──────────────────────
    # Each chunk falls inside a single channel because chunks walk the
    # contiguous (channel, HW) layout. We compute the channel for each
    # element of the block to fetch the right (W, B) entries.
    for off in tl.range(0, hidden, BLOCK_SIZE):
        idx = off + block_range
        mask = idx < hidden
        x = tl.load(X + base_offset + idx, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd

        if scale_by_weight or add_bias:
            # Channel index within the GROUP, then add the group's chan_start.
            chan_in_group = idx // HW
            chan_global = chan_start + chan_in_group
            chan_mask = mask & (chan_in_group < group_size)
            if scale_by_weight:
                w = tl.load(W + chan_global, mask=chan_mask, other=1.0).to(tl.float32)
                x_hat = x_hat * w
            if add_bias:
                b = tl.load(B + chan_global, mask=chan_mask, other=0.0).to(tl.float32)
                x_hat = x_hat + b

        tl.store(Y + base_offset + idx, x_hat, mask=mask)
