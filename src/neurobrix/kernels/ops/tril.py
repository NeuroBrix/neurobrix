"""Lower triangular — pure @triton.jit kernels. Adapted from FlagGems triu.

tril_kernel: 2D version, iterates over N blocks per row.
tril_batch_kernel: batched version, handles N-D tensors flattened to (batch, M*N).

The only difference from triu: condition is >= (keep below diagonal) vs <= (keep above).
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["diagonal"])
def tril_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask

        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(row + diagonal >= cols, x, 0.0)
        tl.store(Y + cols, y, mask=mask)


@triton.jit(do_not_specialize=["diagonal"])
def tril_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask and mn_mask
    x = tl.load(X + cols, mask, other=0.0)
    m = cols // N
    n = cols % N
    y = tl.where(m + diagonal >= n, x, 0.0)
    tl.store(Y + cols, y, mask=mask)
