"""KV-append kernel family — constant-tuple decode cache writes (B3).

The decode step's ONLY structurally varying launches after B1+B2'
(census: 144 = 48 layers x {k-append, v-append, mask-validity write})
were the __setitem__ scatters whose destination offsets ride HOST
state (current_len). This family makes them constant-tuple: the write
POSITION is read from a DEVICE counter inside the kernel (the
device-scalar pattern's third application — timestep, then GEMV
scalars, now the KV position), and one fused kernel writes k, v and
zeroes the pad mask at that position. A one-thread increment kernel
advances the counter in a SECOND launch (fusing it raced: program 0's
increment could land before a late program's position load — the
separate kernel orders it by stream serialization; both tuples stay
constant). One counter per layer.

Launch tuple: (k_src, v_src, k_buf, v_buf, mask_buf, pos_ptr,
strides..., constexprs) — every pointer is a fixed-address buffer
(slab-stable src post-B1, pre-allocated bufs), so the tuple is
CONSTANT across steps by construction. Decode contract: new_len == 1
(one position appended per step).
"""

import triton
import triton.language as tl


@triton.jit
def kv_append_kernel(
    k_src_ptr, v_src_ptr,           # [B, H_kv, 1, D_k] / [.., D_v] (contig)
    k_buf_ptr, v_buf_ptr,           # [B, H_kv, MAX, D_k] / [.., D_v]
    mask_buf_ptr,                   # [1, 1, 1, MAX] additive (-inf/0)
    pos_ptr,                        # [1] int32 device counter (write pos)
    stride_kb_h, stride_kb_l,       # k_buf strides (head, len) in elems
    stride_vb_h, stride_vb_l,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch per layer per decode step (grid = KV heads): writes
    the new K/V row at the device-counter position and zeroes the pad
    mask there. The counter advances in kv_pos_inc_kernel, launched
    AFTER on the same stream (serialized — no intra-grid race on the
    position load)."""
    pid = tl.program_id(0)          # one program per KV head
    pos = tl.load(pos_ptr)

    offs_k = tl.arange(0, BLOCK)
    mk = offs_k < D_K
    k_vals = tl.load(k_src_ptr + pid * D_K + offs_k, mask=mk, other=0.0)
    tl.store(k_buf_ptr + pid * stride_kb_h + pos * stride_kb_l + offs_k,
             k_vals, mask=mk)

    mv = offs_k < D_V
    v_vals = tl.load(v_src_ptr + pid * D_V + offs_k, mask=mv, other=0.0)
    tl.store(v_buf_ptr + pid * stride_vb_h + pos * stride_vb_l + offs_k,
             v_vals, mask=mv)

    if pid == 0:
        tl.store(mask_buf_ptr + pos, 0.0)


@triton.jit
def kv_pos_inc_kernel(pos_ptr):
    """Advance the device position counter by one (stream-ordered
    after the append; constant launch tuple)."""
    tl.store(pos_ptr, tl.load(pos_ptr) + 1)
