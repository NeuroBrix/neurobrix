# Fused Attention
# ===============
# This is an adaptaion of Triton implementation of the Flash Attention algorithm for mixed precision
# (see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf;
# Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)


import torch
import triton
import triton.language as tl


_SUPPORTED_SIZES = {16, 32, 64, 128}


def _get_configs():
    configs = []
    for block_m in [64, 128, 256]:
        for block_n in [32, 64, 128]:
            for num_stage in [3, 4, 5, 6, 7, 8]:
                for num_warps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_warps=num_warps,
                            num_stages=num_stage,
                        )
                    )
    return configs


@triton.autotune(
    configs=_get_configs(),
    key=["N_CTX", "H", "Z"],
)
@triton.heuristics({"EVEN_CTX": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0})
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    qkv_scale_ptr,
    out_scale_ptr,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    EVEN_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop

    qkv_scale = tl.load(qkv_scale_ptr)
    qk_scale = qkv_scale * qkv_scale * sm_scale * 1.44269504

    # load q: it will stay in SRAM throughout
    if EVEN_CTX:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k --
        if EVEN_CTX:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        # -- compute qk ---
        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        qk = tl.dot(q, k, allow_tf32=False, out_dtype=tl.int32)
        qk_fp32 = qk * qk_scale

        # -- compute scaling constant ---
        m_ij = tl.maximum(m_i, tl.max(qk_fp32, 1))
        p = tl.math.exp2(qk_fp32 - m_ij[:, None])
        # -- scale and update acc --
        alpha = tl.math.exp2(m_i - m_ij)
        m_i = m_ij
        # -- load v --
        if EVEN_CTX:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        v = (v * qkv_scale).to(tl.bfloat16)
        acc *= alpha[:, None]
        acc += tl.dot(
            p.to(tl.bfloat16),
            v,
            allow_tf32=True,
        )
        l_i = l_i * alpha + tl.sum(p, 1)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_scale = tl.load(out_scale_ptr)
    acc = tl.math.llrint(acc / (l_i[:, None] * out_scale)).to(tl.int8)
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if EVEN_CTX:
        tl.store(O_block_ptr, acc)
    else:
        tl.store(O_block_ptr, acc, boundary_check=(0,))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        qkv_scale,
        out_scale,
    ):
        # only support for Ampere and newer
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in _SUPPORTED_SIZES
        o = torch.empty_like(q)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(q.shape[2], META["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )
        if isinstance(qkv_scale, float):
            qkv_scale = torch.tensor(qkv_scale, device=q.device)
            out_scale = torch.tensor(out_scale, device=q.device)
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            qkv_scale,
            out_scale,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_DMODEL=Lk,
        )
        return o
