"""Copy — pure @triton.jit kernel.

Ported from FlagGems copy (Apache-2.0 license).
Copies elements from src to dst, with optional dtype conversion
handled by Triton's automatic casting on store.
"""

import triton
import triton.language as tl

@triton.jit
def copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SATURATE_F16: tl.constexpr = False,
):
    """Copy src to dst. Dtype conversion happens automatically via Triton's
    store when src and dst pointer types differ.

    SATURATE_F16: protected float16 downcast — finite values beyond the
    fp16 range clamp to ±65504 instead of overflowing to ±inf. This
    mirrors the compiled DtypeEngine's hardware-protected conversion
    contract (bf16/fp32 sentinel fills like finfo.min must survive a
    V100 fp16 remap as finite extremes; an additive attention mask that
    overflows to -inf poisons softmax rows — the DeepSeek-V2 wrong-token
    incident). NaN and true ±inf inputs pass through unchanged.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    data = tl.load(src_ptr + offset, mask=mask)
    if SATURATE_F16:
        f = data.to(tl.float32)
        is_finite = (f == f) & (f != float("inf")) & (f != float("-inf"))
        clamped = tl.minimum(tl.maximum(f, -65504.0), 65504.0)
        data = tl.where(is_finite, clamped, f)
    tl.store(dst_ptr + offset, data, mask=mask)
