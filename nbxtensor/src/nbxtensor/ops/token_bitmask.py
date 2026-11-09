"""A token bitmask applied to a decode step's logits — the mechanism of constrained decoding (reste_apres_064 point 9a).

A grammar engine (the serving layer's constraint: a JSON schema, a tool-call envelope) says which tokens may come next as
a bitmask: int32 words, `ceil(vocab / 32)` per sequence, bit `v % 32` of word `v // 32` set when token `v` is allowed —
the layout llguidance and XGrammar both fill. The kernel writes -inf into every disallowed logit, in place, before the
sampler reads them (a masked token then has probability 0 under any temperature, top-k or top-p). fp32 logits only: the
generator reads them at fp32 before sampling (triton/generator.py). Zero torch (R33)."""
from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

from nbxtensor.backend.launcher import install as _install_launcher
from nbxtensor.dtype.nbx_dtype import NBXDtype

# every kernel[grid] of this module goes through the library's launcher, as ops/wrappers.py: on Metal a first launch
# without it reaches the backend's own driver and refuses the NBXTensor arguments (the gathered KV copies, 2026-11-09)
_install_launcher()

@triton.jit
def _token_bitmask_kernel(logits_ptr, mask_ptr, vocab, words, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    v = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = v < vocab
    word = tl.load(mask_ptr + row * words + v // 32, mask=inside, other=-1)
    allowed = ((word >> (v % 32).to(tl.int32)) & 1) != 0
    tl.store(logits_ptr + row * vocab + v, float("-inf"), mask=inside & ~allowed)


def bitmask_words(vocab: int) -> int:
    return -(-int(vocab) // 32)


def apply_token_bitmask(logits, bitmask: np.ndarray) -> None:
    """`logits` (NBXTensor fp32 [rows, vocab], contiguous): every token whose bit is clear in `bitmask` (int32
    [rows, ceil(vocab / 32)]) set to -inf, in place."""
    from nbxtensor.backend.runtime import set_device as launch_on
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops.elementwise_launch import ew_meta
    from nbxtensor.tensor import NBXTensor
    if logits._dtype != NBXDtype.float32 or logits.ndim != 2:
        raise ValueError(f"apply_token_bitmask: fp32 [rows, vocab] logits, got {logits._dtype} {tuple(logits.shape)}")
    rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    bitmask = np.ascontiguousarray(bitmask, dtype=np.int32)
    if bitmask.shape != (rows, bitmask_words(vocab)):
        raise ValueError(f"apply_token_bitmask: a bitmask of {(rows, bitmask_words(vocab))} int32 words for "
                         f"{(rows, vocab)} logits, got {bitmask.shape}")
    if not bitmask.any():
        raise RuntimeError("ZERO FALLBACK: the constraint allows no token at this step (an empty bitmask)")
    DeviceAllocator.set_device(logits._device_idx)      # the mask on the logits' own card (a head on another card)
    mask = NBXTensor.from_numpy(bitmask)
    launch_on(logits)
    m = ew_meta(vocab)                                    # the element-wise launch's certified tile and warps
    _token_bitmask_kernel[(rows, triton.cdiv(vocab, m["BLOCK_SIZE"]))](logits, mask, vocab, bitmask.shape[1],
                                                                       BLOCK=m["BLOCK_SIZE"], num_warps=m["num_warps"])


def token_bitmask(allowed, vocab: int) -> np.ndarray:
    """The [1, ceil(vocab / 32)] int32 bitmask allowing exactly the token ids in `allowed` (a reference and a helper)."""
    words = np.zeros(bitmask_words(vocab) * 32, dtype=np.uint8)
    words[np.asarray(list(allowed), dtype=np.int64)] = 1
    packed = np.packbits(words.reshape(-1, 32)[:, ::-1], axis=1).view(">u4").astype(np.uint32)
    return packed.view(np.int32).reshape(1, -1)
