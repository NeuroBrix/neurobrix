"""Attention-mask rules shared by every torch-side attention site (compiled SDPA,
the KV-cache interceptor): one reading of a mask whose extent froze at the build
length, never a guess."""
from __future__ import annotations

import itertools

import torch as _t


def require_causal_frozen_mask(mask, seq_q: int, seq_k: int, op_name: str,
                               align_lengths: bool = True) -> None:
    """A mask SHORTER than the sequence it must cover froze at the build length.

    The only reading that extends it without guessing is the causal one, and
    only when the mask's own content proves it: every leading slice is the
    lower-triangular pattern at the mask's extent (bool: True on and below the
    diagonal; additive: exactly 0 there and -inf / the dtype's minimum above),
    and — when `align_lengths` — the query and key lengths agree (torch's
    `is_causal` aligns them top-left). A KV-cache decode step passes
    `align_lengths=False`: its queries sit at the END of the cached keys, where
    the causal reading is "attend to every cached key". Anything else — a padding row, a block-diagonal or windowed
    mask, an additive bias — has no defined value beyond its extent, and the
    old branch silently replaced it by causal attention. Refused by name.
    """
    mq, mk = int(mask.shape[-2]), int(mask.shape[-1])
    why = None
    if mq != mk:
        why = f"its last two axes differ ({mq} x {mk}), so it is not a square causal pattern"
    elif align_lengths and seq_q != seq_k:
        why = f"query and key lengths differ ({seq_q} vs {seq_k}), which `is_causal` aligns top-left"
    elif not _is_causal_content(mask):
        why = "its content is not the causal (lower-triangular) pattern"
    if why is not None:
        raise RuntimeError(
            f"{op_name}: the attention mask {tuple(mask.shape)} is shorter than the sequence it must cover (query "
            f"{seq_q}, key {seq_k}) — its extent froze at the build-time length, a symbolic-coverage defect of the "
            f"container — and {why}. Refusing to extend it: re-propagate the mask symbolically in the container builder.")


#: the proof's own boolean temporaries alive at once per element of a band (its triangle, a comparison, its negation, the
#: combined test): the bands are sized so they never hold more than the mask they read
_PROOF_BYTES_PER_ELEMENT = 4


def _is_causal_content(mask) -> bool:
    """Every leading [q, k] slice of `mask` is the lower-triangular pattern (bool: True on and below the diagonal;
    additive: exactly 0 there and -inf / the dtype's minimum above), read where the mask lives, one distinct slice at a
    time (a broadcast axis, stride 0, read once), in bands of rows whose temporaries never exceed the mask's own bytes
    — inside the chunk's priced attention reserve (rows x keys, keys >= rows), before the attention spends it. The
    whole-slice proof held the mask's float copy, its boolean masks and the triangle's int64 indices: 8 GiB for a
    served 32 768-row chunk of Qwen3-30B-A3B-Thinking on a V100-32GB, every request out of memory (2026-11-13 00:35);
    read on the host instead, it cost a copy and a CPU pass at EVERY layer (a view per layer: Qwen3-Coder-30B-A3B
    served --compiled 1.5-2.3x slower, 2026-11-13 04:16)."""
    band = max(1, int(mask.shape[-2]) * mask.element_size() // _PROOF_BYTES_PER_ELEMENT)
    lead = [range(n) if mask.stride(i) else range(1) for i, n in enumerate(mask.shape[:-2])]
    return all(_slice_is_causal(mask[idx], r0, band)
               for idx in itertools.product(*lead) for r0 in range(0, int(mask.shape[-2]), band))


def _slice_is_causal(m, r0: int, band: int) -> bool:
    """Rows r0 .. r0 + band of one [q, k] slice against the lower triangle, built for those rows only."""
    s = m[r0:r0 + band]
    tri = _t.arange(s.shape[-1], device=s.device)[None, :] <= _t.arange(r0, r0 + len(s), device=s.device)[:, None]
    if s.dtype == _t.bool:
        return bool(_t.equal(s, tri))
    blocked = _t.isneginf(s) | (s <= _t.finfo(s.dtype).min)
    return bool(_t.equal(blocked, ~tri)) and bool(_t.equal(s == 0, tri))
