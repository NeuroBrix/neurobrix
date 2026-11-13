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
        why = (f"query and key lengths differ ({seq_q} vs {seq_k}), where `is_causal` "
               f"would align them top-left")
    elif not _is_causal_content(mask):
        why = "its content is not the causal (lower-triangular) pattern"
    if why is not None:
        raise RuntimeError(
            f"{op_name}: the attention mask {tuple(mask.shape)} is shorter than the "
            f"sequence it must cover (query {seq_q}, key {seq_k}) — its extent froze at "
            f"the build-time length, a symbolic-coverage defect of the container — and "
            f"{why}. Refusing to extend it: re-propagate the mask symbolically in the container builder.")


def _is_causal_content(mask) -> bool:
    """Every leading [q, k] slice of `mask` is the lower-triangular pattern (bool: True on and below the diagonal;
    additive: exactly 0 there and -inf / the dtype's minimum above). Read on the HOST, one distinct slice at a time (a
    broadcast axis, stride 0, read once): the proof holds no card memory. On the card it held the mask's float copy,
    its boolean masks and the triangle's indices — 8 GiB for a served 32 768-row chunk of Qwen3-30B-A3B-Thinking on a
    V100-32GB, every request out of memory (2026-11-13 00:35); the Triton engine reads it on the host too
    (`nbxtensor.memory.paged_kv.require_causal_mask`)."""
    tri = _t.ones(tuple(mask.shape[-2:]), dtype=_t.bool).tril()
    lead = [range(n) if mask.stride(i) else range(1) for i, n in enumerate(mask.shape[:-2])]
    return all(_slice_is_causal(mask[idx].to("cpu"), tri) for idx in itertools.product(*lead))


def _slice_is_causal(m, tri) -> bool:
    """One host [q, k] slice against the lower triangle `tri`."""
    if m.dtype == _t.bool:
        return bool((m == tri).all())
    f = m.float()
    blocked = _t.isneginf(f) | (f <= _t.finfo(m.dtype).min)
    return bool((blocked == ~tri).all()) and bool(((f == 0) == tri).all())
