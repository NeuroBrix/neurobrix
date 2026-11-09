"""An embedding model's pooling, on the host: the flow's last hidden state [B, T, H] (both engines hand it to the host
as an array) reduced to one vector per sequence, as the model's own pooling declares it (point 9b; the engine reads the
kind from the container, never chooses it). Host math, numpy, torch-free: the pooled output is [B, H], read once per
request.

Kinds, bounded to what embedding models ship (sentence-transformers' 1_Pooling modes):
  mean — the average over the positions the attention mask keeps (padding excluded);
  cls  — position 0;
  last — each sequence's last kept position (left or right padding alike).
A kind outside these, a mask that keeps no position of a sequence, or shapes that disagree: refused by name.
"""
from __future__ import annotations

import numpy as np

POOLING_KINDS = ("mean", "cls", "last")


def pool_hidden(hidden: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    """[B, T, H] hidden states and [B, T] attention mask (1 kept, 0 padding) -> [B, H] in float32."""
    if kind not in POOLING_KINDS:
        raise ValueError(f"pool_hidden: pooling {kind!r} is not one of {POOLING_KINDS}")
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError(f"pool_hidden: hidden {hidden.shape} and mask {mask.shape} are not [B, T, H] and [B, T]")
    keep = mask.astype(bool)
    counts = keep.sum(axis=1)
    if (counts == 0).any():
        raise ValueError(f"pool_hidden: sequence(s) {np.flatnonzero(counts == 0).tolist()} keep no position")
    h = hidden.astype(np.float32, copy=False)
    if kind == "cls":
        return h[:, 0, :].copy()
    if kind == "last":
        last = hidden.shape[1] - 1 - np.argmax(keep[:, ::-1], axis=1)       # the last kept position, any padding side
        return h[np.arange(h.shape[0]), last, :]
    # one contraction over the positions, no [B, T, H] temporary: 2.4 ms for [8, 512, 2048] fp32 (17 ms with one)
    return np.einsum("bth,bt->bh", h, keep.astype(np.float32)) / counts[:, None].astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Each row scaled to unit L2 norm; a zero row is refused by name (its direction does not exist)."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("l2_normalize: a zero vector has no direction to normalise")
    return x / norms
