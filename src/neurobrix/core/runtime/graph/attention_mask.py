"""Attention-mask rules shared by every torch-side attention site (compiled SDPA,
the KV-cache interceptor): one reading of a mask whose extent froze at the trace
length, never a guess."""
from __future__ import annotations

import torch as _t


def require_causal_frozen_mask(mask, seq_q: int, seq_k: int, op_name: str,
                               align_lengths: bool = True) -> None:
    """A mask SHORTER than the sequence it must cover froze at the trace length.

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
    else:
        tri = _t.ones(mq, mk, dtype=_t.bool, device=mask.device).tril()
        if mask.dtype == _t.bool:
            ok = bool((mask == tri).all())
        else:
            m = mask.float()
            blocked = _t.isneginf(m) | (m <= _t.finfo(mask.dtype).min)
            ok = bool(((~blocked) == tri).all()) and bool((m[..., tri] == 0).all()) \
                if m.dim() >= 2 else False
        if not ok:
            why = "its content is not the causal (lower-triangular) pattern"
    if why is not None:
        raise RuntimeError(
            f"{op_name}: the attention mask {tuple(mask.shape)} is shorter than the "
            f"sequence it must cover (query {seq_q}, key {seq_k}) — its extent froze at "
            f"the trace length, a symbolic-coverage defect of the container — and "
            f"{why}. Refusing to extend it: re-propagate the mask symbolically in Forge.")
