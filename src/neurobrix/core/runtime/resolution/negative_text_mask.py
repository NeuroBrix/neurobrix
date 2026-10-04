"""The unconditional branch's text mask — the mask of the embedding it masks.

Classifier-free guidance cross-attends two text sequences, [unconditional, conditional], each under its own
padding mask. The encoder handler's finalization may change a sequence's length after encoding: a Sana prompt
is encoded behind its instruction prefix and cut back to [BOS] + its last N-1 positions; a T5 sequence is
re-padded with zeros up to the model's length. The finalizer returns the mask cut or padded the same way, and
the conditional path has always stored that one.

The unconditional path kept the tokenizer's mask instead, at the encoder graph's input length, beside an
embedding the finalizer had just shortened; the CFG engines then met a mask of another length than its
embedding and replaced it with ones. The unconditional branch attended every padding position where the
vendor attends one. Measured 2026-10-04 on SANA-Video_2B_720p_diffusers (256x640, 4 steps, the vendor's
SanaVideoPipeline in fp32 fed the engine's noise): the engine's batched mask held 310 ones over the two rows,
the vendor's 11; the first divergent op was block 0's cross-attention, and the step-0 guided output stood at
rel 0.557 from the vendor's (cos 0.8825). The vendor transformer given an all-ones unconditional row
reproduced the engine to rel 4.4e-4.

Two rules, shared by both engines (R30) and torch-free (R33) — they read `.shape` and nothing else:

    finalized_mask        the finalizer's own mask when it returned one, else the tokenizer's
    negative_mask_for     the recorded negative mask, refused by name when its length is not its embedding's;
                          None when the flow recorded none (the caller then attends every position)
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def finalized_mask(tokenizer_mask: Any, finalized: Optional[Mapping[str, Any]]) -> Any:
    """The mask that belongs to a finalized embedding."""
    mask = finalized.get("attention_mask") if finalized else None
    return tokenizer_mask if mask is None else mask


def negative_mask_for(neg_mask: Any, neg_hidden: Any, encoder_comp: str) -> Any:
    """The unconditional branch's mask as the flow recorded it, or None when it recorded none.

    A mask whose last extent is not the embedding's sequence extent masks another sequence. It is refused,
    never replaced: a substitute of ones attends the padding, and nothing downstream can see that it did."""
    if neg_mask is None:
        return None
    mask_len, seq_len = int(neg_mask.shape[-1]), int(neg_hidden.shape[1])
    if mask_len != seq_len:
        raise RuntimeError(
            f"ZERO FALLBACK: '{encoder_comp}.negative_attention_mask' has {mask_len} positions, "
            f"'{encoder_comp}.negative_hidden_state' has {seq_len}: the unconditional mask is not the mask "
            f"of the unconditional embedding (a finalization changed the embedding's length and its mask "
            f"was not stored with it).")
    return neg_mask
