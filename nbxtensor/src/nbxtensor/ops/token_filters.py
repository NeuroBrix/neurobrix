"""Two filters a sampler applies to a decode step's logits on the device, the transformers LogitsProcessor semantics:

- `penalize_repeats(logits, seen_ids, penalty)`: RepetitionPenaltyLogitsProcessor — every id already seen has its
  logit divided by `penalty` when positive, multiplied by it otherwise; in place, computed in float32.
- `outside_nucleus(cumulative, top_p)`: TopPLogitsWarper's mask over probabilities sorted descending — True where the
  running sum before the token already passed `top_p`, so the most probable token is always kept (a peaked row whose
  first probability exceeds `top_p` keeps it, instead of masking everything and drawing from NaN).

Both read the row on the host (the ids are few and the row one decode step's) and write the result back to the
device the input is on. Torch-free.
"""
from __future__ import annotations

import numpy as np

from nbxtensor.random.seeded_draw import apply_repetition_penalty

__all__ = ["penalize_repeats", "outside_nucleus"]


def penalize_repeats(logits, seen_ids, penalty: float):
    """`logits` (its last axis the vocabulary; read in float32) with every id of `seen_ids` (a tensor of ids or a
    sequence) penalised, in place; returned."""
    if seen_ids is None or penalty == 1.0:
        return logits
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    ids = seen_ids.numpy().reshape(-1) if isinstance(seen_ids, NBXTensor) else np.asarray(list(seen_ids))
    row = logits.to(NBXDtype.float32).numpy().reshape(-1)
    apply_repetition_penalty(row, ids.tolist(), penalty)
    logits.copy_(NBXTensor.from_numpy(row.reshape(logits.shape), device=logits.device_idx))
    return logits


def outside_nucleus(cumulative, top_p: float):
    """A bool tensor shaped like `cumulative` (running sums of probabilities sorted descending along the last axis):
    True where the sum BEFORE the token exceeds `top_p` — the first token never."""
    from nbxtensor.ops import elementwise_wrappers as E
    from nbxtensor.tensor import NBXTensor
    past = E.gt(cumulative, top_p).numpy()
    shifted = np.zeros_like(past)
    shifted[..., 1:] = past[..., :-1]
    return NBXTensor.from_numpy(shifted, device=cumulative.device_idx)
