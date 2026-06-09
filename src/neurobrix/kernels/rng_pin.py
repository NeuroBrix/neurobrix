"""Gated cross-engine deterministic RNG for noise pinning.

`NBX_FORCE_RAND_SEED=<int>` makes every `rand`/`randn`/`*_like` op draw from a
single `numpy.RandomState`, so the PyTorch and Triton engines produce IDENTICAL
noise when they execute the same DAG (same op order + shapes). Two uses:

1. Diagnostic — pin the noise so an op-by-op cross-engine diff is meaningful
   past the first random op (otherwise every downstream op "diverges" purely
   because each engine drew different noise, masking real kernel bugs).
2. Shared-seeded determinism — the production way to make a stochastic vocoder
   (e.g. chatterbox s3gen CFM + NSF/SineGen source) reproducible and equal
   across all four execution modes, same discipline as the openaudio dual_ar /
   VibeVoice diffusion seeded noise.

Zero torch: this module returns numpy arrays. The Triton side wraps them as
`NBXTensor.from_numpy`; the PyTorch side wraps them as `torch.from_numpy`. The
draw order/shape must match across engines for the arrays to be bit-identical;
both run the same component DAG, so they do.
"""

import os
import numpy as np

_STATE = None
_SEED_USED = "<unset>"


def pinned_seed():
    """Return the configured seed (int) or None when pinning is disabled."""
    v = os.environ.get("NBX_FORCE_RAND_SEED")
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return 1234


def _state():
    global _STATE, _SEED_USED
    s = pinned_seed()
    if _STATE is None or _SEED_USED != s:
        _STATE = np.random.RandomState(s if s is not None else 1234)
        _SEED_USED = s
    return _STATE


def _shape(shape):
    return tuple(int(d) for d in shape)


def pinned_normal(shape):
    """Standard-normal N(0,1) float32 array of the given shape."""
    return _state().standard_normal(size=_shape(shape)).astype(np.float32)


def pinned_uniform(shape):
    """Uniform [0,1) float32 array of the given shape."""
    return _state().random_sample(size=_shape(shape)).astype(np.float32)
