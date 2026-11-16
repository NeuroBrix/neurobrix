"""The gated delta rule's operands for its tests, drawn in float32 on the host: q, k [B, L, Hk, dk], v [B, L, Hv, dv],
g [B, L, Hv] (<= 0, `-A * softplus(a)` as the layer forms it), beta [B, L, Hv] in (0, 1), and a non-zero state
[B, Hv, dk, dv] when asked. `decay_max` bounds A: a mild decay keeps a starting state alive over the length, so a test
from a non-zero state reads it."""
import numpy as np


def draw(B, L, Hk, Hv, dk, dv, seed, nonzero_state, decay_max=1.0):
    r = np.random.default_rng(seed)
    q = r.standard_normal((B, L, Hk, dk)).astype(np.float32)
    k = r.standard_normal((B, L, Hk, dk)).astype(np.float32)
    v = r.standard_normal((B, L, Hv, dv)).astype(np.float32)
    A = r.uniform(0.01, decay_max, Hv).astype(np.float32)
    g = (-A * np.log1p(np.exp(r.standard_normal((B, L, Hv))))).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-r.standard_normal((B, L, Hv))))).astype(np.float32)
    state = (0.5 * r.standard_normal((B, Hv, dk, dv))).astype(np.float32) if nonzero_state else None
    return q, k, v, g, beta, state
