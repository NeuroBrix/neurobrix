"""NBX_DECODE_BOUND — universal bounded-decode harness (diagnostics).

When `NBX_DECODE_BOUND=N` is set, every autoregressive / TTS / audio_llm /
dual_ar / encoder_decoder decode loop hard-caps its step count to N. This lets
op-by-op triton-vs-oracle diffs (the 4-mode method) run on a 5-10 token window
in seconds instead of the full 2048-token generation (~27 min op-by-op). It does
NOT change generation semantics when unset (returns max_tokens unchanged).

Pure-Python (os only) — zero torch — so it is R33-safe to import from both
`core/flow/` and `triton/flow/`. Gated, default-off, zero runtime impact unset.
"""
import os


def decode_bound(max_tokens):
    """Return `min(max_tokens, NBX_DECODE_BOUND)` when the env var is a positive
    int, else `max_tokens` unchanged. Tolerant of `None`/non-int inputs."""
    bound = os.environ.get("NBX_DECODE_BOUND")
    if not bound:
        return max_tokens
    try:
        n = int(bound)
    except (ValueError, TypeError):
        return max_tokens
    if n <= 0:
        return max_tokens
    if max_tokens is None:
        return n
    try:
        return min(int(max_tokens), n)
    except (ValueError, TypeError):
        return n
