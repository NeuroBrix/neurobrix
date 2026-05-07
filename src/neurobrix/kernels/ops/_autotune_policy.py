"""NBX_DISABLE_AUTOTUNE — diagnostic env var that forces all autotuned
matmul-class kernels to use a single fixed config.

Purpose (P-SANA-4KPX-RUNTIME 2026-05-07): isolate autotune as the
cause of shape-dependent numerical divergence at Sana 4Kpx scale.
When `NBX_DISABLE_AUTOTUNE=1`, the autotune `configs=[...]` list is
filtered down to a single `Config` known to be Volta-viable, so the
autotuner has no degrees of freedom and the same bit-pattern of
accumulation runs at every shape. If running with this flag yields a
coherent PNG that the autotune-enabled run does not, autotune is
confirmed as the cause.

This is **not for production** — autotune is required to reach
cuBLAS perf parity per CLAUDE.md autotune doctrine. Use only as a
diagnostic / temporary workaround until a buggy config is removed
from the search space or its kernel is fixed.

Universal across hardware (R23): the fixed config is `BLOCK_M=64,
BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2` for matmul-class
kernels — Volta-viable per CLAUDE.md autotune policy and within
sm_80+ SMEM headroom too.
"""

import os

NBX_DISABLE_AUTOTUNE = os.environ.get("NBX_DISABLE_AUTOTUNE", "0") == "1"


def maybe_pin_single(configs, predicate):
    """If `NBX_DISABLE_AUTOTUNE=1`, return a single-element list with
    the FIRST config matching `predicate`. Otherwise return `configs`
    unchanged.

    `predicate` is a callable `Config -> bool`. The first match in the
    `configs` list is selected, so callers should prefer ordering with
    the desired Volta-safe target near the top (or use a unique
    predicate).
    """
    if not NBX_DISABLE_AUTOTUNE:
        return configs
    for c in configs:
        if predicate(c):
            return [c]
    # Fallback: keep a single arbitrary config if the predicate finds
    # nothing (e.g. depthwise has different field names). Caller must
    # adjust predicate per kernel.
    return configs[:1]


def is_matmul_pinned(c):
    """Match the matmul/bmm/addmm Volta-safe target:
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2.
    """
    kw = c.kwargs
    return (kw.get('BLOCK_M') == 64 and kw.get('BLOCK_N') == 64
            and kw.get('BLOCK_K') == 32
            and c.num_warps == 4 and c.num_stages == 2)


def is_conv2d_pinned(c):
    """Match conv2d Volta-safe target:
    BLOCK_SIZE_BHW=64, BLOCK_SIZE_OUTF=64, BLOCK_SIZE_INF=32, warps=4, stages=2.
    """
    kw = c.kwargs
    return (kw.get('BLOCK_SIZE_BHW') == 64
            and kw.get('BLOCK_SIZE_OUTF') == 64
            and kw.get('BLOCK_SIZE_INF') == 32
            and c.num_warps == 4 and c.num_stages == 2)


def is_depthwise_conv2d_pinned(c):
    """Match depthwise conv2d Volta-safe target — depthwise uses
    different config field names (BLOCK_C, BLOCK_HW). Pick a balanced
    mid-range default if no explicit pin matches.
    """
    kw = c.kwargs
    return kw.get('BLOCK_C') == 32 and kw.get('BLOCK_HW') == 64
