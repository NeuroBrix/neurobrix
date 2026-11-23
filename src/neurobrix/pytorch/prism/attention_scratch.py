"""The scores a device's compiled attention writes — priced by Prism, cut by the compiled engine, one definition.

On CUDA PyTorch runs `scaled_dot_product_attention` on its flash or memory-efficient kernels: the full
[B, H, Tq, Tk] scores are never written, and Prism prices the memory-efficient workspace
(`memory_estimator.estimate_op_workspace_bytes`). On a device whose vendor file states
`memory.compiled_sdpa_full_scores: true` (MPS: "just the math implementation, fused", pytorch/pytorch#179294) the scores
ARE written: Allegro at 79 200 tokens added ~13 GB in one second on an M4 Pro (2026-10-15). There, the compiled engine
runs the attention in query chunks whose fp32 scores stay within `compiled_sdpa_scores_device_fraction` of the device's
memory (exact: each query row's softmax is its own), and Prism prices exactly what a chunk writes.

ONE budget: Prism computes it from the profile (`PrismSolver._attention_budget`), prices with it and states it on every
component it places on such a device (`ComponentAllocation.attention_scores_budget_bytes`); the engine cuts with the
figure the plan states, never with a reading of its own. The chunk sizes are the TilingEngine's (`tiling_sizes.
sdpa_chunk_plan`), the row block the vendor's `memory.sdpa_math_min_chunk_rows`. Torch-free (Prism imports it).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

_MIB = 1024 * 1024


def vendor_memory_of(brand, architecture) -> Dict[str, Any]:
    """The `memory:` section of a vendor file, by the device's brand and architecture."""
    from neurobrix.core.config.loader import get_vendor_config
    brand = getattr(brand, "value", brand)
    return dict((get_vendor_config(brand, architecture) or {}).get("memory") or {})


def vendor_memory(device) -> Dict[str, Any]:
    """The `memory:` section of the vendor file of a profile device."""
    return vendor_memory_of(getattr(device, "brand", None), device.architecture)


def writes_full_scores(memory: Dict[str, Any]) -> bool:
    """Whether the vendor file states that the compiled attention writes its full scores. A vendor file that does not
    state it is refused by name (no default)."""
    if "compiled_sdpa_full_scores" not in memory:
        raise KeyError("ZERO FALLBACK: the vendor file states no memory.compiled_sdpa_full_scores — whether this "
                       "device's compiled attention writes its full scores is a stated fact, never a default")
    return bool(memory["compiled_sdpa_full_scores"])


def scores_budget(memory: Dict[str, Any], device_memory_mb) -> Optional[int]:
    """The bytes of scores a chunk may write on this device, or None where the compiled attention writes none."""
    if not writes_full_scores(memory):
        return None
    if device_memory_mb is None:
        raise ValueError("ZERO FALLBACK: the device states no memory: the attention scores budget is a fraction of it")
    return int(float(device_memory_mb) * _MIB * float(memory["compiled_sdpa_scores_device_fraction"]))


def chunks_in_flight(memory: Dict[str, Any]) -> int:
    """How many chunks the device holds at once: the engine waits for it every that many chunks (the vendor's
    `memory.compiled_sdpa_chunks_in_flight`, refused by name when a device writing full scores does not state it)."""
    if "compiled_sdpa_chunks_in_flight" not in memory:
        raise KeyError("ZERO FALLBACK: the vendor file states compiled_sdpa_full_scores and no "
                       "memory.compiled_sdpa_chunks_in_flight — how many chunks its device holds at once is stated")
    n = int(memory["compiled_sdpa_chunks_in_flight"])
    if n < 1:
        raise ValueError(f"ZERO FALLBACK: memory.compiled_sdpa_chunks_in_flight is {n}; at least one chunk runs")
    return n


def chunk_plan(memory: Dict[str, Any], budget: int, batch: int, heads: int, tq: int, tk: int):
    """(slices per chunk, query rows per chunk), or None when the whole attention's scores fit the budget
    (`tiling_sizes.sdpa_chunk_plan`). A chunk is sized at the budget shared by the chunks in flight. Refused by name
    when one slice of one row block exceeds that share."""
    from nbxtensor.ops.tiling_sizes import sdpa_chunk_plan, sdpa_scores_bytes
    if sdpa_scores_bytes(batch, heads, tq, tk) <= budget:
        return None
    block = int(memory["sdpa_math_min_chunk_rows"])
    share = int(budget) // chunks_in_flight(memory)
    group, rows = sdpa_chunk_plan(share, batch, heads, tq, tk, block)
    if rows == 0:
        raise ValueError(f"ZERO FALLBACK: one head's {block}-row block of an attention over {tk} keys writes "
                         f"{block * tk * 4 / 2**20:.0f} MiB of scores, over the {share / 2**20:.0f} MiB share of "
                         f"each chunk in flight")
    return group, rows


def scratch_bytes(memory: Dict[str, Any], budget: Optional[int], batch: int, heads: int, tq: int, tk: int) -> int:
    """The scores the compiled attention holds while it runs: none where the device writes none; the whole scores
    where they fit the budget; one chunk's otherwise."""
    from nbxtensor.ops.tiling_sizes import sdpa_scores_bytes
    if budget is None:
        return 0
    plan = chunk_plan(memory, budget, batch, heads, tq, tk)
    if plan is None:
        return sdpa_scores_bytes(batch, heads, tq, tk)
    group, rows = plan
    return chunks_in_flight(memory) * sdpa_scores_bytes(1, group, rows, tk)
