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


def triton_routes(profile) -> list:
    """(the math route's scores budget, its row block, its chunk ceiling, the matrix unit) of EACH card the plan may run
    on — the route the Triton wrappers take there (`wrappers._sdpa_math_scores_budget_bytes_for`: the executing
    device's budget, `tiling_sizes.sdpa_math_scores_budget`), each read from THIS device's vendor file (never the
    planning process's own profile): on a rack of 16 and 32 GB cards an attention between the two budgets runs the math
    route on the larger one."""
    from nbxtensor.ops import tiling_sizes as _ts
    from nbxtensor.ops._configs import matrix_unit
    from neurobrix.core.config.loader import vendor_config_for_device
    routes = []
    for device in profile.devices:
        memory = vendor_memory(device)
        brand = getattr(getattr(device, "brand", None), "value", getattr(device, "brand", None))
        unit = matrix_unit(vendor_config_for_device(brand, device.architecture,
                                                    getattr(device, "compute_capability", None),
                                                    model=getattr(device, "name", None)))
        routes.append((_ts.sdpa_math_scores_budget(int(memory.get("sdpa_math_max_scores_bytes") or 0),
                                                   float(memory.get("sdpa_math_scores_device_fraction") or 0.0),
                                                   device.memory_mb),
                       int(memory.get("sdpa_math_min_chunk_rows") or 0), int(memory.get("sdpa_math_max_chunks") or 0),
                       unit))
    return routes


def triton_scratch(profile, dtype: str):
    """(b, h, tq, tk, D=, Dv=, Hk=, mask=, budget=) -> what the Triton engines' attention holds beside its output while
    it runs, priced at the dearest of the cards' routes (`triton_routes`; `budget`: the math route's budget a plan states
    for this attention, the wrapper's `scores_budget_bytes`, in place of each card's): the math route's buffers as the
    wrapper takes them (`attention_wrappers.math_attention_cost` / `chunked_math_attention_cost`: the scores' product,
    its scale, a mask made dense, the softmax, the casts, the output's product), less the output the walk already
    counts; the flash route writes no scores. It carries `scores_bounds(D, budget)`, each card's math-route bound at
    head dim D (`launch_keys.sdpa_math_scores_bound`). None with no device. `dtype`: the run dtype of the component
    (`_run_dtype`, the DtypeEngine's resolution), an unknown one refused by name."""
    if profile is None or not profile.devices:
        return None
    from nbxtensor.ops import launch_keys as _lk
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    routes = triton_routes(profile)
    unit_dtype = NBXDtype[dtype]
    native_bf16 = bool(profile.has_native_bf16)         # the run's flag is the profile's (`set_hardware_profile`)
    takes = {}                                          # (device route, head dim) -> the unit runs its flash

    def scratch(b, h, tq, tk, D, Dv, Hk=None, mask=False, budget=None):
        from nbxtensor.ops import attention_wrappers as _aw
        from nbxtensor.ops._cost import Spec
        hk = int(Hk or h)
        q, k = Spec.contiguous((b, h, tq, D), dtype), Spec.contiguous((b, hk, tk, D), dtype)
        v, m = Spec.contiguous((b, hk, tk, Dv), dtype), (Spec.contiguous((b, 1, tq, tk), dtype) if mask else None)
        worst = 0
        for i, (own, min_rows, max_chunks, unit) in enumerate(routes):
            if (i, D) not in takes:
                takes[(i, D)] = _lk.unit_flash_takes(D, unit_dtype, unit_dtype, unit_dtype, unit)
            route, rows = _lk.sdpa_route(b, h, tq, tk, D, Dv, own if budget is None else int(budget), min_rows,
                                         max_chunks, unit_flash=takes[(i, D)])
            if route == "flash":
                continue
            facts = dict(native_bf16=native_bf16, force_accum=False, contract_half=False, unit=unit)
            cost = (_aw.math_attention_cost(q, k, v, m, False, **facts) if route == "math"
                    else _aw.chunked_math_attention_cost(q, k, v, rows, m, False, **facts))
            worst = max(worst, cost.peak_bytes - sum(x.nbytes for x in cost.outputs))
        return worst
    scratch.scores_bounds = lambda D, budget=None: [_lk.sdpa_math_scores_bound(own if budget is None else int(budget),
                                                                             int(D)) for own, _r, _c, _u in routes]
    return scratch
