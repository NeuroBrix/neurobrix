"""What a plan holds in HOST memory, per engine — priced from the plan Prism chose, never from a model's name.

The owner's rule (2026-09-27 14:27): the host footprint of a run is computed by Prism from the plan it
chose for this container on this hardware profile; a host ledger reserves what the plan says, and a
cell's measured peak (`host_peak_rss` in its matrix row) is the PROOF of the figure, never its source.

The holdings below are the runtime's own rules, read in both engines (2026-09-27) — each line names
where the rule lives, so a change there is a change to price here:

  steady — held for as long as the component is loaded:
    compiled, a component COMPUTED on the host (its device is "cpu"): its weights in float32 plus its
        activations and overhead (graph_executor.py: the host path runs fp32; dtype/engine.py);
    compiled, a component whose shards Prism mapped to "cpu" (zero3): those weights at the plan's
        dtype, pinned, for the whole run (strategies/zero3.py `_pin_cpu_weights`);
    triton, a component with shards on "cpu" (zero3, or a "cpu" staging placement): its BLOCK weights
        pinned; everything else and all compute stay on the card (triton/weight_loader.py
        `_load_to_pinned_cpu`, graph_executor.py re-routes compute to a card);
  transient — while a component loads:
    compiled: the component passes through the host whole — read at its stored width, and pinned at
        the plan's width for the non-blocking upload (core/io/weight_loader.py `_load_with_pinned_dma`:
        load_file to cpu, `_convert_weights_dtype`, `_transfer_with_pinned_memory`). The pinned copies
        are not given back: PyTorch's caching host allocator keeps a freed pinned block on its free
        list and never calls cudaFreeHost (pytorch#134332; PyTorch DevLog 2026-08-09, "Pinned memory:
        what it is for, and why nobody gives it back"), so the worker count bounds the pace, not the
        bytes;
    triton: one tensor at a time, as read and as converted (triton/weight_loader.py);
  resident — what the planning process already holds when Prism prices: the interpreter, the CLI and
    the parsed container (NBXContainer.load keeps every component's graph and profile, 1.5-4.8x their
    JSON bytes, measured 2026-09-27 — not a constant, so it is read, not priced), passed in measured;
  base — what the engine's device work adds on top (its modules, device context, the compute libraries
    a run's first ops load): a MEASURED value of this machine, carried by the hardware profile per
    engine (`cpu.runtime_base_mb`); a profile without it prices no base and says so.

Under lazy loading one component is resident at a time, so steady is the largest component's; eager
loading holds them all. The compiled load is priced the same way, as a bound: the loader synchronises once
per component (core/io/weight_loader.py `load_component`), so a component's pinned blocks can all be
outstanding until then, and afterwards they return to the allocator's free list, where the next component
reuses them only where sizes fit. An eager plan's sum is the most its passes can hold (none reused), a
lazy plan's largest pass what one component alone holds; the matrix's peaks lie between the two for
eager plans and at or under the largest pass for lazy ones (66 cells, 2026-09-27). layer_streaming re-reads a segment from disk per run (no host cache) and KV
caches live on the card, except under cpu_execution, whose components are priced as host compute.

Pure arithmetic, no torch: Prism is torch-free (R33).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

#: Copies of one tensor the triton loader holds: as read and as converted (triton/weight_loader.py).
TRITON_COPIES_PER_TENSOR = 2
#: Bytes per element the compiled host path computes in.
HOST_COMPUTE_BYTES = 4

ENGINES = ("compiled", "triton")


def engine_of(mode: str) -> str:
    """The engine a run mode executes on: triton and triton_sequential are the Triton engine."""
    return "triton" if str(mode).startswith("triton") else "compiled"


def _on_host(device) -> bool:
    return str(device).startswith("cpu")


def host_footprint(plan, key_sizes: Mapping[str, Mapping[str, int]],
                   shard_sizes: Mapping[str, Mapping[str, int]], engine: str,
                   base_mb: Optional[int], dtype_bytes: Mapping[str, int],
                   is_block_key, stored_dtypes: Optional[Mapping[str, set]] = None,
                   resident_bytes: int = 0) -> Dict:
    """The host bytes `plan` holds on `engine`: {total, resident, base, steady (+ per component), transient}.

    key_sizes    {component: {weight key: stored bytes}}  (the weights index)
    shard_sizes  {component: {shard name: bytes}}         (the container's shards)
    stored_dtypes {component: {stored floating dtypes}}  (the weights index): what a load reads at, beside
                 the plan's dtype it pins at
    base_mb      what this engine's device work adds on this machine, or None (not measured)
    resident_bytes what the planning process holds when it prices (the caller measures it)
    """
    if engine not in ENGINES:
        raise ValueError(f"ZERO FALLBACK: no host rules for engine {engine!r} (known: {ENGINES})")
    steady: Dict[str, int] = {}
    for name, alloc in plan.components.items():
        mem = plan.component_memory.get(name)
        keys = key_sizes.get(name) or {}
        shards = shard_sizes.get(name) or {}
        host_shards = {s for s, d in (alloc.shard_map or {}).items() if _on_host(d)}
        held = 0
        if engine == "compiled":
            if _on_host(alloc.device):
                if mem is None:
                    raise ValueError(f"ZERO FALLBACK: {name} computes on the host and has no priced memory")
                width = dtype_bytes[str(alloc.dtype)]
                held = mem.weight_bytes * HOST_COMPUTE_BYTES // width + mem.activation_bytes + mem.overhead_bytes
            elif host_shards and mem is not None:
                total = sum(shards.values())
                on_host = sum(shards.get(s, 0) for s in host_shards)
                held = mem.weight_bytes * on_host // total if total else mem.weight_bytes
        else:
            if host_shards or _on_host(alloc.device):
                held = sum(n for k, n in keys.items() if is_block_key(k))
        if held:
            steady[name] = int(held)
    steady_bytes = (max(steady.values(), default=0) if plan.loading_mode == "lazy"
                    else sum(steady.values()))
    if engine == "compiled":
        def _passes_through(name, stored_bytes):
            alloc = plan.components.get(name)
            widths = [dtype_bytes[d] for d in ((stored_dtypes or {}).get(name) or ()) if d in dtype_bytes]
            pinned = stored_bytes
            if alloc is not None and widths and str(alloc.dtype) in dtype_bytes:
                pinned = stored_bytes * dtype_bytes[str(alloc.dtype)] // min(widths)
            return stored_bytes + pinned
        passes = [_passes_through(name, sum(sh.values())) for name, sh in shard_sizes.items() if sh]
        transient = sum(passes) if plan.loading_mode == "eager" else max(passes, default=0)
    else:
        transient = TRITON_COPIES_PER_TENSOR * max(
            (n for ks in key_sizes.values() for n in ks.values()), default=0)
    base = int(base_mb) << 20 if base_mb is not None else 0
    return {"engine": engine, "total_bytes": int(resident_bytes) + base + steady_bytes + transient,
            "resident_bytes": int(resident_bytes), "base_bytes": base, "base_measured": base_mb is not None,
            "steady_bytes": steady_bytes, "steady": steady, "transient_bytes": transient,
            "loading": plan.loading_mode}


def resident_bytes_now() -> int:
    """This process's resident memory so far (its peak RSS: at planning time, the container just parsed),
    read from the OS — Linux reports KiB, macOS bytes."""
    import resource
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) << 10


def summary(hf: Mapping) -> str:
    """The one line a run and `--explain-plan` print for the plan's host figure — the line the regression
    matrix reads into a cell's row as the estimate its measured peak judges."""
    base = f"{hf['base_bytes'] / 2**20:.0f} MB" if hf["base_measured"] else "UNMEASURED on this profile"
    held = ", ".join(f"{k} {v / 2**20:.0f} MB" for k, v in hf["steady"].items())
    return (f"{hf['total_bytes'] / 2**20:.0f} MB on the {hf['engine']} engine = resident "
            f"{hf.get('resident_bytes', 0) / 2**20:.0f} MB + engine {base}"
            f" + held {hf['steady_bytes'] / 2**20:.0f} MB ({hf['loading']})"
            f" + loading {hf['transient_bytes'] / 2**20:.0f} MB" + (f"  [held: {held}]" if held else ""))
