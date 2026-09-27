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
    compiled: up to `io_workers()` shards in flight, each as read, converted and pinned copies
        (core/io/weight_loader.py: load to cpu, dtype conversion, pin_memory, non_blocking copy);
    triton: one tensor at a time, as read and as converted (triton/weight_loader.py);
  base — the runtime's own process before any weight (interpreter, libraries, backend context): a
    MEASURED value of this machine, carried by the hardware profile per engine
    (`cpu.runtime_base_mb`); a profile without it prices no base and says so.

Under lazy loading one component is resident at a time, so steady is the largest component's; eager
loading holds them all. layer_streaming re-reads a segment from disk per run (no host cache) and KV
caches live on the card, except under cpu_execution, whose components are priced as host compute.

Pure arithmetic, no torch: Prism is torch-free (R33).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

#: Copies of one shard the compiled loader holds while that shard is in flight: as read and pinned for
#: the non-blocking upload, plus a converted copy when the stored dtype is not the plan's
#: (core/io/weight_loader.py `_load_weight_file`: load_file to cpu, `_convert_weights_dtype` returns
#: early when the dtype already matches, `_transfer_with_pinned_memory`). The loader's own steps,
#: judged by the matrix's measured peaks — not a tuning value.
COMPILED_COPIES_PER_SHARD = 2
COMPILED_CONVERSION_COPIES = 1
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
                   base_mb: Optional[int], io_workers: int, dtype_bytes: Mapping[str, int],
                   is_block_key, stored_dtypes: Optional[Mapping[str, set]] = None) -> Dict:
    """The host bytes `plan` holds on `engine`: {total, base, steady (+ per component), transient}.

    key_sizes    {component: {weight key: stored bytes}}  (the weights index)
    shard_sizes  {component: {shard name: bytes}}         (the container's shards)
    stored_dtypes {component: {stored floating dtypes}}  (the weights index): a component loaded at
                 another dtype holds a converted copy per shard in flight
    base_mb      the runtime's measured base on this machine for this engine, or None (not measured)
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
        def _copies(name):
            alloc = plan.components.get(name)
            stored = (stored_dtypes or {}).get(name) or set()
            converts = alloc is not None and bool(stored) and any(d != str(alloc.dtype) for d in stored)
            return COMPILED_COPIES_PER_SHARD + (COMPILED_CONVERSION_COPIES if converts else 0)
        transient = max((min(io_workers, len(sh)) * _copies(name) * max(sh.values())
                         for name, sh in shard_sizes.items() if sh), default=0)
    else:
        transient = TRITON_COPIES_PER_TENSOR * max(
            (n for ks in key_sizes.values() for n in ks.values()), default=0)
    base = int(base_mb) << 20 if base_mb is not None else 0
    return {"engine": engine, "total_bytes": base + steady_bytes + transient,
            "base_bytes": base, "base_measured": base_mb is not None,
            "steady_bytes": steady_bytes, "steady": steady, "transient_bytes": transient,
            "loading": plan.loading_mode}


def summary(hf: Mapping) -> str:
    """The one line a run and `--explain-plan` print for the plan's host figure — the line the regression
    matrix reads into a cell's row as the estimate its measured peak judges."""
    base = f"{hf['base_bytes'] / 2**20:.0f} MB" if hf["base_measured"] else "UNMEASURED on this profile"
    held = ", ".join(f"{k} {v / 2**20:.0f} MB" for k, v in hf["steady"].items())
    return (f"{hf['total_bytes'] / 2**20:.0f} MB on the {hf['engine']} engine = base {base}"
            f" + held {hf['steady_bytes'] / 2**20:.0f} MB ({hf['loading']})"
            f" + loading {hf['transient_bytes'] / 2**20:.0f} MB" + (f"  [held: {held}]" if held else ""))
