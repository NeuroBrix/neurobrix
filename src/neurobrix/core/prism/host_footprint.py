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
  output — what the output boundary holds while it saves: the largest graph output of the plan's
    components at the request, times what the family's save path holds per element
    (core/runtime/output_dispatch.py `host_bytes_per_output_element`: two float32 copies and the
    integer image for png/mp4, the waveform twice and PCM_16 for wav, nothing for text);
  base — what the engine's device work adds on top (its modules, device context, the compute libraries
    a run's first ops load): a MEASURED value of this machine, carried by the hardware profile per
    engine (`cpu.runtime_base_mb`); a profile without it prices no base and says so;
  device — on a UNIFIED device (the profile's `has_unified_memory`), the device plan itself: its bytes
    ARE host memory, so the plan's planned device memory is held on the host too. On a discrete card
    the pools are disjoint and this term is 0. The Mac, 2026-10-03: granite-speech triton planned
    18 085 MB on the device and 1 019 MB of host footprint from one 24 GB pool; a ledger reserving
    the 1 019 MB alone admits it beside a VM holding most of the machine, and the run swaps.

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
                   resident_bytes: int = 0, output_bytes: int = 0, device_bytes: int = 0,
                   streamed_pieces: Optional[Mapping[str, tuple]] = None) -> Dict:
    """The host bytes `plan` holds on `engine`: {total, resident, base, steady (+ per component), transient}.

    key_sizes    {component: {weight key: stored bytes}}  (the weights index)
    shard_sizes  {component: {shard name: bytes}}         (the container's shards)
    stored_dtypes {component: {stored floating dtypes}}  (the weights index): what a load reads at, beside
                 the plan's dtype it pins at
    base_mb      what this engine's device work adds on this machine, or None (not measured)
    resident_bytes what the planning process holds when it prices (the caller measures it)
    output_bytes   what the output boundary holds (the largest graph output x the family's save cost)
    device_bytes   the device plan's bytes when the device draws on host memory (unified), 0 otherwise
    streamed_pieces {component: (dearest piece's weight bytes, all its pieces' weight bytes)} for the
                 components the plan streams: the strategy loads them one piece at a time, so one load
                 holds the dearest piece's share of the stored bytes, never the whole component
                 (deepseek-moe-16b-chat, the Mac 2026-10-08: "loading 61 670 MB" priced, 3 606 MB peak)
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
        def _one_load(name, stored_bytes):
            dearest, total = (streamed_pieces or {}).get(name, (1, 1))
            return stored_bytes * dearest // total
        passes = [_passes_through(name, _one_load(name, sum(sh.values())))
                  for name, sh in shard_sizes.items() if sh]
        transient = sum(passes) if plan.loading_mode == "eager" else max(passes, default=0)
    else:
        transient = TRITON_COPIES_PER_TENSOR * max(
            (n for ks in key_sizes.values() for n in ks.values()), default=0)
    base = int(base_mb) << 20 if base_mb is not None else 0
    return {"engine": engine,
            "total_bytes": int(resident_bytes) + base + steady_bytes + transient + int(output_bytes) + int(device_bytes),
            "resident_bytes": int(resident_bytes), "output_bytes": int(output_bytes), "device_bytes": int(device_bytes),
            "base_bytes": base, "base_measured": base_mb is not None,
            "steady_bytes": steady_bytes, "steady": steady, "transient_bytes": transient,
            "loading": plan.loading_mode}


def process_footprint_now() -> int:
    """This process's memory NOW, as the host's memory guard counts it — the floor a price door adds a
    key's priced peak to, and the `resident` term of a plan's host footprint (what the planning process
    holds when it prices; solve()'s unified descent takes it off as already out of the reading). ONE
    reader for both: the resident term was read by its own function, ru_maxrss on macOS — a high-water
    mark, not what the process holds, so the descent took off more than the reading had excluded and
    the host ledger reserved it in full. DeepSeek-Coder-V2-Lite-Instruct on the Mac (2026-10-04 20:25,
    15.5-15.7 GB free): resident 5 025 MB, yet the plan read at least 12 288 MB free in that same
    process (it planned the 12 288 rung, base 0), so it held at most ~3.2 GB then; a host side of
    17 128 MB was accepted as 12 103 and no ledger could admit it. Linux: VmRSS (/proc/self/status). macOS: the physical footprint
    (`proc_pid_rusage`, `ri_phys_footprint` — what `footprint -f` reports and what a unified-memory
    guard kills on), never `ru_maxrss`, a high-water mark: the certifier read it ONCE at its start
    (425 MiB on the Mac, 2026-10-03) while the process sat at 1.1-1.7 GB between keys, so the door
    admitted up to ~1 GB more than its budget. Anywhere else: refused by name (ZERO FALLBACK)."""
    import os
    import sys
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) << 10
        raise RuntimeError("ZERO FALLBACK: /proc/self/status carries no VmRSS")
    if sys.platform == "darwin":
        import ctypes

        class _RusageInfoV0(ctypes.Structure):                      # <sys/resource.h>, RUSAGE_INFO_V0
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), ("ri_user_time", ctypes.c_uint64),
                        ("ri_system_time", ctypes.c_uint64), ("ri_pkg_idle_wkups", ctypes.c_uint64),
                        ("ri_interrupt_wkups", ctypes.c_uint64), ("ri_pageins", ctypes.c_uint64),
                        ("ri_wired_size", ctypes.c_uint64), ("ri_resident_size", ctypes.c_uint64),
                        ("ri_phys_footprint", ctypes.c_uint64), ("ri_proc_start_abstime", ctypes.c_uint64),
                        ("ri_proc_exit_abstime", ctypes.c_uint64)]
        info = _RusageInfoV0()
        rc = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 0, ctypes.byref(info))
        if rc != 0:
            raise RuntimeError(f"ZERO FALLBACK: proc_pid_rusage returned {rc}")
        return int(info.ri_phys_footprint)
    raise RuntimeError(f"ZERO FALLBACK: no reader of a process's footprint on {sys.platform}")


def summary(hf: Mapping) -> str:
    """The one line a run and `--explain-plan` print for the plan's host figure — the line the regression
    matrix reads into a cell's row as the estimate its measured peak judges."""
    base = f"{hf['base_bytes'] / 2**20:.0f} MB" if hf["base_measured"] else "UNMEASURED on this profile"
    held = ", ".join(f"{k} {v / 2**20:.0f} MB" for k, v in hf["steady"].items())
    return (f"{hf['total_bytes'] / 2**20:.0f} MB on the {hf['engine']} engine = resident "
            f"{hf.get('resident_bytes', 0) / 2**20:.0f} MB + engine {base}"
            f" + held {hf['steady_bytes'] / 2**20:.0f} MB ({hf['loading']})"
            f" + loading {hf['transient_bytes'] / 2**20:.0f} MB + output {hf.get('output_bytes', 0) / 2**20:.0f} MB"
            + (f" + device plan {hf['device_bytes'] / 2**20:.0f} MB (unified)" if hf.get("device_bytes") else "")
            + (f"  [held: {held}]" if held else ""))
