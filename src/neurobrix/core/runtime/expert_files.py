"""Routed experts read from the container's files, never loaded whole — the unified expert store's host side.

On a unified device (Apple silicon: one pool for host and device) Prism holds a routed component as an
expert store whose routed experts stay in the container's safetensors shards (`_place_component`
Strategy 2.5; the plan's `ComponentAllocation.expert_files`). zero3's form — every expert loaded into
pinned host memory — is not available there: on Metal "pinned" host memory is a shared MTLBuffer
counted against the same working set as the device (`metal_device.py` `malloc_host`), so the 55 GB
of Qwen3-30B-A3B's experts would be loaded into the 24 GB the machine has.

So the loader loads the DENSE part to the device as usual, and binds each routed expert weight to a
VIEW of its bytes in its shard, mapped privately (copy-on-write, never written): nothing is read at
load, a miss reads the expert's file pages when the store copies it into a device slot, and those
pages are the page cache's — clean and reclaimable, not the process's anonymous memory.

A view is the STORED bytes, so it is the weight the engine computes with only when the stored dtype
is the dtype the engine holds that weight in (`stored_dtype_in_compute` on the triton engine; the
plan's dtype for a floating weight on the compiled one). Any other pair would need a conversion per
miss, which the store's byte-copy contract does not make: it is refused by name, never converted
silently. Only the fused MoE op reads these weights, through the store (`expert_store.py`); a view
reaching any kernel is a defect the engines refuse (an mmap address is no device buffer).

Torch-free here; each engine builds its own view from `ExpertFileSpan`. `held_dtype` is the one rule
both engines' views and Prism's plan read (it imports the triton loader's torch-free
`stored_dtype_in_compute` for the triton engine only).
"""
from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

from neurobrix.core.runtime.expert_residency import routed_layers
from neurobrix.nbx.weight_presence import AbsentWeightError, read_safetensors_header, read_weights_index


@dataclass(frozen=True)
class ExpertFileSpan:
    """Where one stored weight's bytes lie: its shard, the absolute offset and length, its stored
    safetensors dtype (`BF16`, `F16`, ...) and shape."""
    shard: str
    offset: int
    nbytes: int
    dtype: str
    shape: Tuple[int, ...]


def expert_graph_names(dag: dict) -> Set[str]:
    """The graph names (`param::` stripped, the reconcile's names) of every routed expert weight the
    fused MoE ops of `dag` read — `routed_layers` refuses a stacked slab by name."""
    out: Set[str] = set()
    for ids in routed_layers(dag or {}).values():
        for proj in ids:
            for tid in proj:
                out.add(tid[7:] if tid.startswith("param::") else tid)
    return out


def expert_loader_keys(dag: dict, binding: Optional[Dict[str, str]], loader_keys: Iterable[str]) -> Set[str]:
    """The loader keys (the weights index's) of the routed experts, among `loader_keys`. `binding` is
    the load's graph name → loader key map (`binding_of_the_loaded`); a name it does not bind is its
    own key. A routed expert the load set does not hold is refused: the store would meet None."""
    keys = set(loader_keys)
    names = expert_graph_names(dag)
    if not names:
        raise RuntimeError("ZERO FALLBACK: the plan maps this component's routed experts from its files, "
                           "and its graph holds no fused routed layer (custom::moe_fused)")
    out = {(binding or {}).get(n, n) for n in names}
    missing = sorted(out - keys)
    if missing:
        raise AbsentWeightError(f"{len(missing)} routed expert weight(s) the store maps from the files are "
                                f"not in the load set: {missing[:3]}")
    return out


def locate(nbx_path, component: str, keys: Iterable[str]) -> Dict[str, ExpertFileSpan]:
    """{key: its span} for `keys`, read from the component's weights index and each shard's header —
    the same two reads the loaders make, nothing else."""
    comp_dir = os.path.join(str(nbx_path), "components", component)
    index = read_weights_index(component, comp_dir) or {}
    headers: Dict[str, Tuple[dict, int]] = {}
    out: Dict[str, ExpertFileSpan] = {}
    for key in sorted(set(keys)):
        entry = index.get(key)
        if not isinstance(entry, dict) or not entry.get("shard"):
            raise AbsentWeightError(f"component '{component}': {key!r} has no shard in its weights index")
        shard = os.path.join(comp_dir, "weights", entry["shard"])
        if shard not in headers:
            headers[shard] = read_safetensors_header(shard)
        header, data_start = headers[shard]
        info = header.get(key)
        if info is None:
            raise AbsentWeightError(f"component '{component}': {key!r} is absent from {shard}, where its "
                                    f"index places it")
        start, end = info["data_offsets"]
        out[key] = ExpertFileSpan(shard, data_start + int(start), int(end) - int(start), str(info["dtype"]),
                                  tuple(int(d) for d in info["shape"]))
    return out


class ShardMaps:
    """Each shard mapped once, privately (MAP_PRIVATE: copy-on-write, never written), for the views'
    lifetime. Every view keeps this object alive through its base."""

    def __init__(self):
        self._maps: Dict[str, mmap.mmap] = {}

    def map(self, shard: str) -> mmap.mmap:
        m = self._maps.get(shard)
        if m is None:
            with open(shard, "rb") as f:
                m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
            self._maps[shard] = m
        return m

    def __len__(self) -> int:
        return len(self._maps)


def held_dtype(stored, compute, engine: str):
    """The dtype (an NBXDtype) `engine` holds a weight stored as `stored` in, under the compute dtype
    `compute` (each anything `parse_dtype` reads: a name, an NBXDtype, a torch dtype) — THE rule a
    file-backed expert is held to, read by both engines' views (`bind_views`'s `held_as`) and by
    Prism when it plans a unified expert store (`_refuse_unheld_expert_files`), so the plan refuses
    exactly what the load would. Triton: `stored_dtype_in_compute`, its loader's one rule. Compiled:
    `core/dtype/config.compiled_weight_dtype`, the ATen loader's one rule (every floating weight at the
    placement dtype, the rest as stored)."""
    from neurobrix.kernels.nbx_tensor import parse_dtype
    s, c = parse_dtype(stored), parse_dtype(compute)
    if engine == "triton":
        from neurobrix.triton.weight_loader import stored_dtype_in_compute
        return stored_dtype_in_compute(s, c)
    if engine == "compiled":
        from neurobrix.core.dtype.config import compiled_weight_dtype
        return parse_dtype(compiled_weight_dtype(s.name, c.name))
    raise RuntimeError(f"ZERO FALLBACK: no engine named {engine!r} (triton, compiled)")


def bind_views(spans: Dict[str, ExpertFileSpan], held_as: Callable[[str], str],
               build: Callable[[mmap.mmap, ExpertFileSpan, str], object]) -> Tuple[Dict[str, object], ShardMaps]:
    """{key: view} over `spans`. `held_as(stored dtype)` names the dtype the engine would hold the
    weight in; a view is built only where the two are the same (module docstring), else refused by
    name. `build(map, span, key)` is the engine's view constructor."""
    maps = ShardMaps()
    views: Dict[str, object] = {}
    for key, span in spans.items():
        want = held_as(span.dtype)
        if want != span.dtype:
            raise RuntimeError(
                f"ZERO FALLBACK: routed expert {key!r} is stored {span.dtype} and this engine computes it "
                f"as {want}; the expert store copies stored bytes into its slots and converts none — "
                f"a file-backed expert must be stored in its compute dtype")
        views[key] = build(maps.map(span.shard), span, key)
    return views, maps


def drop_pages(m: mmap.mmap, offset: int, nbytes: int) -> Tuple[int, int]:
    """Give the kernel back the pages of one expert's span in its private shard map, once its bytes
    sit in a device slot: `madvise(MADV_DONTNEED)` over the page-aligned range (a private, never
    written mapping re-reads the file on the next miss). The pages a copy touched otherwise stay
    counted against the pool on a unified device: the Mac, 2026-10-14, Qwen3-30B-A3B-Thinking native —
    the process's own footprint flat at 8.6 GB while free memory fell and swap grew 7 GB during the
    first forward's misses over 55 GB of mapped experts. Returns (start, length) of the range given
    back. A platform whose mmap cannot advise is refused by name, never left to swap."""
    if not hasattr(m, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError("ZERO FALLBACK: this platform's mmap carries no madvise/MADV_DONTNEED — a file-backed "
                           "expert store cannot give its pages back after a copy")
    page = mmap.PAGESIZE
    start = offset - offset % page
    length = (offset + nbytes) - start
    m.madvise(mmap.MADV_DONTNEED, start, length)
    return start, length

