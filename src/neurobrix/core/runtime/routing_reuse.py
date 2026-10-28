"""The routing reuse curve of a routed (MoE) component: the fraction of the experts a decode token routes to that a
least-recently-used cache of `s` slots per routed layer already holds, for every `s` of a ladder read from the model
(multiples of its top_k up to its expert count). Measured from the engine's own decode — an expert store notes each
decode call's routed set (`ExpertTrace`), and its release writes the curve once per model and component — and kept
in the engine store beside the calibration records (`core/dtype/calibration.STORE_ROOT`). Prism sizes an expert cache
and prices its misses from it; while a model has none, the plan says so (no default curve).

Qwen3-30B-A3B-Thinking, 256 decode tokens, 17 store layers (2026-10-28): 8 slots 0.30, 26 slots 0.67, 64 slots 0.93,
128 slots 0.96 — a cache of 60 slots over 14 layers missed almost every routed expert (tpot 381 -> 468 ms).

Pure Python, torch-free: both engines' stores import it (R33)."""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def slot_ladder(top_k: int, n_experts: int) -> List[int]:
    """The slot counts the curve is read at: every multiple of `top_k` up to `n_experts`."""
    if top_k < 1 or n_experts < top_k:
        raise ValueError(f"ZERO FALLBACK: a routing reuse curve needs 1 <= top_k <= n_experts (top_k={top_k}, "
                         f"n_experts={n_experts})")
    return list(range(top_k, n_experts + 1, top_k)) + ([n_experts] if n_experts % top_k else [])


def lru_hit_rate(calls: List[List[int]], slots: int) -> float:
    """The fraction of routed experts an LRU of `slots` holds, over `calls` (each a routed set) in order."""
    cache: "collections.OrderedDict[int, None]" = collections.OrderedDict()
    hits = total = 0
    for routed in calls:
        for e in routed:
            total += 1
            if e in cache:
                hits += 1
                cache.move_to_end(e)
            else:
                cache[e] = None
                if len(cache) > slots:
                    cache.popitem(last=False)
    return hits / total if total else 0.0


def curve(traces: Dict[str, List[List[int]]], top_k: int, n_experts: int) -> Dict[int, float]:
    """{slots per layer: mean hit rate over the traced layers} for every slot count of the ladder."""
    layers = [calls for calls in traces.values() if calls]
    if not layers:
        raise ValueError("ZERO FALLBACK: no traced decode call — a routing reuse curve needs the decode's routed sets")
    return {s: round(sum(lru_hit_rate(c, s) for c in layers) / len(layers), 4) for s in slot_ladder(top_k, n_experts)}


def _root() -> Path:
    from neurobrix.core.dtype.calibration import STORE_ROOT
    return Path(STORE_ROOT)


def path(model: str, component: str) -> Path:
    return _root() / model / f"{component}.routing_reuse.json"


def load(model: str, component: str) -> Optional[Dict[str, object]]:
    """The recorded fact ({"curve": {slots: hit}, "tokens", "layers", "top_k", "n_experts", "source"}), or None."""
    p = path(model, component)
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    data["curve"] = {int(k): float(v) for k, v in (data.get("curve") or {}).items()}
    return data


def record(model: str, component: str, traces: Dict[str, List[List[int]]], top_k: int, n_experts: int,
           source: str) -> Optional[Path]:
    """Write the curve of `traces` for `model`/`component` unless one is recorded already; returns the path written."""
    p = path(model, component)
    if p.is_file():
        return None
    tokens = max((len(c) for c in traces.values()), default=0)
    data = {"curve": {str(k): v for k, v in curve(traces, top_k, n_experts).items()}, "tokens": tokens,
            "layers": sum(1 for c in traces.values() if c), "top_k": top_k, "n_experts": n_experts, "source": source}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, p)
    return p


class ExpertTrace:
    """Each decode call's routed set per layer, noted by an expert store (`acquire`), written as the curve when the
    store is released. A call routing more than `top_k` experts (a prefill) is not a decode token and is not noted."""

    def __init__(self, model: str, component: str, top_k: int, n_experts: int):
        self.model, self.component, self.top_k, self.n_experts = model, component, int(top_k), int(n_experts)
        self.calls: Dict[str, List[List[int]]] = {}
        self.written = False
        # a generation's store is not released before the process ends: the decode it saw is written then
        import atexit
        atexit.register(self.write, "recorded by an expert store's decode (written at the process's end)")

    def note(self, layer, routed) -> None:
        if len(routed) <= self.top_k:
            self.calls.setdefault(str(layer), []).append(sorted(int(e) for e in routed))

    def write(self, source: str) -> Optional[Path]:
        if self.written or not any(self.calls.values()):
            return None
        self.written = True
        return record(self.model, self.component, self.calls, self.top_k, self.n_experts, source)
