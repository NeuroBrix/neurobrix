"""The expert topology of a routed (mixture-of-experts) component, and the working set it implies.

A routed component is not a dense one that happens to be large. Per token, each routed layer reads
its router, then `top_k` of its `n_experts` expert FFNs; the rest of its weights sit idle. A plan
that prices the component as one dense block (every expert, every token) streams the idle experts
too: a 30B-parameter model with 3B active on a 32 GB card was priced at ~57 GB and streamed whole per token, although
8 of 128 experts per layer run.

This module READS the topology; it decides nothing. Sparsity is a dimension of the plan, not a
strategy: the figure is stated beside the plan's other figures. Placement reads the topology itself
(the solver's `_expert_store_floor`: an expert store's dense part and its widest top_k experts), never
this stated working set, which is priced on the profile's standard rung.

Where the topology comes from — the container, nothing else:

* the graph, as the runtime executes it: the declared-MoE fusion (`moe_fusion.detect_and_fuse_moe`,
  the recogniser both engines run) turns each router -> topk -> indexed-experts pattern into one
  `custom::moe_fused` op whose attributes carry `num_experts`, `top_k`, `norm_topk_prob` and the
  expert weights (per-expert matrices, or a stacked slab). One recogniser, so the plan reads the
  experts the engines will read;
* the bytes: each expert weight's size from `weights_index.json` (else its graph shape and dtype),
  scaled by the component's dtype factor, the same reading `_consumed_weight_bytes` sizes the
  component with — the routed bytes are a part of the figure the plan already states;
* the shared experts: the parameters of the same block whose NeuroTax key carries the shared-expert
  token. Their count is the package's declaration (`lm_config.num_shared_experts`) when the package
  declares this component; their bytes are always read from the graph;
* the cross-check: for the component the package's `lm_config` declares, the graph's `num_experts`
  and `top_k` must equal `lm_config.num_experts` / `num_experts_per_tok`. A disagreement is refused
  by name — one of the two is a defect, and a plan does not pick one silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from neurobrix.nbx.neurotax import SynonymRegistry as _NeuroTax

MB = 1024 * 1024

#: The fused op the runtime's MoE recogniser emits — the topology is read from it.
FUSED_OP = "custom::moe_fused"

# The key tokens, as the one NeuroTax parser names them.
_LAYERS = _NeuroTax.resolve("layers")
_SHARED = _NeuroTax.resolve("shared_experts")
_BLOCK_OF = re.compile(rf"^(?:param::)?{re.escape(_LAYERS)}\.(\d+)\.")



@dataclass
class RoutedLayer:
    """One routed layer, as its fused op reads it."""
    op_uid: str
    block: Optional[int]
    n_experts: int
    top_k: int
    norm_topk_prob: Optional[bool]
    expert_bytes: int          # one routed expert's weights, at the component's runtime dtype
    shared_bytes: int          # the layer's shared-expert weights (0 when it has none)
    stacked: bool = False      # the experts are one [E, ...] slab per projection, not per-expert matrices
    #: the routed experts' weight names (the graph's `weight_name`, the weights index's keys) — what an
    #: expert store reads from the container's files per miss; empty for a stacked slab
    expert_weight_names: Tuple[str, ...] = ()


@dataclass
class ExpertTopology:
    """A routed component's experts: what a token reads, what sits idle."""
    component: str
    layers: List[RoutedLayer] = field(default_factory=list)
    shared_experts: Optional[int] = None    # declared count; None when the package does not declare it

    @property
    def n_experts(self) -> List[int]:
        return sorted({l.n_experts for l in self.layers})

    @property
    def top_k(self) -> List[int]:
        return sorted({l.top_k for l in self.layers})

    @property
    def routed_bytes(self) -> int:
        """Every routed expert of every layer — what a dense plan streams per token."""
        return sum(l.n_experts * l.expert_bytes for l in self.layers)

    @property
    def active_bytes(self) -> int:
        """The `top_k` experts per layer a token reads."""
        return sum(l.top_k * l.expert_bytes for l in self.layers)

    @property
    def shared_bytes(self) -> int:
        return sum(l.shared_bytes for l in self.layers)

    def record(self) -> Dict[str, Any]:
        def one(values):
            return values[0] if len(values) == 1 else values
        experts_mb = sorted({round(l.expert_bytes / MB, 3) for l in self.layers})
        shared_mb = sorted({round(l.shared_bytes / MB, 3) for l in self.layers})
        norm = sorted({l.norm_topk_prob for l in self.layers}, key=str)
        return {"component": self.component, "routed_layers": len(self.layers),
                "n_experts": one(self.n_experts), "top_k": one(self.top_k),
                "norm_topk_prob": one(norm), "shared_experts": self.shared_experts,
                "expert_mb": one(experts_mb), "shared_expert_mb_per_layer": one(shared_mb),
                "routed_mb": round(self.routed_bytes / MB, 1),
                "active_mb": round(self.active_bytes / MB, 1)}


def _tensor_bytes(tid: str, tensors: Dict[str, Any], sizes: Optional[Dict[str, Any]]) -> int:
    t = tensors.get(tid)
    if t is None:
        raise RuntimeError(f"ZERO FALLBACK: the fused MoE op reads {tid!r}, which the graph does not hold")
    name = t.get("weight_name") or tid
    if sizes is not None and name in sizes:
        return int((sizes[name] or {}).get("size_bytes", 0))
    n = 1
    for d in t.get("shape") or []:
        if not isinstance(d, int) or d < 0:
            raise RuntimeError(f"ZERO FALLBACK: expert weight {tid!r} has a symbolic shape {t.get('shape')} "
                               "and no weights_index entry — its bytes cannot be read")
        n *= d
    from nbxtensor.dtype import itemsize as _itemsize   # the one table (config/dtypes.yml)
    try:
        width = _itemsize.itemsize(t.get("dtype"))
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"ZERO FALLBACK: expert weight {tid!r} has dtype {t.get('dtype')!r}, of unknown "
                           f"width ({e})") from e
    return n * width


def _block_of(tid: str, tensors: Dict[str, Any]) -> Optional[int]:
    t = tensors.get(tid) or {}
    m = _BLOCK_OF.match(t.get("weight_name") or tid)
    return int(m.group(1)) if m else None


def read_expert_topology(component: str, graph: Dict[str, Any], weights_index: Optional[Dict[str, Any]],
                         dtype_mult: float, lm_config: Optional[Dict[str, Any]] = None,
                         declared_component: Optional[str] = None) -> Optional[ExpertTopology]:
    """The topology of `graph` (the graph AS EXECUTED, i.e. after the declared-MoE fusion), or None
    when it holds no fused routed layer. `lm_config` is cross-checked only for the component it
    declares (`declared_component`)."""
    ops = graph.get("ops") or {}
    tensors = graph.get("tensors") or {}
    order = graph.get("execution_order") or list(ops)
    sizes = None
    if isinstance(weights_index, dict) and isinstance(weights_index.get("tensors"), dict):
        sizes = weights_index["tensors"]
    fused = [u for u in order if (ops.get(u) or {}).get("op_type") == FUSED_OP]
    if not fused:
        return None

    # The shared-expert parameters, by block: the NeuroTax token, one segment of the key.
    shared_by_block: Dict[int, int] = {}
    for tid, t in tensors.items():
        if not t.get("is_parameter"):
            continue
        name = t.get("weight_name") or tid
        if _SHARED not in name.split("."):
            continue
        b = _block_of(tid, tensors)
        if b is not None:
            shared_by_block[b] = shared_by_block.get(b, 0) + _tensor_bytes(tid, tensors, sizes)

    topo = ExpertTopology(component=component)
    for uid in fused:
        a = ops[uid].get("attributes") or {}
        for key in ("num_experts", "top_k"):
            if a.get(key) is None:
                raise RuntimeError(f"ZERO FALLBACK: fused MoE op {uid} carries no {key!r}")
        e, k = int(a["num_experts"]), int(a["top_k"])
        st = a.get("stacked_experts")
        names: Tuple[str, ...] = ()
        if st:
            slab = (_tensor_bytes(st["input_linear_tid"], tensors, sizes)
                    + _tensor_bytes(st["output_linear_tid"], tensors, sizes))
            first = st["input_linear_tid"]
        else:
            ids = [list(a.get(f) or []) for f in
                   ("expert_gate_weight_ids", "expert_up_weight_ids", "expert_down_weight_ids")]
            if any(len(x) != e for x in ids):
                raise RuntimeError(f"ZERO FALLBACK: fused MoE op {uid} lists {[len(x) for x in ids]} expert "
                                   f"weights for {e} experts")
            slab = sum(_tensor_bytes(t, tensors, sizes) for x in ids for t in x)
            first = ids[0][0]
            names = tuple((tensors.get(t) or {}).get("weight_name") or t for x in ids for t in x)
        if slab % e:
            raise RuntimeError(f"fused MoE op {uid}: {slab} expert bytes do not divide into {e} equal experts")
        block = _block_of(first, tensors)
        topo.layers.append(RoutedLayer(
            op_uid=uid, block=block, n_experts=e, top_k=k,
            norm_topk_prob=(None if a.get("norm_topk_prob") is None else bool(a["norm_topk_prob"])),
            expert_bytes=int(slab // e * dtype_mult),
            shared_bytes=int(shared_by_block.get(block, 0) * dtype_mult) if block is not None else 0,
            stacked=bool(st), expert_weight_names=names))

    if lm_config is not None and declared_component == component:
        want_e, want_k = lm_config.get("num_experts"), lm_config.get("num_experts_per_tok")
        if want_e is not None and topo.n_experts != [int(want_e)]:
            raise RuntimeError(f"{component}: the graph routes over {topo.n_experts} experts, "
                               f"lm_config declares num_experts={want_e}")
        if want_k is not None and topo.top_k != [int(want_k)]:
            raise RuntimeError(f"{component}: the graph selects top_k={topo.top_k}, "
                               f"lm_config declares num_experts_per_tok={want_k}")
        if lm_config.get("num_shared_experts") is not None:
            topo.shared_experts = int(lm_config["num_shared_experts"])
    if topo.shared_experts is None and not any(l.shared_bytes for l in topo.layers):
        topo.shared_experts = 0
    return topo


def price_working_set(topologies: Dict[str, ExpertTopology], weight_bytes: Dict[str, int],
                      rung_mb: int, reserved_bytes: int, rung_source: str) -> Optional[Dict[str, Any]]:
    """The routed working set, stated — never used to place (yet).

    * dense: every weight of the plan's components less the routed experts (attention, norms,
      embeddings, routers, shared experts, every dense component);
    * minimum: dense + the `top_k` experts each routed layer reads for one token;
    * expert cache: what the profile's standard rung leaves after the minimum and the plan's
      non-weight reservations (`reserved_bytes`: activations and the KV cache), capped at the idle
      experts — the room an expert store would hold recently routed experts in;
    * working set: minimum + expert cache.
    """
    if not topologies:
        return None
    total = sum(int(b) for b in weight_bytes.values())
    routed = sum(t.routed_bytes for t in topologies.values())
    active = sum(t.active_bytes for t in topologies.values())
    dense = total - routed
    minimum = dense + active
    room = rung_mb * MB - minimum - int(reserved_bytes)
    cache = max(0, min(room, routed - active))
    per_layer_expert = sum(l.expert_bytes for t in topologies.values() for l in t.layers)
    extra = int(cache // per_layer_expert) if per_layer_expert else 0
    extra = min(extra, min(l.n_experts - l.top_k for t in topologies.values() for l in t.layers))
    return {
        "basis": "weights at the plan's dtype; stated, not used to place",
        "dense_equivalent_mb": round(total / MB, 1),
        "dense_mb": round(dense / MB, 1),
        "routed_mb": round(routed / MB, 1),
        "active_experts_mb": round(active / MB, 1),
        "minimum_mb": round(minimum / MB, 1),
        "rung_mb": int(rung_mb), "rung_source": rung_source,
        "reserved_mb": round(int(reserved_bytes) / MB, 1),
        "expert_cache_mb": round(cache / MB, 1),
        "cached_experts_per_layer": extra,
        "working_set_mb": round((minimum + cache) / MB, 1),
        "components": [t.record() for t in topologies.values()],
    }


def routing_reuse(cache_path, component: str) -> Optional[List[Tuple[float, float]]]:
    """The routing's reuse curve of a routed component — {slots per layer: hit rate}, the share of a store layer's routed
    experts an LRU cache of that many slots per layer finds, recorded from a decode (the Dell, Qwen3-30B-A3B,
    2026-10-28: 8 -> 0.30, 16 -> 0.53, 32 -> 0.74, 64 -> 0.93, 96 -> 0.96). Read as the calibration record is
    (`precision_contract.load_calibration`): the ARTIFACT's own section first (components/<component>/profile.json
    `routing_reuse.hit_by_slots_per_layer`, embedded by Forge), else the ENGINE STORE beside the calibration records
    (`<calibration STORE_ROOT>/<model>/<component>.routing_reuse.json` `curve`, written by the expert store at the end
    of a decode, `core/runtime/routing_reuse`). [(slots, hit)] ascending, or None when neither states one (the price
    then refuses by name). Card-less tests pin the store empty (`_pinned_machine.pin_no_calibration`)."""
    import json
    from pathlib import Path
    if cache_path is None:
        return None
    curve = None
    f = Path(cache_path) / "components" / component / "profile.json"
    if f.exists():
        curve = ((json.loads(f.read_text()) or {}).get("routing_reuse") or {}).get("hit_by_slots_per_layer")
        src = f
    if not curve:
        from neurobrix.core.dtype.calibration import STORE_ROOT
        from neurobrix.core.runtime.precision_contract import registry_model_name
        src = Path(STORE_ROOT) / str(registry_model_name(cache_path)) / f"{component}.routing_reuse.json"
        if src.exists():
            curve = (json.loads(src.read_text()) or {}).get("curve")
    if not curve:
        return None
    pts = sorted((float(k), float(v)) for k, v in curve.items())
    if any(not 0.0 <= h <= 1.0 for _, h in pts):
        raise ValueError(f"ZERO FALLBACK: {src}: routing_reuse hit rates outside [0, 1]: {pts}")
    return pts


def hit_rate(curve: List[Tuple[float, float]], slots_per_layer: float) -> float:
    """The curve read at `slots_per_layer`: linear between its points, 0 hits with no slot, its last rate beyond it
    (a cache larger than the measured range hits at least as often)."""
    if slots_per_layer <= 0:
        return 0.0
    prev = (0.0, 0.0)
    for s, h in curve:
        if slots_per_layer <= s:
            return prev[1] + (h - prev[1]) * (slots_per_layer - prev[0]) / (s - prev[0])
        prev = (s, h)
    return curve[-1][1]
