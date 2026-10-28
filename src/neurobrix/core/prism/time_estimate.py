"""The estimated time of a candidate plan for a request (docs/design/prism-time-score.md §3), from the work each
component does (`core/prism/work.py`), how many times the request runs it, and the hardware's MEASURED facts
(`tools/profile_facts.py`, recorded on the hardware profile with their source).

  compute    per use, on the device the plan places the component on: a roofline,
             max(flops / matmul throughput at the compute dtype, (weight + activation bytes) / memory bandwidth);
             on the host: the same roofline against the host's GEMM throughput and memory bandwidth
  movement   what the strategy moves: every component's weights loaded once over the host link; a strategy that
             streams weights per use (zero3, layer_streaming) moves them every use; one that loads on demand per phase
             (the lazy strategies) once per phase; cpu_streaming reads them from storage every use

A fact the estimate needs and the profile does not carry is refused by name (`MissingFact`), never defaulted: an
estimate built on an invented throughput would rank plans by the invention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from neurobrix.core.prism.work import Work

#: strategies whose weights cross the host link at every use of a component
STREAM_EVERY_USE = frozenset({"zero3", "layer_streaming"})
#: strategies that load a component's weights on demand, once per phase that uses it
LOAD_PER_PHASE = frozenset({"component_placement_lazy", "lazy_sequential", "single_gpu_lifecycle"})
#: strategies that read a component's weights from storage at every use
STORAGE_EVERY_USE = frozenset({"cpu_streaming"})
#: strategies that hold every weight of a component on the accelerators: no expert store, no per-use expert traffic
RESIDENT_STRATEGIES = frozenset({"single_gpu", "single_gpu_lifecycle", "component_placement", "block_scatter",
                                 "pipeline_parallel", "weight_sharding", "op_level_tiling"})


class MissingFact(KeyError):
    """The profile does not carry a measured fact the estimate needs."""


@dataclass(frozen=True)
class DeviceFacts:
    matmul_flops: Mapping[str, float]       # FLOP/s per compute dtype ("float16", "float32", ...)
    memory_bps: float                       # device memory bandwidth, bytes/s
    h2d_bps: float                          # host -> device, page-locked, bytes/s
    h2d_pageable_bps: Optional[float] = None  # host -> device from pageable memory, bytes/s (`h2d_pageable_gbps`)
    #: the engine's own LOADER, GB landed/s (`weight_load_gbps`, tools/profile_facts.py): {engine: {stored dtype ("BF16",
    #: "F32", "F16"): {"device"|"host": {"cold": GB/s, "warm": GB/s}}}} — read + decode + convert + land; None: not measured
    load_rates: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class HostFacts:
    matmul_flops: float                     # host FLOP/s (float32 GEMM)
    memory_bps: float                       # host memory bandwidth, bytes/s (a copy in host memory)
    storage_bps: Optional[float] = None     # sequential read of a weight file, bytes/s
    #: what the loader really takes to RELOAD a streamed piece's weights onto the device, bytes/s — read from the
    #: container (page cache), decode, convert to the device dtype, upload (`weight_reload_gbps`, measured): a strategy that
    #: re-reads a generating component's weights every token pays it at every reload after the first
    reload_bps: Optional[float] = None
    #: what a sliced stretch whose inputs wait on the host moves through host memory per second — its slices cut from
    #: the host tensors and its outputs assembled (`host_slab_gbps`, measured in situ: mochi's VAE at an 8 GB rung on
    #: the M4 Pro, the Mac's diagnosis 4b 2026-10-28)
    slab_bps: Optional[float] = None


def device_facts(measured: Optional[Mapping[str, Any]], where: str) -> DeviceFacts:
    """A device's `measured:` block of the hardware profile (TFLOP/s and GB/s as `profile_facts.py` prints them)."""
    if not measured:
        raise MissingFact(f"{where}: no `measured:` facts on the hardware profile (tools/profile_facts.py)")
    try:
        pageable = measured.get("h2d_pageable_gbps")
        return DeviceFacts(matmul_flops={k: float(v) * 1e12 for k, v in measured["matmul_tflops"].items()},
                           memory_bps=float(measured["memory_gbps"]) * 1e9, h2d_bps=float(measured["h2d_gbps"]) * 1e9,
                           h2d_pageable_bps=float(pageable) * 1e9 if pageable is not None else None,
                           load_rates=measured.get("weight_load_gbps"))
    except KeyError as exc:
        raise MissingFact(f"{where}: the measured facts lack {exc.args[0]!r}") from exc


def host_facts(measured: Optional[Mapping[str, Any]]) -> HostFacts:
    for key in ("host_matmul_gflops", "host_memory_gbps"):
        if not measured or key not in measured:
            raise MissingFact(f"host: no measured `{key}` on the hardware profile (tools/profile_facts.py)")
    s = measured.get("storage_read_gbps")
    r = measured.get("weight_reload_gbps")
    sl = measured.get("host_slab_gbps")
    return HostFacts(matmul_flops=float(measured["host_matmul_gflops"]) * 1e9,
                     memory_bps=float(measured["host_memory_gbps"]) * 1e9,
                     storage_bps=float(s) * 1e9 if s is not None else None,
                     reload_bps=float(r) * 1e9 if r is not None else None,
                     slab_bps=float(sl) * 1e9 if sl is not None else None)


#: a component's allocation strategies whose weights land in HOST memory (pinned) and reach the card from there
HOST_HELD = frozenset({"zero3", "expert_store"})


#: a weights index's dtype name -> the safetensors name the loader rates are keyed by
STORED_AS = {"bfloat16": "BF16", "float32": "F32", "float16": "F16"}


def _loader_bps(f: DeviceFacts, engine: Optional[str], stored: Optional[str], where: str, state: str,
                dev: str) -> float:
    """The engine's measured loader rate for weights stored as `stored`, to `where` ("device" | "host") in `state`
    ("cold" | "warm"), bytes/s."""
    eng = engine or "compiled"
    st = STORED_AS.get(str(stored), str(stored))
    try:
        return float(f.load_rates[eng][st][where][state]) * 1e9
    except (TypeError, KeyError) as exc:
        raise MissingFact(f"{dev}: no measured `weight_load_gbps.{eng}.{st}.{where}.{state}` on the hardware profile — "
                          f"a load is read + decode + convert + land, not a bare copy (tools/profile_facts.py "
                          f"--weights)") from exc


def compute_seconds(work: Work, facts: DeviceFacts, dtype: str) -> float:
    """One use of a component on a device: the roofline of its flops and its bytes."""
    if dtype not in facts.matmul_flops:
        raise MissingFact(f"no measured matmul throughput at {dtype!r} (measured: {sorted(facts.matmul_flops)})")
    return max(work.flops / facts.matmul_flops[dtype], (work.weight_bytes + work.activation_bytes) / facts.memory_bps)


def _device_of(alloc: Any) -> str:
    dev = alloc[0] if isinstance(alloc, tuple) else str(alloc)
    for prefix in ("zero3:", "fgp:", "tp:"):
        if dev.startswith(prefix):
            dev = dev[len(prefix):]
            break
    return dev.split(",")[0].strip()


def _devices_of(alloc: Any) -> list:
    """Every device string an allocation spans, in order (a `fgp:`/`tp:` list, a shard map's devices, or one device)."""
    out = []
    parts = alloc if isinstance(alloc, tuple) else (alloc,)
    for p in parts:
        if isinstance(p, str):
            d = p
            for prefix in ("zero3:", "fgp:", "tp:"):
                if d.startswith(prefix):
                    d = d[len(prefix):]
                    break
            for x in d.split(","):
                x = x.strip()
                if x and x not in out:
                    out.append(x)
        elif isinstance(p, dict):
            for x in p.values():
                if isinstance(x, str) and x not in out:
                    out.append(x)
    return out


def link_seconds(strategy: str, allocations: Mapping[str, Any], works: Mapping[str, Work], uses: Mapping[str, int],
                 links: Mapping[tuple, float]) -> float:
    """The device-to-device term (docs/design/prism-fastest-plan-that-fits.md §2c): for every boundary a plan crosses,
    the bytes crossing per use divided by the link's measured bandwidth (`links`: {(a, b): bytes/s}, both orders, from
    the profile's `interconnect` or its stated `pcie_fallback` — never a literal). Inside a scattered component
    (`fgp:` / `tp:` over n devices) the residual stream crosses n-1 boundaries per use at the component's output width;
    between two components placed on different devices the producer's output crosses once per use of the consumer.
    `weight_sharding` moves partial sums per sliced GEMM per step: not derived yet, refused by name."""
    if strategy == "weight_sharding":
        raise MissingFact("weight_sharding's partial sums per sliced GEMM are not derived yet: no link term")
    total = 0.0
    prev_dev, prev_comp = None, None
    for comp, alloc in allocations.items():
        devs = [d for d in _devices_of(alloc) if not d.startswith("cpu")]
        w, n = works.get(comp), int(uses.get(comp, 1))
        if w is None or not devs:
            prev_dev, prev_comp = (devs[-1] if devs else None), comp
            continue
        for a, b in zip(devs, devs[1:]):                      # the boundaries inside a scattered component
            total += n * w.output_bytes / _link(links, a, b)
        if prev_dev is not None and prev_dev != devs[0] and prev_comp in works:
            total += n * works[prev_comp].output_bytes / _link(links, prev_dev, devs[0])
        prev_dev, prev_comp = devs[-1], comp
    return total


def _link(links: Mapping[tuple, float], a: str, b: str) -> float:
    bps = links.get((a, b)) or links.get((b, a))
    if not bps:
        raise MissingFact(f"no measured bandwidth between {a} and {b} on the hardware profile (interconnect / pcie_fallback)")
    return float(bps)


def plan_seconds(strategy: str, allocations: Mapping[str, Any], works: Mapping[str, Work], uses: Mapping[str, int],
                 devices: Mapping[str, DeviceFacts], host: Optional[HostFacts], dtype: str,
                 phases: Optional[Mapping[str, int]] = None, prompt: Optional[Mapping[str, Work]] = None,
                 links: Optional[Mapping[tuple, float]] = None, engine: Optional[str] = None,
                 stored: Optional[Mapping[str, str]] = None) -> float:
    """The plan's estimated seconds for the request. `devices`: {device string ("cuda:0"): its facts}; `uses`: {component:
    runs per request}; `phases`: {component: phases that load it} for the strategies that load per phase (1 when
    absent); `prompt`: {component: the work of its one prompt pass} for a generating component whose `works` entry is
    the work of ONE token — the prompt pass runs once more, over the same loaded weights; `stored`: {component: the
    dtype most of its weights are stored in} (the weights index), which the loader rate is read for."""
    total = 0.0
    for comp, alloc in allocations.items():
        if comp not in works:
            continue
        w, n = works[comp], int(uses.get(comp, 1))
        first = (prompt or {}).get(comp)                      # the prompt pass, one more use of the same weights
        passes = n + (1 if first is not None else 0)
        dev = _device_of(alloc)
        if dev.startswith("cpu"):
            if host is None:
                raise MissingFact(f"{comp} is placed on the host and the host's facts are not measured")
            roof = lambda x: max(x.flops / host.matmul_flops, (x.weight_bytes + x.activation_bytes) / host.memory_bps)
            total += n * roof(w) + (roof(first) if first is not None else 0.0)
            if strategy in STORAGE_EVERY_USE:
                if host.storage_bps is None:
                    raise MissingFact("cpu_streaming needs the measured storage read bandwidth")
                total += passes * w.weight_bytes / host.storage_bps
            continue
        if dev not in devices:
            raise MissingFact(f"{comp} is placed on {dev!r}, whose facts are not measured")
        f = devices[dev]
        total += n * compute_seconds(w, f, dtype) + (compute_seconds(first, f, dtype) if first is not None else 0.0)
        loads = passes if strategy in STREAM_EVERY_USE else (int((phases or {}).get(comp, 1))
                                                             if strategy in LOAD_PER_PHASE else 1)
        # A load is the engine's LOADER — read (storage or the page cache), decode, convert to the device dtype, land —
        # at its measured rate (`weight_load_gbps`), never a bare copy: the rule-13 proof (V100 rack, 2026-10-27) loaded
        # Wan2.2-I2V-A14B in 123-173 s and Qwen3-30B-A3B in 294-447 s against estimates at the H2D copy rate, and so
        # could not rank what a request really waits for. The first load is cold, a later one finds the page cache warm.
        how = getattr(alloc, "strategy", None) or strategy
        st = (stored or {}).get(comp)
        if st is None:
            raise MissingFact(f"{comp}: the weights index states no stored dtype the loader rate is read for")
        # host-held: the plan's zero3 placement — the component's strategy, or its device written `zero3:<card>` (how a
        # lazy plan states the component it streams from host memory, Qwen3-30B's expert store, Wan2.2's ratchet)
        dev_str = alloc[0] if isinstance(alloc, tuple) else str(alloc)
        host_held = how in HOST_HELD or strategy == "zero3" or str(dev_str).startswith("zero3:")
        if host_held:
            # the blocks land in page-locked host memory once; then, a ratchet streams the blocks beyond what the card
            # keeps at every pass, an expert store only the routed experts a use reads (priced per use below)
            total += w.weight_bytes / _loader_bps(f, engine, st, "host", "cold", dev)
            if not w.expert_bytes_per_use:
                resident = float(getattr(alloc, "resident_weight_mb", 0.0) or 0.0) * 2 ** 20
                total += passes * max(0.0, w.weight_bytes - resident) / f.h2d_bps
        elif strategy in STREAM_EVERY_USE:
            # a piece reloaded from the container at every use after the first (layer_streaming): at the measured
            # reload rate when the profile states one (`weight_reload_gbps`, the Janus case on the Mac), else the
            # loader's warm rate to the device
            again = host.reload_bps if (host is not None and host.reload_bps) else _loader_bps(f, engine, st, "device", "warm", dev)
            total += w.weight_bytes / _loader_bps(f, engine, st, "device", "cold", dev) + (passes - 1) * w.weight_bytes / again
        else:
            total += w.weight_bytes / _loader_bps(f, engine, st, "device", "cold", dev) + \
                (loads - 1) * w.weight_bytes / _loader_bps(f, engine, st, "device", "warm", dev)
        if strategy not in RESIDENT_STRATEGIES:
            # an expert store: the routed experts a use reads cross the host link at every use (per token for a
            # generating component, the prompt's set once) — unpriced, Qwen3-30B estimated 15 s against 174 s measured
            total += n * w.expert_bytes_per_use / f.h2d_bps
            if first is not None:
                total += first.expert_bytes_per_use / f.h2d_bps
    if links is not None:
        total += link_seconds(strategy, allocations, works, uses, links)
    return total


def resolve_flow_name(name: str, components) -> str:
    """The container component a flow names: the exact name, else the one component whose name ends with
    `.<name>` (the flow of GLM-4.1V names `language_model`, the component is `model.language_model`: the executor
    resolves by that suffix, graph_executor `key.endswith("." + leaf)`). Ambiguity or no match: the name as given
    (the caller then prices the component the flow names, or nothing)."""
    names = list(components or [])
    if name in names:
        return name
    hits = [c for c in names if c.endswith("." + name)]
    return hits[0] if len(hits) == 1 else name


def uses_per_request(flow: Mapping[str, Any], steps: Optional[int], tokens: Optional[int],
                     components=None) -> Dict[str, tuple]:
    """{component: (kind, runs)} for a request, read from the container's flow (topology.json) — never from a family:
    a component of an iterative process's loop runs `steps` times ("loop"); the flow's language model and its head
    (`generation.lm_component` / `head_component`) and a stage executed "autoregressive" run once at the prompt and
    once per generated token ("token", `tokens`); every other component once ("once"). `steps` / `tokens` are the
    request's (or the container's defaults); a flow that loops or generates without them is refused by name."""
    out: Dict[str, tuple] = {}
    for comp in list(flow.get("order") or []) + list(flow.get("pre_loop") or []) + list(flow.get("post_loop") or []):
        out.setdefault(comp, ("once", 1))
    loop = (flow.get("loop") or {}).get("components") or []
    if loop:
        if not steps:
            raise MissingFact(f"the flow loops over {loop} and the request names no step count")
        for comp in loop:
            out[comp] = ("loop", int(steps))
        if (flow.get("generation") or {}).get("boundary_ratio") and len(loop) >= 2:
            # A dual denoiser (Wan2.2-A14B's two experts, `core/flow/iterative_process._setup_dual_denoiser`): EACH step
            # runs ONE expert, the high-noise one above the boundary, the low-noise one below — `steps` uses between
            # them, not `steps` each (the rule-13 proof priced Wan2.2's layer_streaming at twice its reloads, 2026-10-27).
            # Every term of a plan's time is linear in an expert's uses and the experts are alike, so the split is read
            # as even: the sum (the steps) is exact, and so is each expert's one cold load while both run.
            k = len(loop)
            for i, comp in enumerate(loop):
                out[comp] = ("loop", int(steps) // k + (1 if i < int(steps) % k else 0))
    gen = flow.get("generation") or {}
    token_comps = [c for c in (gen.get("lm_component"), gen.get("head_component")) if c]
    token_comps += [s["component"] for s in (flow.get("stages") or []) if s.get("execution") == "autoregressive"]
    # the flow names components as the executor resolves them (a `.<name>` suffix): GLM-4.1V's `language_model` is
    # `model.language_model` — unresolved, its per-token weight reloads went unpriced and layer_streaming estimated 2.5 s
    token_comps = [resolve_flow_name(c, components) for c in token_comps]
    for comp in list(out):
        if comp not in (components or out) and resolve_flow_name(comp, components) != comp:
            out[resolve_flow_name(comp, components)] = out.pop(comp)
    if token_comps:
        if not tokens:
            raise MissingFact(f"the flow generates with {token_comps} and the request names no token count")
        for comp in token_comps:
            out[comp] = ("token", int(tokens))
    return out
