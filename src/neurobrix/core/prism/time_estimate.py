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


class MissingFact(KeyError):
    """The profile does not carry a measured fact the estimate needs."""


@dataclass(frozen=True)
class DeviceFacts:
    matmul_flops: Mapping[str, float]       # FLOP/s per compute dtype ("float16", "float32", ...)
    memory_bps: float                       # device memory bandwidth, bytes/s
    h2d_bps: float                          # host -> device, page-locked, bytes/s


@dataclass(frozen=True)
class HostFacts:
    matmul_flops: float                     # host FLOP/s (float32 GEMM)
    memory_bps: float                       # host memory bandwidth, bytes/s (a copy in host memory)
    storage_bps: Optional[float] = None     # sequential read of a weight file, bytes/s


def device_facts(measured: Optional[Mapping[str, Any]], where: str) -> DeviceFacts:
    """A device's `measured:` block of the hardware profile (TFLOP/s and GB/s as `profile_facts.py` prints them)."""
    if not measured:
        raise MissingFact(f"{where}: no `measured:` facts on the hardware profile (tools/profile_facts.py)")
    try:
        return DeviceFacts(matmul_flops={k: float(v) * 1e12 for k, v in measured["matmul_tflops"].items()},
                           memory_bps=float(measured["memory_gbps"]) * 1e9, h2d_bps=float(measured["h2d_gbps"]) * 1e9)
    except KeyError as exc:
        raise MissingFact(f"{where}: the measured facts lack {exc.args[0]!r}") from exc


def host_facts(measured: Optional[Mapping[str, Any]]) -> HostFacts:
    for key in ("host_matmul_gflops", "host_memory_gbps"):
        if not measured or key not in measured:
            raise MissingFact(f"host: no measured `{key}` on the hardware profile (tools/profile_facts.py)")
    s = measured.get("storage_read_gbps")
    return HostFacts(matmul_flops=float(measured["host_matmul_gflops"]) * 1e9,
                     memory_bps=float(measured["host_memory_gbps"]) * 1e9,
                     storage_bps=float(s) * 1e9 if s is not None else None)


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


def plan_seconds(strategy: str, allocations: Mapping[str, Any], works: Mapping[str, Work], uses: Mapping[str, int],
                 devices: Mapping[str, DeviceFacts], host: Optional[HostFacts], dtype: str,
                 phases: Optional[Mapping[str, int]] = None, prompt: Optional[Mapping[str, Work]] = None) -> float:
    """The plan's estimated seconds for the request. `devices`: {device string ("cuda:0"): its facts}; `uses`: {component:
    runs per request}; `phases`: {component: phases that load it} for the strategies that load per phase (1 when
    absent); `prompt`: {component: the work of its one prompt pass} for a generating component whose `works` entry is
    the work of ONE token — the prompt pass runs once more, over the same loaded weights."""
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
        total += loads * w.weight_bytes / f.h2d_bps              # the weights cross once per load, never per token
    return total


def uses_per_request(flow: Mapping[str, Any], steps: Optional[int], tokens: Optional[int]) -> Dict[str, tuple]:
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
    gen = flow.get("generation") or {}
    token_comps = [c for c in (gen.get("lm_component"), gen.get("head_component")) if c]
    token_comps += [s["component"] for s in (flow.get("stages") or []) if s.get("execution") == "autoregressive"]
    if token_comps:
        if not tokens:
            raise MissingFact(f"the flow generates with {token_comps} and the request names no token count")
        for comp in token_comps:
            out[comp] = ("token", int(tokens))
    return out
