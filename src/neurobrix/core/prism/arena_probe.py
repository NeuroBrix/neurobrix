"""The Triton arena's hold over a component's walked activations, MEASURED on this device (the owner, 2026-11-02 15:01):
the cache's smallest container, its first component run whole twice through the Triton executor on zero inputs at its
traced extents (no flow, no prompt), the first run seating what runs once (sweeps, recordings), every free byte then
given back, the allocator's peak window around the second; hold = the window's peak - its start, divided by the walk
Prism prices (`ComponentMemory`, the KV cache apart). `python -m neurobrix.core.prism.arena_probe <dir> <component>
<hardware id> [scale]` runs one probe in its own process and prints `NBX-ARENA-PROBE {json}`. Triton branch: no torch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MARK = "NBX-ARENA-PROBE "
#: the allocator's refusal, word for word (nbxtensor.memory.allocator: "the message is unchanged to the byte")
_OOM_TEXT = "GPU malloc failed"


class ProbeOutOfMemory(RuntimeError):
    """The probe's request did not fit the device: the measure's own upper bound, not a failure of the measure."""


class ProbeUncertified(ProbeOutOfMemory):
    """The probe's request formed a kernel key the certified directory does not serve: the probe runs certified-only
    and never sweeps (a probe that swept took a fresh Mac's first run to 42 minutes, 2026-11-07), so the largest
    request whose keys are all certified is its bound — the same bound as the device's own refusal."""


_SPATIAL = ("height", "width")


def scaled_names(dag: Dict[str, Any]) -> Tuple[str, ...]:
    """The symbols a probe of this graph scales — in its inputs AND in the request Prism prices, the same ones, so the
    hold and the priced walk grow together: the spatial symbols when its inputs carry them; else its sequence when it
    has ONE (one seq_len trace: the request names one length); else none (the probe runs at its traced extents only).
    Scaling the two last axes of every input instead grew a decoder's batch and hidden width while its priced walk
    stayed at x1: a fresh Mac home measured whisper-large-v3-turbo's decoder at 28.27 where the lab's 1.477 holds."""
    syms = [v for v in ((dag.get("symbolic_context") or {}).get("symbols") or {}).values()
            if str(v.get("source", "")).startswith("input::")]
    if any(v.get("name") in _SPATIAL for v in syms):
        return _SPATIAL
    if len({v.get("trace_value") for v in syms if v.get("name") == "seq_len"}) == 1:
        return ("seq_len",)
    return ()


def _component_bytes(root: Path, component: str) -> int:
    return sum(f.stat().st_size for f in (root / "components" / component).rglob("*.safetensors"))


def smallest_container(cache: Path) -> Tuple[Path, str]:
    """(container, neural component) the factor is measured on — never a model named in code: the smallest component
    by weight bytes whose inputs carry spatial symbols (the probe's request then grows the walk it prices); else, in
    the smallest container that has a component with one sequence length, its component of the most ops (a model's
    walk — a lone projection such as an lm_head holds about its output, 0.99 on a V100 where the decoder holds 1.38);
    else the cache's smallest component, probed at its traced extents."""
    from neurobrix.nbx.container import NBXContainer
    pairs = []
    for d in sorted(Path(cache).iterdir()):
        if not ((d / "manifest.json").is_file() and (d / "components").is_dir()):
            continue
        comps = [(c.name, _component_bytes(d, c.name)) for c in NBXContainer.load(str(d)).get_neural_components()]
        total = sum(n for _c, n in comps)
        pairs += [(n, total, d, c) for c, n in comps if n > 0 and (d / "components" / c / "graph.json").is_file()]
    if not pairs:
        raise RuntimeError(f"ZERO FALLBACK: the arena factor is measured on a container of the cache and {cache} holds "
                           f"none: import one model first (neurobrix import <org>/<name>)")
    pairs.sort(key=lambda t: (t[0], str(t[2]), t[3]))
    sequence = []
    for n, total, d, c in pairs:                # smallest first: a graph is read only until a spatial one is found
        dag = json.loads((d / "components" / c / "graph.json").read_text())
        names = scaled_names(dag)
        if names == _SPATIAL:
            return d, c
        if names:
            sequence.append((total, -len(dag.get("ops") or {}), str(d), c, d))
    if sequence:
        best = min(sequence)
        return best[4], best[3]
    return pairs[0][2], pairs[0][3]


def _inputs(dag: Dict[str, Any], scale: int) -> Dict[str, Any]:
    """Every graph input at its traced extent and dtype, zero (an index of any table), each dimension bound to a symbol
    of `scaled_names` x `scale` — the dimensions the request Prism prices scales, and no other."""
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import parse_dtype
    syms = (dag.get("symbolic_context") or {}).get("symbols") or {}
    grow = set(scaled_names(dag))
    out = {}
    for tid in dag.get("input_tensor_ids") or []:
        meta = dag["tensors"][tid]
        dims = [int(x) for x in meta["shape"]]
        sym_dims = ((meta.get("symbolic_shape") or {}).get("dims") or [])
        for i, sd in enumerate(sym_dims[:len(dims)]):
            if isinstance(sd, dict) and (syms.get(sd.get("id")) or {}).get("name") in grow:
                dims[i] *= scale
        out[str(tid).split("::", 1)[1]] = NBXTensor.zeros(tuple(dims), parse_dtype(meta["dtype"]), "cuda:0")
    return out


def measure(container_dir: str, component: str, hardware_id: str, scale: int = 1) -> Dict[str, Any]:
    """One probe: the component's hold and its priced walk, in bytes. The request = the graph's own traced extents,
    named by its symbols, scaled like the inputs: Prism prices the binding the run makes."""
    from neurobrix.nbx.container import NBXContainer
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.core.prism.profiler import InputConfig
    from neurobrix.core.prism.solver import PrismSolver, ComponentAllocation
    from neurobrix.core.runtime.factory import ExecutorFactory
    from neurobrix.triton.dtype import declare_hardware_profile
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    container, profile = NBXContainer.load(container_dir), load_profile(hardware_id)
    declare_hardware_profile(profile)
    dag = json.loads((Path(container_dir) / "components" / component / "graph.json").read_text())
    field = {"height": "height", "width": "width", "time": "chunk_frames", "seq_len": "seq_len", "batch": "batch_size"}
    grow = set(scaled_names(dag))
    req = {field[i["name"]]: i["trace_value"] * (scale if i["name"] in grow else 1)
           for i in ((dag.get("symbolic_context") or {}).get("symbols") or {}).values()
           if i.get("name") in field and isinstance(i.get("trace_value"), int)}
    solver, dev = PrismSolver(), profile.devices[0]
    solver._mode = "triton"
    dtypes = solver._resolve_component_dtypes(container.get_neural_components(), profile, container)
    mem = solver._compute_memory(container, [c for c in container.get_neural_components() if c.name == component],
                                 InputConfig(**req), dtypes[component], profile, dtypes)[component]
    walked = int(mem.activation_bytes) - int(getattr(mem, "kv_bytes", 0) or 0)
    if walked <= 0:
        raise RuntimeError(f"ZERO FALLBACK: {Path(container_dir).name}:{component} prices no walk to divide by")
    alloc = ComponentAllocation(name=component, devices=["cuda:0"], dtype=dtypes[component],
                                memory_mb=mem.weight_bytes / (1 << 20), architecture=dev.architecture,
                                vendor=getattr(dev.brand, "value", dev.brand))
    executor = ExecutorFactory.create(component, alloc, str(container_dir), dag, mode="triton")
    executor.load_weights(str(container_dir), component)
    executor.run(_inputs(dag, scale))
    DA.device_synchronize(0)
    if DA._range_vmm(0) is not None:
        DA._range_reclaim(0, None)
    DA._pool_flush(0)
    inputs = _inputs(dag, scale)
    token = DA.open_window(0)
    executor.run(inputs)
    DA.device_synchronize(0)
    start, peak = DA.window_start(token), DA.close_window(0, token)
    return {"container": Path(container_dir).name, "component": component, "hold_bytes": int(peak) - int(start),
            "walked_bytes": walked}


def reloads(container_dir: str, component: str, hardware_id: str, engine: str, n: int = 4) -> Dict[str, Any]:
    """The reload a streamed run times (`layer_streaming`: `load_weights` after `unload_weights`), on `component` n times
    after its first load, each handed to that run's own writer (`_note_reload`: three steady reloads, their median)."""
    import time
    from neurobrix.nbx.container import NBXContainer
    from neurobrix.core.prism import autodetect as AD
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.core.prism.solver import PrismSolver, ComponentAllocation
    from neurobrix.core.runtime.factory import ExecutorFactory
    from neurobrix.core.strategies import layer_streaming as LS
    container, profile, AD.ACTIVE_HARDWARE_ID = NBXContainer.load(container_dir), load_profile(hardware_id), hardware_id
    if engine == "triton":
        from neurobrix.triton.dtype import declare_hardware_profile
        declare_hardware_profile(profile)
    solver, dev = PrismSolver(), profile.devices[0]
    solver._mode = engine
    dtype = solver._resolve_component_dtypes(container.get_neural_components(), profile, container)[component]
    alloc = ComponentAllocation(name=component, devices=["cuda:0" if engine == "triton" else dev.get_device_string()],
                                dtype=dtype, memory_mb=0.0, architecture=dev.architecture,
                                vendor=getattr(dev.brand, "value", dev.brand))
    dag = json.loads((Path(container_dir) / "components" / component / "graph.json").read_text())
    executor = ExecutorFactory.create(component, alloc, str(container_dir), dag, mode=engine)
    executor.load_weights(container_dir, component)
    draws = []
    for _ in range(n):
        executor.unload_weights()
        t0 = time.perf_counter()
        executor.load_weights(container_dir, component)
        draws.append((LS._weights_bytes(executor), time.perf_counter() - t0))
        LS._note_reload(executor, container_dir, component, *draws[-1])
    return {"container": Path(container_dir).name, "component": component, "draws": draws}


def child(container_dir: str, component: str, hardware_id: str, scale) -> Dict[str, Any]:
    """One probe in its own process (a clean allocator). Its own exception comes back by name: a device that cannot
    hold the request raises `ProbeOutOfMemory`, anything else a RuntimeError carrying the child's error — never the
    tail of its stderr (the interpreter's shutdown noise hid the cause on a fresh Mac home, 2026-11-07)."""
    env = dict(os.environ, NBX_AUTOTUNE_CERTIFIED_ONLY="1")        # never a sweep: a missing key bounds the measure
    r = subprocess.run([sys.executable, "-m", __name__, container_dir, component, hardware_id, str(scale)],
                       capture_output=True, text=True, env=env)
    line = next((l for l in r.stdout.splitlines() if l.startswith(MARK)), None)
    got = json.loads(line[len(MARK):]) if line is not None else None
    where = f"the arena probe of {Path(container_dir).name}:{component} x{scale}"
    if got is not None and "error" in got:
        kind = ProbeUncertified if got.get("uncertified") else ProbeOutOfMemory if got.get("oom") else RuntimeError
        raise kind(f"{where} failed: {got['error']}")
    if r.returncode != 0 or got is None:
        raise RuntimeError(f"{where} failed: {(r.stderr or r.stdout)[-400:]}")
    return got


def factor_on_card(usable_bytes: int, weight_bytes: int, run, reps: int, steady: float,
                   grows: bool = True) -> Tuple[Optional[float], list]:
    """The arena's factor over a LARGE walk: the request doubled while the next (its walk four-fold: two axes doubled)
    still fits `usable_bytes` with the weights at the ratio just measured, then `reps` runs at the largest, their
    median kept when they agree within `steady` (HAT-S on a V100-16GB: 3.50 at x1, 2.39 at x2, 1.48 at x4 — a fixed
    hold a small walk over-weights); (None, runs) when they disagree. `run(scale)` -> {hold_bytes, walked_bytes}."""
    scale, last = 1, run(1)
    while grows and weight_bytes + 4 * last["hold_bytes"] <= usable_bytes:     # a graph with nothing to scale stays x1
        try:
            nxt = run(scale * 2)
        except ProbeOutOfMemory:
            break               # the device's own bound (the 4x prediction under-sized it): the largest that fitted
        scale, last = scale * 2, nxt
    runs = [last] + [run(scale) for _ in range(reps - 1)]
    ratios = sorted(r["hold_bytes"] / r["walked_bytes"] for r in runs)
    return (ratios[len(ratios) // 2] if ratios[-1] <= ratios[0] * (1 + steady) else None), runs


if __name__ == "__main__":
    if sys.argv[4:5] and sys.argv[4] in ("triton", "compiled"):     # the reload probe: `<dir> <component> <hw> <engine>`
        print(MARK + json.dumps(reloads(*sys.argv[1:5])), flush=True)
    else:
        try:
            out = measure(sys.argv[1], sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:5]))
        except Exception as exc:          # said to the parent by name: the cause, and whether the device was too small
            print(MARK + json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:600]}", "oom": _OOM_TEXT in str(exc),
                                     "uncertified": type(exc).__name__ == "KeyNotCertified"
                                     or "CERTIFIED-ONLY" in str(exc)}), flush=True)
            sys.exit(3)
        print(MARK + json.dumps(out), flush=True)
