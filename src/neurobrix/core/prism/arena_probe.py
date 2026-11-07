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


def smallest_container(cache: Path) -> Tuple[Path, str]:
    """(the cache's smallest container by weight bytes, its first neural component) — never a model named in code."""
    sizes = {d: sum(f.stat().st_size for f in (d / "components").rglob("*.safetensors"))
             for d in sorted(Path(cache).iterdir()) if (d / "manifest.json").is_file() and (d / "components").is_dir()}
    sizes = {d: n for d, n in sizes.items() if n > 0}
    if not sizes:
        raise RuntimeError(f"ZERO FALLBACK: the arena factor is measured on a container of the cache and {cache} holds "
                           f"none: import one model first (neurobrix import <org>/<name>)")
    from neurobrix.nbx.container import NBXContainer
    root = min(sizes, key=sizes.get)
    return root, NBXContainer.load(str(root)).get_neural_components()[0].name


def _inputs(dag: Dict[str, Any], scale: int) -> Dict[str, Any]:
    """Every graph input at its traced extent (its two last axes x `scale`) and dtype, zero (an index of any table)."""
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import parse_dtype
    out = {}
    for tid in dag.get("input_tensor_ids") or []:
        meta = dag["tensors"][tid]
        dims = [int(x) for x in meta["shape"]]
        dims[-2:] = [d * scale for d in dims[-2:]] if len(dims) >= 2 else dims[-2:]
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
    req = {field[i["name"]]: i["trace_value"] * (scale if i["name"] in ("height", "width") else 1)
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
    r = subprocess.run([sys.executable, "-m", __name__, container_dir, component, hardware_id, str(scale)],
                       capture_output=True, text=True, env=dict(os.environ))
    line = next((l for l in r.stdout.splitlines() if l.startswith(MARK)), None)
    got = json.loads(line[len(MARK):]) if line is not None else None
    where = f"the arena probe of {Path(container_dir).name}:{component} x{scale}"
    if got is not None and "error" in got:
        raise (ProbeOutOfMemory if got.get("oom") else RuntimeError)(f"{where} failed: {got['error']}")
    if r.returncode != 0 or got is None:
        raise RuntimeError(f"{where} failed: {(r.stderr or r.stdout)[-400:]}")
    return got


def factor_on_card(usable_bytes: int, weight_bytes: int, run, reps: int, steady: float) -> Tuple[Optional[float], list]:
    """The arena's factor over a LARGE walk: the request doubled while the next (its walk four-fold: two axes doubled)
    still fits `usable_bytes` with the weights at the ratio just measured, then `reps` runs at the largest, their
    median kept when they agree within `steady` (HAT-S on a V100-16GB: 3.50 at x1, 2.39 at x2, 1.48 at x4 — a fixed
    hold a small walk over-weights); (None, runs) when they disagree. `run(scale)` -> {hold_bytes, walked_bytes}."""
    scale, last = 1, run(1)
    while weight_bytes + 4 * last["hold_bytes"] <= usable_bytes:
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
            print(MARK + json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:600]}", "oom": _OOM_TEXT in str(exc)}),
                  flush=True)
            sys.exit(3)
        print(MARK + json.dumps(out), flush=True)
