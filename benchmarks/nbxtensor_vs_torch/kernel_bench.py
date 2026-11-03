"""Kernel level of the nbxtensor benchmark: each instance (`instances.py`) run by three contenders on one device.

Contenders, each through the engine's own per-op entry so the call is the one a run makes:
- `torch_eager`: `NativeATenDispatcher.dispatch` (the compiled engine's ATen call) on the device;
- `torch_compile` (`--compile-modes`, default "default,max-autotune"): `torch.compile` of that same call;
- `nbxtensor`: `TritonSequentialDispatcher.dispatch` (the Triton engine's kernel call, which becomes nbxtensor).

Two environments, one pass (the supervisor, 2026-11-02 11:55 and 21:35): torch runs in an oracle environment, never
in the library's. A coordinator (`--python-torch`, `--python-nbxtensor`) drives two persistent workers, instance by
instance: the nbxtensor worker (the library's torch-free environment) runs the call and leaves ONLY its output, at its
own width, in a scratch file; the torch worker computes the float64 reference in memory, times torch eager and each
torch.compile mode, and compares every contender with the reference TILE BY TILE as it goes — the nbxtensor output
read through a memory map; the scratch file is removed at once. Nothing is kept but the result rows: the float64
reference is never written (the 02/11 runs filled 227 GB on the Dell and 21 GB here with whole references).

The guard: an instance whose operands, held at once with the float64 reference this pass needs, exceed the op budget
Prism grants an op on this device (`tiling_sizes.op_budget_bytes` of the profile's device memory: the library's
`op_level.op_budget_card_fraction`, never a threshold written here) is not run whole — its row says `out_of_class`
with the numbers and the cut keys the census certifies for its sites (a cut op runs in Prism's slices, which the bench
does not replay). Never killed, never skipped in silence.

Protocol (README): the machine runs nothing else; the first call is timed APART (it carries compilation, autotune,
first-touch allocation) and reported as `first_ms`; then `--warmup` untimed calls; then `--reps` timed calls, each
bracketed by a device synchronize; the median and the 10th/90th percentiles are written with N. Every contender's
output is compared with the float64 reference computed by torch on the host from the same inputs: relative error =
max|out - ref| / max|ref|, held to the output dtype's bound (`TOLERANCE`); a contender over it is `wrong` — a faster
wrong result is a red, never a win. Peak memory: the bytes the contender's allocator held above its level before the
call (CUDA: `max_memory_allocated`; MPS keeps no peak: its driver sampled through the timed calls, said `sampled`;
nbxtensor: `DeviceAllocator`'s own peak counter).

One TSV per device, appended as each instance finishes (a stopped run keeps what it measured; a rerun skips the
instances its side already wrote).

    <library python> benchmarks/nbxtensor_vs_torch/kernel_bench.py --hardware <profile id> \\
        --instances <campaign>/instances.jsonl --out <campaign>/kernel.tsv --scratch <dir> --registry <dir> \\
        --python-torch <oracle python> --python-nbxtensor <library python>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The relative error bound per output dtype: a few units in the last place of the dtype's mantissa, scaled for the
# long reductions the catalogue's GEMMs and convolutions accumulate (K up to ~16k). Stated once, for every contender.
TOLERANCE = {"float32": 2e-3, "float16": 2e-2, "bfloat16": 6e-2}
FIELDS = ["key", "op_type", "shapes", "dtypes", "sites", "contender", "status", "first_ms", "compile_ms", "median_ms",
          "p10_ms", "p90_ms", "n", "rel_err", "mean_err", "tolerance", "peak_mb", "peak_how", "note"]
# the elements compared at once: the comparison holds one tile of each side in float64, never the whole
TILE = 1 << 22


class _ArgResolver:
    """The engine's own argument resolution (`GraphExecutor._resolve_sequential_arg`), bound to a store of the
    contender's tensors — one resolver, both engines."""

    def __init__(self):
        from neurobrix.core.runtime.graph_executor import GraphExecutor
        self._fn = GraphExecutor._resolve_sequential_arg

    def _resolve_sequential_arg(self, *a, **k):
        return self._fn(self, *a, **k)

    def args(self, raw, store, symbols, dispatcher):
        return [self._resolve_sequential_arg(a, store, symbols, dispatcher) for a in (raw or [])]


def _symbols(binding: dict):
    from neurobrix.triton.symbols import SymbolResolver
    res = SymbolResolver({"symbols": {}})
    for sid, v in binding.items():
        res._bindings[sid] = int(v)
    return res


def feeds(inst: dict) -> list:
    """[(tensor id, shape, dtype)] the contenders materialise: the op's inputs, or — when the graph forms an input as a
    strided view — the root of its view chain, the chain then replayed (`apply_prelude`)."""
    if inst.get("roots") is not None:
        return [(r["id"], r["shape"], r["dtype"]) for r in inst["roots"]]
    return list(zip(inst.get("inputs") or [None] * len(inst["shapes"]), inst["shapes"], inst["dtypes"]))


def apply_prelude(inst: dict, store: dict, resolve, dispatch) -> None:
    """The metadata ops (views) the graph applies between the roots and the op's inputs, untimed, each through the
    contender's own dispatcher: the op then reads the tensor the engine hands it — a transposed weight as a view."""
    for p in inst.get("prelude") or []:
        args = resolve(p["args"], store)
        store[p["output"]] = dispatch(p, args, {**p["attributes"], "args": p["args"]})


def host_inputs(inst: dict, seed: int = 0) -> list:
    """One numpy array per feed (`feeds`), deterministic: floats ~ N(0, 1) scaled by 1/sqrt(last dim) (a GEMM's K, so half
    outputs stay in range); integers below the SMALLEST dim of the op's first floating input — in range for whichever
    axis they index (an embedding's table rows: [10, 32] x [1, 36000] drew up to 32 for 10 rows on 02/11, a device
    assert on both engines that poisoned the CUDA context of every later instance); booleans fair."""
    rng = np.random.default_rng(seed)
    out = []
    first = next((s for s, d in zip(inst["shapes"], inst["dtypes"]) if "float" in d), None)
    float_extent = min((int(x) for x in first), default=2) if first else 2
    for _tid, shape, dt in feeds(inst):
        if "float" in dt:
            k = shape[-1] if shape else 1
            out.append(np.asarray(rng.standard_normal(shape) / np.sqrt(max(k, 1)), dtype=np.float32).reshape(shape))
        elif "bool" in dt:
            out.append(np.asarray(rng.integers(0, 2, size=shape), dtype=bool).reshape(shape))
        else:
            out.append(np.asarray(rng.integers(0, max(int(float_extent), 2), size=shape), dtype=np.int64).reshape(shape))
    return out


def _torch_dtype(name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
            "float64": torch.float64, "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}[name]


def _nbx_dtype(name: str):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    return getattr(NBXDtype, name)


def _to_numpy(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if type(x).__module__.startswith("torch"):   # a torch tensor (the torch side; nbxtensor is never imported there)
        return x.detach().to("cpu").double().numpy()
    return np.asarray((x.float() if x.is_floating_point() else x).numpy(), dtype=np.float64)


_SHORT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "fp64": "float64"}


def _dtype_of(x) -> str:
    if isinstance(x, (list, tuple)):
        x = x[0]
    name = str(getattr(x, "dtype", "?")).replace("torch.", "").replace("NBXDtype.", "")
    return _SHORT.get(name, name)


def _bound(dtypes) -> float | None:
    """The tolerance of the narrowest floating dtype the op reads or writes (its rounding bounds the result)."""
    got = [TOLERANCE[d] for d in dtypes if d in TOLERANCE]
    return max(got) if got else None


def _stats(times):
    t = sorted(times)
    q = statistics.quantiles(t, n=10) if len(t) >= 2 else [t[0]] * 9
    return statistics.median(t), q[0], q[-1]



class _TorchDevice:
    """The device torch runs on (the torch side), its synchronize and its cache release."""

    def __init__(self):
        import torch
        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.name = "cuda:0" if self.cuda else "mps"

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()
        else:
            self.torch.mps.synchronize()

    def release(self):
        import gc
        gc.collect()
        if self.cuda:
            self.torch.cuda.empty_cache()
        else:
            self.torch.mps.empty_cache()


class _NbxDevice:
    """The device the Triton engine runs on (the nbxtensor side, no torch), its synchronize and its pool release."""

    def nbx_sync(self):
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.device_synchronize(0)

    def release(self):
        import gc
        gc.collect()
        from nbxtensor.memory.allocator import DeviceAllocator
        self.nbx_sync()
        DeviceAllocator.empty_cache_pool()


def _time(call, sync, warmup: int, reps: int):
    sync()
    t0 = time.perf_counter()
    out = call()
    sync()
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        call()
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return out, first, times


def reference(inst, arrays, attrs_resolver):
    """float64 on the host through the compiled engine's own ATen call."""
    import torch
    from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher
    disp = NativeATenDispatcher(device="cpu", compute_dtype=torch.float64)
    store = {}
    for (tid, _s, d), a in zip(feeds(inst), arrays):
        t = torch.from_numpy(a)
        # the inputs the contenders read, rounded to their dtype, then widened: the error measured is the
        # computation's, never the rounding of the inputs both contenders share
        store[tid] = t.to(_torch_dtype(d)).double() if t.is_floating_point() else t
    syms, attrs_ = _symbols(inst["symbols"]), _NativeAttrs(disp)
    apply_prelude(inst, store, lambda raw, st: attrs_resolver.args(raw, st, syms, attrs_),
                  lambda p, args, at: disp.dispatch(p["op_type"], args, at))
    args = attrs_resolver.args(inst["args"], store, syms, attrs_)
    return disp.dispatch(inst["op_type"], args, {**inst["attributes"], "args": inst["args"]})


class _NativeAttrs:
    """The compiled dispatcher's attribute reader under the name the resolver calls."""

    def __init__(self, disp):
        self.resolve_attr = disp._resolve_attr_value


def run_torch(inst, arrays, dev, res, warmup, reps, compile_mode=None):
    import torch
    from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher
    disp = NativeATenDispatcher(device=dev.name)
    # torch's kernels take one floating dtype per call: inputs the Triton engine reads at mixed widths (an fp32
    # island beside half weights) are given to torch at the widest of them, as a torch user would write it
    floats = [d for d in inst["dtypes"] if "float" in d]
    widest = max(floats, key=lambda d: {"float16": 0, "bfloat16": 0, "float32": 1, "float64": 2}[d]) if floats else None
    store = {tid: torch.from_numpy(a).to(dev.name, dtype=_torch_dtype(widest if "float" in d else d))
             for (tid, _s, d), a in zip(feeds(inst), arrays)}
    syms, attrs_ = _symbols(inst["symbols"]), _NativeAttrs(disp)
    apply_prelude(inst, store, lambda raw, st: res.args(raw, st, syms, attrs_),
                  lambda p, args, at: disp.dispatch(p["op_type"], args, at))
    args = res.args(inst["args"], store, syms, attrs_)
    attrs = {**inst["attributes"], "args": inst["args"]}

    def eager():
        return disp.dispatch(inst["op_type"], args, attrs)
    call = eager
    if compile_mode is not None:
        compiled = torch.compile(lambda: disp.dispatch(inst["op_type"], args, attrs), mode=compile_mode,
                                 dynamic=False, fullgraph=False)
        mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)

        def call():
            # max-autotune replays CUDA graphs: each run begins a step, or a later replay overwrites the output
            if mark is not None and dev.cuda:
                mark()
            return compiled()
    peak_how = "max_memory_allocated" if dev.cuda else "sampled driver"
    if dev.cuda:
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out, first, times = _time(call, dev.sync, warmup, reps)
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    else:
        from neurobrix.core.device_utils import SampledDevicePeak
        base = torch.mps.driver_allocated_memory()
        s = SampledDevicePeak(torch.mps.driver_allocated_memory).start()
        out, first, times = _time(call, dev.sync, warmup, reps)
        peak = (s.stop() - base) / 2 ** 20
    # the output compared is one more call's, cloned at once: a graph replay reuses its output buffer
    out = call()
    dev.sync()
    out = [o.clone() for o in out] if isinstance(out, (list, tuple)) else out.clone()
    return out, first, times, peak, peak_how


def release_compile_caches(dev) -> None:
    """torch.compile's caches given back after each instance (its graphs, guards and generated kernels): held across
    1 682 instances they pressed the unified memory of the Mac until the session was stopped (02/11 19:44)."""
    import torch
    torch._dynamo.reset()
    dev.release()


def _engine_dtype_facts(inst) -> dict:
    """The dtype facts the engine's dispatcher runs this op under, as the census derived its key (instances.py): the
    component's compute dtype, the graph's traced dtype and the precision contract's role for the op. They were once
    guessed from the container's record of the op's output — the TRACE dtype: on Apple the conv [20, 4, 90, 160] ran
    in fp16 and the VAE convs stored fp32 where the engine runs bf16, keys no certificate holds, swept at runtime (11
    of the first 13 instances, 03/11 00:43). An instance file without them is refused, never guessed."""
    missing = [k for k in ("compute_dtype", "graph_dtype", "contract", "op_uid") if k not in inst]
    if missing:
        raise KeyError(f"instance {inst.get('key')}: no {', '.join(missing)} — an instances file from before "
                       f"03/11; rebuild it with instances.py")
    c = inst["contract"]
    uid = inst["op_uid"]
    return {"compute": inst["compute_dtype"], "graph": inst["graph_dtype"], "uid": uid,
            "contract": (bool(c["safe"]), {uid} if c["island"] == "fp32" else set(),
                         {uid} if c["island"] == "narrow" else set())}


def engine_dispatcher(inst, stores_fp64: bool):
    """The Triton dispatcher the engine runs this op through: its compute dtype, graph dtype and precision contract."""
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    facts = _engine_dtype_facts(inst)
    disp = TritonSequentialDispatcher(device_idx=0, compute_dtype=_nbx_dtype(facts["compute"]),
                                      precision_contract=facts["contract"], graph_dtype=facts["graph"],
                                      stores_fp64=stores_fp64)
    return disp, facts


def run_nbx(inst, arrays, dev, res, warmup, reps):
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.tensor import NBXTensor
    disp, facts = engine_dispatcher(inst, _stores_fp64())
    disp.symbol_resolver = _symbols(inst["symbols"])
    store = {}
    for (tid, _s, d), arr in zip(feeds(inst), arrays):
        t = NBXTensor.from_numpy(np.ascontiguousarray(arr))      # the host array in its own container
        store[tid] = t.to(_nbx_dtype(d)) if "float" in d and d != "float32" else t
    with disp:
        apply_prelude(inst, store, lambda raw, st: res.args(raw, st, disp.symbol_resolver, disp),
                      lambda p, args, at: disp.dispatch(p["op_type"], args, at, op_uid=p["op_uid"]))
        args = res.args(inst["args"], store, disp.symbol_resolver, disp)
        attrs = {**inst["attributes"], "args": inst["args"]}
        record = {"output_dtypes": inst.get("output_dtypes"), "op_type": inst["op_type"]}

        def call():
            return disp.dispatch(inst["op_type"], args, attrs, op_uid=facts["uid"], op_record=record)
        out, first, times = _time(call, dev.nbx_sync, warmup, reps)
        # the peak of ONE call with the pool emptied first, so every byte it takes reaches the driver counter
        dev_idx = 0
        dev.nbx_sync()
        DeviceAllocator.empty_cache_pool()
        base = DeviceAllocator._cuda_live_bytes.get(dev_idx, 0)
        DeviceAllocator._cuda_peak_bytes[dev_idx] = base
        call()
        dev.nbx_sync()
        peak = (DeviceAllocator._cuda_peak_bytes.get(dev_idx, 0) - base) / 2 ** 20
    return out, first, times, peak, "DeviceAllocator peak, pool emptied"


def _stores_fp64() -> bool:
    """The Triton branch's fp64 storage, as the census reads it (`derived_census._triton_fp64`)."""
    sys.path.insert(0, str(REPO / "tools"))
    import derived_census as DC
    return bool(DC._triton_fp64())




def _row(inst, name) -> dict:
    return {"key": inst["key"], "op_type": inst["op_type"], "shapes": json.dumps(inst["shapes"]),
            "dtypes": ",".join(inst["dtypes"]), "sites": len(inst["sites"]), "contender": name}


_ITEMSIZE = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4, "bool": 1, "uint8": 1}


def _elems(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def instance_bytes(inst) -> int:
    """What the torch worker holds at once for an instance — the larger of the two workers, which never overlap (the
    library's releases its pool and arrays before the torch side starts): per input element the host array (float32,
    `host_inputs`), its float64 copy (the reference) and its device copy at its width; per output element (no wider
    than the largest operand, for every op of the catalogue: an elementwise, a reduction, a GEMM or a convolution at
    the request's shapes) the float64 result, the device output at its width and the host copy compared (float32)."""
    els = [_elems(s) for s in inst["shapes"]]
    inputs = sum(e * (4 + 8 + _ITEMSIZE.get(d, 8)) for e, d in zip(els, inst["dtypes"]))
    widest = max((_ITEMSIZE.get(d, 8) for d in inst["dtypes"]), default=8)
    return inputs + (max(els) if els else 0) * (8 + widest + 4)


def device_budget_bytes(profile) -> int:
    """The op budget Prism grants one op on this device: `op_budget_bytes` of the profile's device memory (nothing
    resident beside it): the library's `op_level.op_budget_card_fraction`, read with its source."""
    from nbxtensor.ops.tiling_sizes import op_budget_bytes
    devices = getattr(profile, "devices", None) or (profile.get("devices") if isinstance(profile, dict) else None)
    if not devices:
        raise RuntimeError("ZERO FALLBACK: the hardware profile declares no device: the guard has no memory to read")
    d0 = devices[0]
    mb = getattr(d0, "memory_mb", None) or (d0.get("memory_mb") if isinstance(d0, dict) else None)
    if not mb:
        raise RuntimeError("ZERO FALLBACK: the profile's device declares no memory_mb: the guard has no memory to read")
    return op_budget_bytes(int(mb) * 2 ** 20, 0)


def census_cut_keys(inst, census_rows) -> list:
    """The kernel keys the census certifies for this instance's sites (model/component/op uid): a cut op's are its
    slices' — named in the row of an instance too large to run whole."""
    sites = set(inst["sites"])
    out = []
    for row in census_rows:
        model = row.get("model") or ""
        for op in row.get("ops") or []:
            if not isinstance(op, str) or not model:         # a row naming no op or no model names no site
                continue
            if f"{model}/{op}" in sites or any(s.endswith("/" + op) and s.startswith(model + "/") for s in sites):
                out.append(f"{row.get('kernel', '?').rsplit('.', 1)[-1]} {row.get('key')}")
    return sorted(set(out))


def out_of_class_row(inst, need: int, budget: int, cut_keys) -> dict:
    row = _row(inst, "*")
    row.update(status="out_of_class",
               note=(f"operands + float64 reference {need / 2**20:.0f} MB > the op budget {budget / 2**20:.0f} MB "
                     f"(profile device memory x op_level.op_budget_card_fraction): not run whole; "
                     + ("the census certifies it cut: " + "; ".join(cut_keys[:4]) if cut_keys
                        else "no cut certified by the census for its sites")))
    return row


def compare_tiles(got_flat, ref_flat):
    """(max |got - ref| / max |ref|, mean |got - ref| / max |ref|) over the two flat arrays, read TILE elements at a
    time in float64 — `got_flat` may be a memory map: neither side is ever held whole in float64."""
    n = int(ref_flat.shape[0])
    if int(got_flat.shape[0]) != n:
        return float("inf"), float("inf")
    mx = sm = rmax = 0.0
    for a in range(0, n, TILE):
        g = np.asarray(got_flat[a:a + TILE], dtype=np.float64)
        r = np.asarray(ref_flat[a:a + TILE], dtype=np.float64)
        d = np.abs(g - r)
        mx = max(mx, float(d.max()) if d.size else 0.0)
        sm += float(d.sum())
        rmax = max(rmax, float(np.abs(r).max()) if r.size else 0.0)
    denom = max(rmax, 1e-30)
    return mx / denom, (sm / max(n, 1)) / denom


def judge_row(row, inst, od: str, first, times, peak, how, errs, ref_note, compile_ms=None) -> dict:
    """The row of one contender: its times, and its output against the float64 reference (`errs` from
    `compare_tiles`, None when there is no reference)."""
    med, p10, p90 = _stats(times)
    # the narrowest of what the op reads, writes, and what the container records it writes (the precision
    # contract rounds an output to the traced dtype: an fp32 island's result handed back in bf16)
    tol = _bound(inst["dtypes"] + [od] + [str(d) for d in (inst.get("output_dtypes") or [])])
    status, err, mean = "ok", None, None
    if errs is not None:
        err, mean = errs
        if not err <= (tol if tol is not None else 0.0):
            status = "wrong"
    row.update(status=status, first_ms=f"{first:.3f}", median_ms=f"{med:.4f}", p10_ms=f"{p10:.4f}",
               p90_ms=f"{p90:.4f}", n=len(times), rel_err="" if err is None else f"{err:.3e}",
               mean_err="" if mean is None else f"{mean:.3e}",
               compile_ms="" if compile_ms is None else f"{compile_ms:.3f}",
               tolerance="" if tol is None else tol, peak_mb=f"{peak:.3f}", peak_how=how,
               note=ref_note or ("" if tol is not None else f"no floating dtype to bound ({od}): exact match asked"))
    return row


def _host_flat(x):
    """A contender's output as a flat host array at its OWN width (float32 for the halves, exact), never float64."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if type(x).__module__.startswith("torch"):
        t = x.detach().to("cpu")
        return (t.float() if t.is_floating_point() else t).numpy().reshape(-1)
    a = (x.float() if x.is_floating_point() else x).numpy()
    return np.asarray(a).reshape(-1)


# ---------------------------------------------------------------------------------------------------- the workers

def nbx_work(inst, dev, res, warmup, reps, scratch: Path) -> dict:
    """The library's side of one instance: the call timed, its output left at its own width in `scratch/<key>.npy`."""
    arrays = host_inputs(inst)
    out, first, times, peak, how = run_nbx(inst, arrays, dev, res, warmup, reps)
    f = scratch / f"{inst['key']}.npy"
    np.save(f, _host_flat(out))
    od = _dtype_of(out)
    del out
    dev.release()
    return {"first": first, "times": times, "peak": peak, "how": how, "od": od, "file": str(f)}


def torch_work(inst, dev, res, warmup, reps, compile_modes, nbx: dict | None):
    """The torch side of one instance, as a stream of messages: the float64 reference in memory; FIRST the nbxtensor
    output judged against it (read by memory map, the scratch file removed), then torch eager and each compile mode
    (compile time apart), each announced (`{"start": name}`) before it runs and its row sent (`{"row": ...}`) as soon
    as it is judged; torch.compile's caches released after each. A worker that dies or is stopped in a contender has
    already handed every row before it — the nbxtensor row among them (the V100 run, 2026-11-03 04:23: a restart lost
    the nbxtensor row already measured)."""
    arrays = host_inputs(inst)
    try:
        ref = _to_numpy(reference(inst, arrays, res)).reshape(-1)
        ref_note = ""
    except Exception as e:
        ref, ref_note = None, f"no float64 reference: {type(e).__name__}: {e}"
    if nbx is not None:
        row = _row(inst, "nbxtensor")
        f = Path(nbx["file"])
        try:
            got = np.load(f, mmap_mode="r")
            errs = None if ref is None else compare_tiles(got, ref)
            del got
            yield {"row": judge_row(row, inst, nbx["od"], nbx["first"], nbx["times"], nbx["peak"], nbx["how"],
                                    errs, ref_note)}
        finally:
            f.unlink(missing_ok=True)
    contenders = [("torch_eager", None)] + [(f"torch_compile[{m}]", m) for m in compile_modes]
    for name, mode in contenders:
        yield {"start": name}
        row = _row(inst, name)
        try:
            out, first, times, peak, how = run_torch(inst, arrays, dev, res, warmup, reps, mode)
            errs = None if ref is None else compare_tiles(_host_flat(out), ref)
            od = _dtype_of(out)
            del out
            med = _stats(times)[0]
            # a compiled call's first launch carries the compilation: its time apart from the execution's
            row = judge_row(row, inst, od, first, times, peak, how, errs, ref_note,
                            compile_ms=(first - med) if mode is not None else None)
        except Exception as e:
            row.update(status="refused", note=f"{type(e).__name__}: {str(e)[:300]}")
        release_compile_caches(dev)
        yield {"row": row}
    del ref
    release_compile_caches(dev)


def worker(side: str, hardware: str, warmup: int, reps: int, compile_modes, scratch: Path) -> int:
    """A persistent worker: one JSON request per stdin line, one JSON answer per stdout line."""
    if side == "nbxtensor":
        # a confirmation run: a key the certified directory does not hold is refused by name, never swept and timed
        os.environ["NBX_AUTOTUNE_CERTIFIED_ONLY"] = "1"
        import instances as I
        I.bind_profile(hardware)
        dev = _NbxDevice()
    else:
        dev = _TorchDevice()
    res = _ArgResolver()
    real_out = sys.stdout
    sys.stdout = sys.stderr                     # the engine's prints never corrupt the answer channel
    def send(msg):
        real_out.write(json.dumps(msg) + "\n")
        real_out.flush()

    for line in sys.stdin:
        req = json.loads(line)
        try:
            if side == "nbxtensor":
                send({"ok": nbx_work(req["inst"], dev, res, warmup, reps, scratch)})
            else:
                for msg in torch_work(req["inst"], dev, res, warmup, reps, compile_modes, req.get("nbx")):
                    send(msg)
                send({"done": True})
        except Exception:
            send({"error": traceback.format_exc().replace("\n", " | ")[-600:]})
    return 0


class RunEnded(Exception):
    """The run's end (`--until`, set by the chain that launched the run) passed while a contender was still running."""


class _Worker:
    def __init__(self, python: str, side: str, a):
        import subprocess
        cmd = [python, __file__, "--worker", side, "--hardware", a.hardware, "--warmup", str(a.warmup),
               "--reps", str(a.reps), "--compile-modes", a.compile_modes, "--scratch", a.scratch]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(Path(a.out).with_suffix(f".{side}.log"), "a"))
        self._buf = b""

    def send(self, req: dict) -> None:
        self.p.stdin.write((json.dumps(req) + "\n").encode())
        self.p.stdin.flush()

    def read(self, until: float | None = None) -> dict:
        """The worker's next message; RunEnded when the run's end passes first."""
        import os as _os
        import select
        while b"\n" not in self._buf:
            wait = None if until is None else until - time.time()
            if wait is not None and wait <= 0:
                raise RunEnded()
            ready, _, _ = select.select([self.p.stdout], [], [], wait)
            if not ready:
                raise RunEnded()
            chunk = _os.read(self.p.stdout.fileno(), 1 << 16)
            if not chunk:
                raise RuntimeError(f"the worker ended (rc {self.p.poll()}): see its log")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def ask(self, req: dict, until: float | None = None) -> dict:
        self.send(req)
        return self.read(until)

    def kill(self):
        self.p.kill()
        self.p.wait()

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def _ask_or_restart(wk, req: dict, until: float | None = None) -> dict:
    """The worker's answer, or the named error of a worker that died on this instance (the caller restarts it).
    RunEnded passes through."""
    try:
        return wk.ask(req, until)
    except RunEnded:
        raise
    except Exception as e:
        return {"error": f"the worker ended on this instance: {type(e).__name__}: {e}"}


def torch_rows(wk, inst: dict, nbx: dict | None, until: float | None):
    """(rows, ended, error) of the torch worker on one instance, read message by message: every row it sent is
    kept whatever happens next. `ended`: the run's end passed in a contender — that contender's row says
    'not finished in the run' with its elapsed time, neither a loss nor a win, never a hand-written bound (the
    supervision, 2026-11-03 07:35: a compilation the run's own end stops is reported as such). `error`: the worker
    died — the contender it was in is refused with the error, the rows before it stand."""
    rows, current, t0 = [], "torch_eager", time.time()
    try:
        wk.send({"inst": inst, "nbx": nbx})
        while True:
            msg = wk.read(until)
            if "start" in msg:
                current, t0 = msg["start"], time.time()
            elif "row" in msg:
                rows.append(msg["row"])
            elif "done" in msg:
                return rows, False, None
            else:                                       # the worker's own error, outside any contender
                r = _row(inst, current); r.update(status="refused", note=msg.get("error", "")[:600])
                rows.append(r)
                return rows, False, msg.get("error")
    except RunEnded:
        r = _row(inst, current)
        r.update(status="not_finished",
                 note=f"not finished in the run: the run's end (--until) passed {time.time() - t0:.0f} s into it")
        rows.append(r)
        return rows, True, None
    except Exception as e:
        r = _row(inst, current); r.update(status="refused", note=f"the worker ended in this contender: {e}")
        rows.append(r)
        return rows, False, str(e)


def _restart(wk, python: str, side: str, a):
    """A fresh worker after an error: a device error that sticks to its context (a CUDA assert) is never carried to
    the next instance — only the instance that raised it is refused (02/11 V100: one bad index refused 1 053)."""
    wk.close()
    return _Worker(python, side, a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worker", choices=("torch", "nbxtensor"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--census", default=None, help="<vendor>/<profile>/<rung>: the cut keys an out-of-class row names")
    ap.add_argument("--instances")
    ap.add_argument("--out")
    ap.add_argument("--scratch", required=True, help="where the nbxtensor output of ONE instance waits for its comparison")
    ap.add_argument("--registry", default=None, help="the dated registry directory every row is appended to as well")
    ap.add_argument("--python-torch", help="the oracle environment's python (torch)")
    ap.add_argument("--python-nbxtensor", help="the library environment's python (no torch)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--compile-modes", default="default,max-autotune")
    ap.add_argument("--ops", default=None, help="comma-separated op types to keep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--until", type=float, default=None,
                    help="the run's end, epoch seconds, set by the chain that launches the run: a contender still "
                         "running then is reported 'not finished in the run' and the run stops (resumable)")
    a = ap.parse_args(argv)
    modes = [m for m in a.compile_modes.split(",") if m]
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if a.worker:
        return worker(a.worker, a.hardware, a.warmup, a.reps, modes, scratch)
    from registry import Registry
    import instances as I
    reg = Registry(a.registry, "kernel", FIELDS, a.hardware, REPO)
    profile = I.load_hardware_profile(a.hardware)
    budget = device_budget_bytes(profile)
    census_rows = I.census_rows(a.census) if a.census else []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["key"] for r in csv.DictReader(f, delimiter="\t")}
    insts = [json.loads(x) for x in open(a.instances)]
    if a.ops:
        keep = set(a.ops.split(","))
        insts = [i for i in insts if i["op_type"] in keep]
    insts = [i for i in insts if i["key"] not in done][:a.limit]
    new = not out.exists()
    nbx_w = torch_w = None
    try:
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
            if new:
                w.writeheader()
            ended = False
            for k, inst in enumerate(insts, 1):
                if a.until is not None and time.time() >= a.until:
                    print(f"[kernel] the run's end (--until) passed: stopped before {k}/{len(insts)} (resumable)",
                          flush=True)
                    break
                need = instance_bytes(inst)
                if need > budget:
                    rows = [out_of_class_row(inst, need, budget, census_cut_keys(inst, census_rows))]
                else:
                    nbx_w = nbx_w or _Worker(a.python_nbxtensor, "nbxtensor", a)
                    torch_w = torch_w or _Worker(a.python_torch, "torch", a)
                    rows = []
                    t0 = time.time()
                    try:
                        nb = _ask_or_restart(nbx_w, {"inst": inst}, a.until)
                    except RunEnded:
                        nbx_w.kill(); nbx_w = None
                        r = _row(inst, "nbxtensor")
                        r.update(status="not_finished", note=f"not finished in the run: the run's end (--until) "
                                                             f"passed {time.time() - t0:.0f} s into it")
                        rows.append(r); ended = True
                        nb = {}
                    if "error" in nb:
                        r = _row(inst, "nbxtensor"); r.update(status="refused", note=nb["error"]); rows.append(r)
                        nbx_w = _restart(nbx_w, a.python_nbxtensor, "nbxtensor", a)
                    if not ended:
                        tr, ended, err = torch_rows(torch_w, inst, nb.get("ok"), a.until)
                        rows += tr
                        if ended:
                            torch_w.kill(); torch_w = None
                        elif err is not None or any(r.get("status") == "refused" for r in tr):
                            # a dead worker, or a refused contender that may leave a sticky device error: the next
                            # instance starts clean
                            torch_w = _restart(torch_w, a.python_torch, "torch", a)
                for r in rows:
                    w.writerow(r)
                    reg.append(r)
                f.flush()
                print(f"[kernel] {k}/{len(insts)} {inst['op_type']} {inst['shapes']}: "
                      + " ".join(f"{r['contender']}={r.get('median_ms') or r['status']}" for r in rows), flush=True)
                if ended:
                    print(f"[kernel] the run's end (--until) passed in {k}/{len(insts)}: stopped (resumable)", flush=True)
                    break
    finally:
        for wk in (nbx_w, torch_w):
            if wk is not None:
                wk.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
