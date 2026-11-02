"""Kernel level of the nbxtensor benchmark: each instance (`instances.py`) run by three contenders on one device.

Contenders, each through the engine's own per-op entry so the call is the one a run makes:
- `torch_eager`: `NativeATenDispatcher.dispatch` (the compiled engine's ATen call) on the device;
- `torch_compile` (`--compile-modes`, default "default,max-autotune"): `torch.compile` of that same call;
- `nbxtensor`: `TritonSequentialDispatcher.dispatch` (the Triton engine's kernel call, which becomes nbxtensor).

Two processes, two environments (the supervisor, 2026-11-02 11:55: torch in a separate oracle environment, never in
the library's): `--side torch` runs the torch contenders AND writes each instance's float64 reference to `--refs`
(`<key>.npy`), in an environment that has torch; `--side nbxtensor` runs nbxtensor in the library's torch-free
environment and judges its output against that reference file (the torch side runs first; a missing reference is
said in the row). Both draw the same inputs (`host_inputs`, seeded) and append to the same TSV, and to the dated
registry (`--registry`, `registry.py`) as each row is written.

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

    <torch python> benchmarks/nbxtensor_vs_torch/kernel_bench.py --side torch --hardware <profile id> \\
        --instances <campaign>/instances.jsonl> --out <campaign>/kernel.tsv> --refs <dir> --registry <dir>
    <library python> benchmarks/nbxtensor_vs_torch/kernel_bench.py --side nbxtensor ... (the same arguments)
"""
from __future__ import annotations

import argparse
import csv
import json
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
FIELDS = ["key", "op_type", "shapes", "dtypes", "sites", "contender", "status", "first_ms", "median_ms", "p10_ms",
          "p90_ms", "n", "rel_err", "tolerance", "peak_mb", "peak_how", "note"]


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


def host_inputs(inst: dict, seed: int = 0) -> list:
    """One numpy array per input, deterministic: floats ~ N(0, 1) scaled by 1/sqrt(last dim) (a GEMM's K, so half
    outputs stay in range); integers within the extent of the input they index (the largest dim of the op's first
    floating input), booleans fair."""
    rng = np.random.default_rng(seed)
    out = []
    float_extent = max((max(s) if s else 1) for s, d in zip(inst["shapes"], inst["dtypes"])
                       if "float" in d) if any("float" in d for d in inst["dtypes"]) else 2
    for shape, dt in zip(inst["shapes"], inst["dtypes"]):
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
    for tid, a, d in zip(inst["inputs"], arrays, inst["dtypes"]):
        t = torch.from_numpy(a)
        # the inputs the contenders read, rounded to their dtype, then widened: the error measured is the
        # computation's, never the rounding of the inputs both contenders share
        store[tid] = t.to(_torch_dtype(d)).double() if t.is_floating_point() else t
    args = attrs_resolver.args(inst["args"], store, _symbols(inst["symbols"]), _NativeAttrs(disp))
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
             for tid, a, d in zip(inst["inputs"], arrays, inst["dtypes"])}
    args = res.args(inst["args"], store, _symbols(inst["symbols"]), _NativeAttrs(disp))
    attrs = {**inst["attributes"], "args": inst["args"]}

    def eager():
        return disp.dispatch(inst["op_type"], args, attrs)
    call = eager
    if compile_mode is not None:
        compiled = torch.compile(lambda: disp.dispatch(inst["op_type"], args, attrs), mode=compile_mode,
                                 dynamic=False, fullgraph=False)
        call = compiled
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
    return out, first, times, peak, peak_how


def _compute_dtype(inst) -> str:
    """The Triton dispatcher's compute dtype for an instance, deterministic: the floating dtype the op writes (the
    container's record of its output), else the one most of its floating inputs carry — a tie (54 of the Apple
    census's instances: an fp32 island beside a half weight) broken by name, never by a set's order, which Python
    salts per process."""
    out = [str(d) for d in (inst.get("output_dtypes") or []) if "float" in str(d)]
    if out:
        return out[0]
    floats = [d for d in inst["dtypes"] if "float" in d]
    if not floats:
        return "float32"
    return sorted(set(floats), key=lambda d: (-floats.count(d), d))[0]


def run_nbx(inst, arrays, dev, res, warmup, reps):
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.tensor import NBXTensor
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    compute = _nbx_dtype(_compute_dtype(inst))
    stores_fp64 = _stores_fp64()
    disp = TritonSequentialDispatcher(device_idx=0, compute_dtype=compute, stores_fp64=stores_fp64)
    disp.symbol_resolver = _symbols(inst["symbols"])
    store = {}
    for tid, arr, d in zip(inst["inputs"], arrays, inst["dtypes"]):
        t = NBXTensor.from_numpy(np.ascontiguousarray(arr))      # the host array in its own container
        store[tid] = t.to(_nbx_dtype(d)) if "float" in d and d != "float32" else t
    with disp:
        args = res.args(inst["args"], store, disp.symbol_resolver, disp)
        attrs = {**inst["attributes"], "args": inst["args"]}
        record = {"output_dtypes": inst.get("output_dtypes"), "op_type": inst["op_type"]}

        def call():
            return disp.dispatch(inst["op_type"], args, attrs, op_record=record)
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


def _judge(row, inst, out, first, times, peak, how, ref, ref_note) -> dict:
    """The row of one contender's run: its times, and its output against the float64 reference."""
    med, p10, p90 = _stats(times)
    od = _dtype_of(out)
    # the narrowest of what the op reads, writes, and what the container records it writes (the precision
    # contract rounds an output to the traced dtype: an fp32 island's result handed back in bf16)
    tol = _bound(inst["dtypes"] + [od] + [str(d) for d in (inst.get("output_dtypes") or [])])
    status, err = "ok", None
    if ref is not None:
        got = _to_numpy(out)
        if got.shape != ref.shape:
            status, err = "wrong", float("inf")
        else:
            err = float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-30))
            if not err <= (tol if tol is not None else 0.0):
                status = "wrong"
    row.update(status=status, first_ms=f"{first:.3f}", median_ms=f"{med:.4f}", p10_ms=f"{p10:.4f}",
               p90_ms=f"{p90:.4f}", n=len(times), rel_err="" if err is None else f"{err:.3e}",
               tolerance="" if tol is None else tol, peak_mb=f"{peak:.3f}", peak_how=how,
               note=ref_note or ("" if tol is not None else f"no floating dtype to bound ({od}): exact match asked"))
    return row


def bench_torch(inst, dev, res, warmup, reps, compile_modes, refs: Path) -> list:
    """The torch side: the float64 reference (written to `refs/<key>.npy` for the nbxtensor side), then torch eager
    and each torch.compile mode."""
    arrays = host_inputs(inst)
    try:
        ref = _to_numpy(reference(inst, arrays, res))
        np.save(refs / f"{inst['key']}.npy", ref)
        ref_note = ""
    except Exception as e:
        ref, ref_note = None, f"no float64 reference: {type(e).__name__}: {e}"
    contenders = [("torch_eager", lambda: run_torch(inst, arrays, dev, res, warmup, reps))]
    contenders += [(f"torch_compile[{m}]", (lambda m=m: run_torch(inst, arrays, dev, res, warmup, reps, m)))
                   for m in compile_modes]
    rows = []
    for name, fn in contenders:
        row = _row(inst, name)
        try:
            rows.append(_judge(row, inst, *fn(), ref, ref_note))
        except Exception as e:
            row.update(status="refused", note=f"{type(e).__name__}: {str(e)[:300]}")
            rows.append(row)
        dev.release()
    return rows


def bench_nbxtensor(inst, dev, res, warmup, reps, refs: Path) -> list:
    """The nbxtensor side: the Triton engine's call, judged against the torch side's float64 reference file."""
    arrays = host_inputs(inst)
    f = refs / f"{inst['key']}.npy"
    ref, ref_note = ((np.load(f), "") if f.exists()
                     else (None, f"no float64 reference: {f.name} not written by the torch side"))
    row = _row(inst, "nbxtensor")
    try:
        row = _judge(row, inst, *run_nbx(inst, arrays, dev, res, warmup, reps), ref, ref_note)
    except Exception as e:
        row.update(status="refused", note=f"{type(e).__name__}: {str(e)[:300]}")
    dev.release()
    return [row]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--side", required=True, choices=("torch", "nbxtensor"),
                    help="torch: the torch contenders and the references (an environment with torch); "
                         "nbxtensor: the library's contender (its torch-free environment)")
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", required=True, help="the float64 references: written by the torch side, read by nbxtensor")
    ap.add_argument("--registry", default=None, help="the dated registry directory every row is appended to as well")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--compile-modes", default="default,max-autotune")
    ap.add_argument("--ops", default=None, help="comma-separated op types to keep")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from registry import Registry
    reg = Registry(a.registry, "kernel", FIELDS, a.hardware, REPO)
    if a.side == "nbxtensor":
        import instances as I
        I.bind_profile(a.hardware)
        dev = _NbxDevice()
    else:
        dev = _TorchDevice()
    res = _ArgResolver()
    refs = Path(a.refs)
    refs.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mine = (lambda c: c == "nbxtensor") if a.side == "nbxtensor" else (lambda c: c != "nbxtensor")
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["key"] for r in csv.DictReader(f, delimiter="\t") if mine(r["contender"])}
    insts = [json.loads(x) for x in open(a.instances)]
    if a.ops:
        keep = set(a.ops.split(","))
        insts = [i for i in insts if i["op_type"] in keep]
    insts = [i for i in insts if i["key"] not in done][:a.limit]
    modes = [m for m in a.compile_modes.split(",") if m]
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new:
            w.writeheader()
        for k, inst in enumerate(insts, 1):
            try:
                rows = (bench_nbxtensor(inst, dev, res, a.warmup, a.reps, refs) if a.side == "nbxtensor"
                        else bench_torch(inst, dev, res, a.warmup, a.reps, modes, refs))
            except Exception:
                rows = [{"key": inst["key"], "op_type": inst["op_type"],
                         "contender": "nbxtensor" if a.side == "nbxtensor" else "torch_eager",
                         "status": "refused", "note": traceback.format_exc().replace("\n", " | ")[-600:]}]
            for r in rows:
                w.writerow(r)
                reg.append(r)
            f.flush()
            print(f"[kernel:{a.side}] {k}/{len(insts)} {inst['op_type']} {inst['shapes']}: "
                  + " ".join(f"{r['contender']}={r.get('median_ms', r['status'])}" for r in rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
