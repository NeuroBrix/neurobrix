"""Kernel level of the nbxtensor benchmark: each instance (`instances.py`) run by three contenders on one device.

Contenders, each through the engine's own per-op entry so the call is the one a run makes:
- `torch_eager`: `NativeATenDispatcher.dispatch` (the compiled engine's ATen call) on the device;
- `torch_compile` (`--compile-modes`, default "default,max-autotune"): `torch.compile` of that same call;
- `nbxtensor`: `TritonSequentialDispatcher.dispatch` (the Triton engine's kernel call, which becomes nbxtensor).

Protocol (README): the machine runs nothing else; the first call is timed APART (it carries compilation, autotune,
first-touch allocation) and reported as `first_ms`; then `--warmup` untimed calls; then `--reps` timed calls, each
bracketed by a device synchronize; the median and the 10th/90th percentiles are written with N. Every contender's
output is compared with a float64 reference computed by torch on the host from the same inputs: relative error =
max|out - ref| / max|ref|, held to the output dtype's bound (`TOLERANCE`); a contender over it is `wrong` — a faster
wrong result is a red, never a win. Peak memory: the bytes the contender's allocator held above its level before the
call (CUDA: `max_memory_allocated`; MPS keeps no peak: its driver sampled through the timed calls, said `sampled`;
nbxtensor: `DeviceAllocator`'s own peak counter).

One TSV per device, appended as each instance finishes (a stopped run keeps what it measured; a rerun skips the
instances already written).

    python benchmarks/nbxtensor_vs_torch/kernel_bench.py --hardware <profile id> \\
        --instances <results/<device>/instances.jsonl> --out <results/<device>/kernel.tsv>
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
    from neurobrix.kernels.nbx_tensor import NBXDtype
    return getattr(NBXDtype, name)


def _to_numpy(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    from neurobrix.kernels.nbx_tensor import NBXTensor
    if isinstance(x, NBXTensor):
        return np.asarray((x.float() if x.is_floating_point() else x).numpy(), dtype=np.float64)
    return x.detach().to("cpu").double().numpy()


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


class _Device:
    """The device both torch and the Triton engine run on, and its synchronize / peak readers."""

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

    def nbx_sync(self):
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.device_synchronize(0)


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


def run_nbx(inst, arrays, dev, res, warmup, reps):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    floats = [d for d in inst["dtypes"] if "float" in d]
    compute = _nbx_dtype(max(set(floats), key=floats.count) if floats else "float32")
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


def bench(inst, dev, res, warmup, reps, compile_modes) -> list:
    rows = []
    arrays = host_inputs(inst)
    try:
        ref = _to_numpy(reference(inst, arrays, res))
    except Exception as e:
        ref = None
        ref_note = f"no float64 reference: {type(e).__name__}: {e}"
    else:
        ref_note = ""
    contenders = [("torch_eager", lambda: run_torch(inst, arrays, dev, res, warmup, reps))]
    contenders += [(f"torch_compile[{m}]", (lambda m=m: run_torch(inst, arrays, dev, res, warmup, reps, m)))
                   for m in compile_modes]
    contenders += [("nbxtensor", lambda: run_nbx(inst, arrays, dev, res, warmup, reps))]
    for name, fn in contenders:
        row = {"key": inst["key"], "op_type": inst["op_type"], "shapes": json.dumps(inst["shapes"]),
               "dtypes": ",".join(inst["dtypes"]), "sites": len(inst["sites"]), "contender": name}
        try:
            out, first, times, peak, how = fn()
        except Exception as e:
            row.update(status="refused", note=f"{type(e).__name__}: {str(e)[:300]}")
            rows.append(row)
            continue
        med, p10, p90 = _stats(times)
        od = _dtype_of(out)
        # the narrowest of what the op reads, writes, and what the container records it writes (the precision
        # contract rounds an output to the traced dtype: an fp32 island's result handed back in bf16)
        tol = _bound(inst["dtypes"] + [od] + [str(d) for d in (inst.get("output_dtypes") or [])])
        status, err = "ok", None
        if ref is not None:
            got = _to_numpy(out)
            if got.shape != ref.shape:
                status = "wrong"
                err = float("inf")
            else:
                err = float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-30))
                if not err <= (tol if tol is not None else 0.0):
                    status = "wrong"
        row.update(status=status, first_ms=f"{first:.3f}", median_ms=f"{med:.4f}", p10_ms=f"{p10:.4f}",
                   p90_ms=f"{p90:.4f}", n=len(times), rel_err="" if err is None else f"{err:.3e}",
                   tolerance="" if tol is None else tol, peak_mb=f"{peak:.3f}", peak_how=how,
                   note=ref_note or ("" if tol is not None else f"no floating dtype to bound ({od}): exact match asked"))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--compile-modes", default="default,max-autotune")
    ap.add_argument("--ops", default=None, help="comma-separated op types to keep")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    import instances as I
    I.bind_profile(a.hardware)
    dev = _Device()
    res = _ArgResolver()
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
    modes = [m for m in a.compile_modes.split(",") if m]
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new:
            w.writeheader()
        for k, inst in enumerate(insts, 1):
            try:
                rows = bench(inst, dev, res, a.warmup, a.reps, modes)
            except Exception:
                rows = [{"key": inst["key"], "op_type": inst["op_type"], "contender": "*", "status": "refused",
                         "note": traceback.format_exc().replace("\n", " | ")[-600:]}]
            for r in rows:
                w.writerow(r)
            f.flush()
            print(f"[kernel] {k}/{len(insts)} {inst['op_type']} {inst['shapes']}: "
                  + " ".join(f"{r['contender']}={r.get('median_ms', r['status'])}" for r in rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
