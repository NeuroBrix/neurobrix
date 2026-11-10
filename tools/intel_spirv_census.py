#!/usr/bin/env python3
"""Intel SPIR-V compile census: every `@triton.jit` kernel we own, pushed through Intel's Triton compiler to SPIR-V —
off-card, torch-free, **compile only, nothing is ever executed** (the owner, 2026-11-10 13:09: no Intel machine in the
lab; what prevents wrong code is what is proven without one).

It runs in an environment holding Intel's Triton build ALONE (`triton-xpu` from download.pytorch.org/whl/xpu: it
installs into the `triton` namespace and cannot share an environment with the CUDA Triton), with no torch, no oneAPI
and no Intel device. The target is the one nbxtensor's launch driver builds (`backend/level_zero_driver.target`):
`GPUTarget("xpu", <facts>, 32)` with the architecture name of a device file handed through `knobs.intel.device_arch`
(so the compiler never builds its SYCL arch parser), `__intel_already_queried_extensions__` set (so it never builds its
SYCL extension query) and no `device_id` (so it never builds its SPIR-V probes).

What a card would answer is given instead:
  --device-file NAME   a device file of nbxtensor/backend/devices/intel (its `compute_capability` is the arch name);
  --facts none|all     the extension facts (`level_zero.EXTENSION_FACTS`): none = a chip whose OpenCL ICD states none
                       of them (the oldest generation tried), all = a chip stating every one;
  the counts a chip answers (sub-group sizes 8/16/32, 1 024 threads a group, 64 KiB of shared local memory) are
  Intel's documented values for every Xe generation (compute-runtime, CL_DEVICE_SUB_GROUP_SIZES_INTEL).

The kernels, their invented signatures and the device-function screen are `tools/metal_compile_census.py`'s (one
harness for every compile census); the stages are Intel's own (`ttir -> ttgir -> llir -> spv`), each failure named
with the stage that raised it. A `ttir` failure is the harness's invented signature, never an Intel gap.

    <venv with triton-xpu>/bin/python tools/intel_spirv_census.py --device-file mtl --facts none --out DIR

Outputs in --out: `intel_spirv_census_<device>_<facts>.json` (one record per kernel) and `.md` (the counts and every
refusal with its stage and reason); `--spv` also keeps each module's bytes (`spv/<module>.<kernel>.spv`). Exit status
0 when the census ran, whatever it found: the refusals are its result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "nbxtensor" / "src"))

SUB_GROUP_SIZES = [8, 16, 32]          # CL_DEVICE_SUB_GROUP_SIZES_INTEL of every Xe generation (compute-runtime)
MAX_GROUP = 1024                       # maxTotalGroupSize
SHARED_LOCAL = 65536                   # maxSharedLocalMemory


def build_target(device_file: str, facts: str):
    """(GPUTarget, architecture name) as `level_zero_driver.target` builds it, the card's answers given."""
    import yaml
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from nbxtensor.backend.level_zero import EXTENSION_FACTS
    if not hasattr(knobs, "intel"):
        raise SystemExit("REFUSING: this Triton has no Intel backend (knobs.intel is absent) — run this tool with the "
                         "python of an environment holding triton-xpu alone")
    path = REPO_ROOT / "nbxtensor" / "src" / "nbxtensor" / "backend" / "devices" / "intel" / f"{device_file}.yml"
    if not path.exists():
        raise SystemExit(f"REFUSING: no Intel device file {path.name} (have: "
                         f"{sorted(p.stem for p in path.parent.glob('*.yml'))})")
    arch = str((yaml.safe_load(path.read_text()) or {}).get("compute_capability") or "")
    if not arch:
        raise SystemExit(f"REFUSING: {path.name} states no compute_capability")
    props = {fact: facts == "all" for fact in EXTENSION_FACTS}
    props.update({"name": f"off-card {device_file}", "vendor": "Intel(R) Corporation", "driver_version": "",
                  "arch": arch, "max_work_group_size": MAX_GROUP, "max_num_sub_groups": MAX_GROUP // min(SUB_GROUP_SIZES),
                  "sub_group_sizes": list(SUB_GROUP_SIZES), "local_mem_size": SHARED_LOCAL, "core_clock_rate": 0,
                  "__intel_already_queried_extensions__": True})
    knobs.intel.device_arch = arch
    return GPUTarget("xpu", props, 32), arch


class _Stages:
    """Intel's stages wrapped so a failure names the stage in flight; the pipeline ends after `spv`."""

    class Done(Exception):
        pass

    def __init__(self):
        self.current, self.completed = None, []

    def install(self):
        from triton.backends.intel.compiler import XPUBackend
        tracer, original = self, XPUBackend.add_stages

        def traced(backend, stages, options, *a, **k):
            original(backend, stages, options, *a, **k)
            for name, fn in list(stages.items()):
                stages[name] = tracer._wrap(name, fn)
        XPUBackend.add_stages = traced

    def _wrap(self, name, fn):
        def wrapped(src, metadata):
            self.current = name
            out = fn(src, metadata)
            self.completed.append(name)
            if name == "spv":
                self.spv = out
                raise _Stages.Done()
            return out
        return wrapped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-file", required=True)
    ap.add_argument("--facts", choices=("none", "all"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="the first N kernels only (the smallest real case)")
    ap.add_argument("--spv", action="store_true", help="keep each module's SPIR-V bytes")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(out / "triton_cache"))
    if "torch" in sys.modules:
        raise SystemExit("REFUSING: torch is imported before the census began")
    import triton
    from triton.compiler.compiler import ASTSource
    import metal_compile_census as H          # the compile censuses' one harness (kernels, signatures, the screen)
    target, arch = build_target(a.device_file, a.facts)
    kernels, import_failures = H.discover_kernels()
    if a.limit:
        kernels = kernels[:a.limit]
    tracer = _Stages()
    tracer.install()
    records = []
    for i, kernel in enumerate(kernels, 1):
        fn = kernel["fn"]
        rec = {"module": kernel["module"], "kernel": kernel["name"]}
        pointers = H._pointer_params_from_source(fn)
        if H._looks_like_device_function(fn, pointers):
            rec.update(status="device_function", stage="n/a")
            records.append(rec)
            continue
        try:
            signature, constexprs, _ = H.build_signature(fn, kernel["autotune_kwargs"], pointers)
        except Exception as exc:  # noqa: BLE001 — the harness's own failure, recorded as such
            rec.update(status="harness", stage="signature", reason=f"{type(exc).__name__}: {exc}"[:600])
            records.append(rec)
            continue
        tracer.current, tracer.completed, tracer.spv = None, [], None
        try:
            triton.compile(ASTSource(fn=fn, signature=signature, constexprs=constexprs), target=target)
            rec.update(status="refused", stage="spv", reason="the pipeline ended without a spv stage")
        except _Stages.Done:
            spv = tracer.spv
            data = spv if isinstance(spv, (bytes, bytearray)) else bytes(spv)
            ok = data[:4] == bytes.fromhex("03022307")
            rec.update(status="compiled" if ok else "refused", stage="spv", spv_bytes=len(data),
                       **({} if ok else {"reason": f"not a SPIR-V module (magic {data[:4].hex()})"}))
            if a.spv and ok:
                (out / "spv").mkdir(exist_ok=True)
                (out / "spv" / f"{kernel['module']}.{kernel['name']}.spv").write_bytes(data)
        except Exception as exc:  # noqa: BLE001 — every refusal is the census's result
            stage = tracer.current or "setup"
            rec.update(status="harness" if stage in ("ttir", "setup") else "refused", stage=stage,
                       exception=type(exc).__name__, reason=str(exc).strip()[:1200],
                       stages_cleared=list(tracer.completed))
        records.append(rec)
        print(f"  [{i:3}/{len(kernels)}] {rec['module']:32} {rec['kernel']:44} {rec['status']}"
              + (f" @{rec['stage']}" if rec["status"] in ("refused", "harness") else ""), flush=True)
    if "torch" in sys.modules:
        raise SystemExit("RED: torch was imported during the census — the compile path is not torch-free")
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    stem = f"intel_spirv_census_{a.device_file}_{a.facts}"
    meta = {"device_file": a.device_file, "arch": arch, "facts": a.facts, "triton": getattr(triton, "__version__", "?"),
            "kernels": len(kernels), "counts": counts, "import_failures": import_failures, "torch_imported": False}
    (out / f"{stem}.json").write_text(json.dumps({"meta": meta, "records": records}, indent=1))
    lines = [f"# Intel SPIR-V compile census — device file `{a.device_file}` (arch `{arch}`), facts `{a.facts}`", "",
             f"Triton {meta['triton']} (Intel build), off-card, torch never imported. {len(kernels)} kernels: "
             + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".", ""]
    refused = [r for r in records if r["status"] == "refused"]
    if refused:
        lines += ["## Refused (an Intel compiler finding)", "", "| module | kernel | stage | reason |", "|---|---|---|---|"]
        lines += [f"| {r['module']} | {r['kernel']} | {r['stage']} | {r.get('reason', '')[:200].replace('|', '/')} |"
                  for r in refused]
    harness = [r for r in records if r["status"] == "harness"]
    if harness:
        lines += ["", "## Harness (the invented signature, not Intel)", "", "| module | kernel | stage | reason |",
                  "|---|---|---|---|"]
        lines += [f"| {r['module']} | {r['kernel']} | {r['stage']} | {r.get('reason', '')[:160].replace('|', '/')} |"
                  for r in harness]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"census: {counts} -> {out / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
