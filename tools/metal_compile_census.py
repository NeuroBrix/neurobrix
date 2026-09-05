#!/usr/bin/env python3
"""Metal compile census: every `@triton.jit` kernel we own, pushed through the
Metal backend's compile path. **Compile only — nothing is ever executed.**

Step 4 of the Metal adoption plan. It exists because a README is not a
measurement: the backend publishes a refusal list written against a general
test suite, and what we need is the refusal list against *our* 424 kernels.
It is also the only thing that can answer the `tl.dot sizePerThread`
question, which the scoping study named as unknowable by reading.

## What is censused

`src/neurobrix/kernels/ops/` — 424 `@triton.jit` kernels across 161 files, the
proprietary compute asset of mode 2. The vendored reference corpus under
`kernels/triton_kernels_ref/` is NOT ours and is not censused.

## The four stages, and why the stage matters more than the verdict

triton-msl declares its pipeline as four named stages:

    ttir  ->  ttgir  ->  msl  ->  metallib

Every failure is attributed to the stage that raised it, because the stages
mean different things and lumping them together would produce a gap list that
is mostly our own harness:

* **ttir** — Triton's own frontend. A failure here is almost always THIS
  TOOL's fault, not Metal's: a census has to invent a signature and a value
  for every `tl.constexpr`, and an invented value can be rejected by the
  kernel's own asserts. Counted as `harness`, never as a Metal gap.
* **ttgir** — Triton's target-independent middle end.
* **msl** — TTGIR lowered to Metal Shading Language. **This is the stage that
  measures Metal coverage.** A kernel that clears it has been lowered; a
  kernel that fails it is a genuine finding, and the backend's own error
  taxonomy says which kind (hardware limit / not implemented / integrity
  refusal / validation).
* **metallib** — MSL handed to `xcrun metal` and `xcrun metallib`. This stage
  is a **toolchain** question, not a coverage one: it needs Apple's offline
  shader compiler, which ships separately from the Command Line Tools.

`--through msl` stops after the lowering stage, which is what makes the
coverage census runnable on a machine without the Metal Toolchain installed.
`--through metallib` runs the whole pipeline.

## How the signatures are invented, said plainly

A census compiles kernels nobody called, so it must supply an argument
signature. The provenance of every value is recorded per kernel in the JSON:

* `autotune`  — read from the kernel's own `@triton.autotune` config. Real
                values the engine itself would use. Highest fidelity.
* `default`   — the parameter's own default.
* `heuristic` — this tool's declared name table (below). Lowest fidelity, and
                the reason `ttir` failures are counted as harness noise.

R33 preserved — this tool imports triton, never torch.
R34 preserved — nothing here is keyed on a model name.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPS_PKG = "neurobrix.kernels.ops"
OPS_DIR = REPO_ROOT / "src" / "neurobrix" / "kernels" / "ops"

# Pipeline stages, in the order triton-msl declares them. `_STAGE_ORDER`
# is what `--through` truncates and what failure attribution reports.
_STAGE_ORDER = ["ttir", "ttgir", "msl", "metallib"]

# ---------------------------------------------------------------------------
# The declared constexpr name table.
#
# Used ONLY when a kernel offers neither an autotune config nor a default. The
# values are deliberately conservative for Apple's 32 KB threadgroup budget
# (config/vendors/apple/apple_silicon.yml): a tile that spills would be
# refused for its size and would tell us nothing about the kernel's ops.
# Every kernel's resolved values are written to the JSON, so a reader can
# reproduce or contest any single row.
# ---------------------------------------------------------------------------
_CONSTEXPR_BY_EXACT_NAME = {
    "num_warps": 4,
    "num_stages": 1,
    "NUM_WARPS": 4,
    "NUM_STAGES": 1,
}

_CONSTEXPR_BLOCK_TOKENS = ("BLOCK", "TILE", "CHUNK", "GROUP_SIZE", "SPLIT")
_CONSTEXPR_BOOL_PREFIXES = ("HAS_", "IS_", "USE_", "DO_", "EVEN_", "NEED_",
                            "ALLOW_", "APPLY_", "WITH_", "ENABLE_", "STORE_",
                            "LOAD_", "REQUIRES_", "SCALE_BY_", "RETURN_")
_CONSTEXPR_BOOL_TOKENS = ("_ENABLED", "_PRESENT", "CAUSAL", "TRANSPOSE",
                          "CONTIGUOUS", "INTERLEAVED")

# Non-constexpr scalars whose name says they are floating point. Everything
# else that is not a pointer is typed i32, which is what shapes and strides
# are throughout `kernels/ops/`.
_FLOAT_SCALAR_TOKENS = ("eps", "epsilon", "scale", "alpha", "beta", "gamma",
                        "momentum", "dropout", "temperature", "theta", "clip",
                        "value", "tol", "delta", "lr", "weight_decay", "p_")


def _is_pointer(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("_ptr") or lowered.endswith("_ptrs") or "ptr" in lowered


def _scalar_type(name: str) -> str:
    lowered = name.lower()
    if any(tok in lowered for tok in _FLOAT_SCALAR_TOKENS):
        return "fp32"
    return "i32"


def _heuristic_constexpr(name: str):
    """The declared fallback value for a constexpr with no better source."""
    if name in _CONSTEXPR_BY_EXACT_NAME:
        return _CONSTEXPR_BY_EXACT_NAME[name]
    upper = name.upper()
    if any(upper.startswith(p) for p in _CONSTEXPR_BOOL_PREFIXES):
        return True
    if any(tok in upper for tok in _CONSTEXPR_BOOL_TOKENS):
        return True
    if any(tok in upper for tok in _CONSTEXPR_BLOCK_TOKENS):
        return 64
    if "SIZE" in upper or "DIM" in upper or "WIDTH" in upper or "LEN" in upper:
        return 64
    if upper == name and name.isupper():
        # An all-caps constexpr with no other signal: a small power of two is
        # the least surprising choice and keeps tiles inside the budget.
        return 16
    return 1


# ---------------------------------------------------------------------------
# Kernel discovery
# ---------------------------------------------------------------------------

def _unwrap(obj):
    """Peel @triton.autotune / @triton.heuristics down to the JITFunction.

    Returns (jit_fn, autotune_kwargs_or_None). The autotune config is worth
    peeling for: it carries the constexpr values the engine itself uses, which
    is the difference between censusing our kernel and censusing a guess.
    """
    autotune_kwargs = None
    seen = 0
    while hasattr(obj, "fn") and seen < 8:
        configs = getattr(obj, "configs", None)
        if configs and autotune_kwargs is None:
            try:
                autotune_kwargs = dict(configs[0].kwargs)
            except Exception:
                autotune_kwargs = None
        obj = obj.fn
        seen += 1
    return obj, autotune_kwargs


def discover_kernels():
    """Import every module in kernels/ops and collect its JITFunctions.

    Keyed by (module, kernel name) and de-duplicated by the JITFunction
    identity, so a kernel re-exported from another module is censused once
    and attributed to the module that defines it.
    """
    from triton.runtime.jit import JITFunction

    sys.path.insert(0, str(REPO_ROOT / "src"))
    kernels, import_failures, seen_ids = [], [], set()

    for path in sorted(OPS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        mod_name = f"{OPS_PKG}.{path.stem}"
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:
            import_failures.append({
                "module": path.name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        for attr in sorted(dir(module)):
            obj = getattr(module, attr, None)
            jit_fn, autotune_kwargs = _unwrap(obj)
            if not isinstance(jit_fn, JITFunction):
                continue
            if id(jit_fn) in seen_ids:
                continue
            seen_ids.add(id(jit_fn))
            kernels.append({
                "module": path.name,
                "name": getattr(jit_fn, "__name__", attr),
                "attr": attr,
                "fn": jit_fn,
                "autotune_kwargs": autotune_kwargs,
            })
    return kernels, import_failures


# ---------------------------------------------------------------------------
# Signature synthesis
# ---------------------------------------------------------------------------

def build_signature(jit_fn, autotune_kwargs):
    """Return (signature, constexprs, provenance) for one kernel."""
    signature, constexprs, provenance = {}, {}, {}

    for param in jit_fn.params:
        name = param.name
        if param.is_constexpr:
            signature[name] = "constexpr"
            if autotune_kwargs and name in autotune_kwargs:
                constexprs[name] = autotune_kwargs[name]
                provenance[name] = "autotune"
            elif getattr(param, "default", None) is not None and \
                    param.default is not inspect_empty():
                constexprs[name] = param.default
                provenance[name] = "default"
            else:
                constexprs[name] = _heuristic_constexpr(name)
                provenance[name] = "heuristic"
            continue

        annotation = (param.annotation or "").strip() if isinstance(
            getattr(param, "annotation", None), str) else ""
        if annotation.startswith("*") or annotation in ("i32", "i64", "fp16",
                                                        "fp32", "fp64", "bf16"):
            signature[name] = annotation
            provenance[name] = "annotation"
        elif _is_pointer(name):
            signature[name] = "*fp32"
            provenance[name] = "heuristic"
        else:
            signature[name] = _scalar_type(name)
            provenance[name] = "heuristic"

    return signature, constexprs, provenance


def inspect_empty():
    import inspect
    return inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Stage-traced compilation
# ---------------------------------------------------------------------------

class _StageTracer:
    """Wrap the backend's stages so a failure names the stage that raised it.

    Triton runs `stages` as an ordered dict of callables; wrapping each one
    records the stage in flight and, with `stop_after`, ends the pipeline at
    a chosen stage by raising a sentinel the caller recognises.
    """

    def __init__(self, stop_after=None):
        self.stop_after = stop_after
        self.current = None
        self.completed = []

    class _Done(Exception):
        pass

    def install(self):
        from triton_msl.backend.compiler import MetalBackend
        tracer = self
        original = MetalBackend.add_stages

        def traced(self_backend, stages, options, language=None):
            original(self_backend, stages, options, language)
            for stage_name, stage_fn in list(stages.items()):
                stages[stage_name] = tracer._wrap(stage_name, stage_fn)

        MetalBackend.add_stages = traced
        self._original = original
        self._backend_cls = MetalBackend

    def restore(self):
        self._backend_cls.add_stages = self._original

    def _wrap(self, stage_name, stage_fn):
        tracer = self

        def wrapped(src, metadata):
            tracer.current = stage_name
            result = stage_fn(src, metadata)
            tracer.completed.append(stage_name)
            if tracer.stop_after and stage_name == tracer.stop_after:
                raise _StageTracer._Done()
            return result

        return wrapped

    def reset(self):
        self.current = None
        self.completed = []


def classify(exc) -> str:
    """Map an exception onto the backend's own refusal taxonomy."""
    try:
        from triton_msl import errors as msl_errors
    except Exception:
        msl_errors = None

    if msl_errors is not None:
        for cls_name, label in (
            ("MetalNonRecoverableError", "integrity_refusal"),
            ("MetalUnsupportedError", "hardware_unsupported"),
            ("MetalNotImplementedError", "not_implemented"),
            ("MetalValidationError", "validation"),
            ("MetalCompilationError", "msl_compile_error"),
            ("MetalCodegenError", "codegen"),
        ):
            cls = getattr(msl_errors, cls_name, None)
            if cls is not None and isinstance(exc, cls):
                return label

    name = type(exc).__name__
    if name in ("CompilationError", "CompileTimeAssertionFailure"):
        return "frontend"
    return f"other:{name}"


def compile_one(kernel, tracer, target, backend_options):
    """Compile one kernel. Returns a record; never raises."""
    import triton
    from triton.compiler.compiler import ASTSource

    jit_fn = kernel["fn"]
    record = {
        "module": kernel["module"],
        "kernel": kernel["name"],
    }
    try:
        signature, constexprs, provenance = build_signature(
            jit_fn, kernel["autotune_kwargs"])
    except Exception as exc:
        record.update(status="harness", stage="signature", category="harness",
                      reason=f"{type(exc).__name__}: {exc}")
        return record

    record["signature"] = signature
    record["constexprs"] = {k: repr(v) for k, v in constexprs.items()}
    record["constexpr_provenance"] = provenance

    tracer.reset()
    try:
        src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
        triton.compile(src, target=target, options=backend_options)
    except _StageTracer._Done:
        record.update(status="compiled", stage=tracer.completed[-1],
                      stages_cleared=list(tracer.completed))
        return record
    except Exception as exc:
        stage = tracer.current or "setup"
        category = classify(exc)
        # A ttir failure is this tool inventing a signature the kernel
        # rejects, not a Metal gap. Say so in the record rather than let it
        # inflate the gap list.
        if stage in ("ttir", "signature", "setup"):
            category = "harness"
        message = str(exc).strip().replace("\r", "")
        record.update(
            status="refused" if category not in ("harness",) else "harness",
            stage=stage,
            category=category,
            reason=message[:1200],
            exception=type(exc).__name__,
            stages_cleared=list(tracer.completed),
        )
        return record
    record.update(status="compiled", stage=_STAGE_ORDER[-1],
                  stages_cleared=list(tracer.completed))
    return record


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(records, import_failures, meta, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta["date"]

    json_path = out_dir / f"metal_compile_census_{stamp}.json"
    with open(json_path, "w") as fh:
        json.dump({"meta": meta,
                   "import_failures": import_failures,
                   "kernels": records}, fh, indent=2, sort_keys=True)

    compiled = [r for r in records if r["status"] == "compiled"]
    refused = [r for r in records if r["status"] == "refused"]
    harness = [r for r in records if r["status"] == "harness"]

    by_category = {}
    for r in refused:
        by_category.setdefault(r["category"], []).append(r)

    md_path = out_dir / f"metal_compile_census_{stamp}.md"
    with open(md_path, "w") as fh:
        w = fh.write
        w(f"# Metal compile census — {stamp}\n\n")
        w(f"**{meta['machine']}**, macOS {meta['macos']}, "
          f"triton {meta['triton_version']}, "
          f"triton-msl {meta['triton_msl_version']}, "
          f"target `{meta['target']}`.\n\n")
        w(f"Pipeline run through stage **`{meta['through']}`** "
          f"of `{' -> '.join(_STAGE_ORDER)}`. "
          f"Compile only; no kernel was executed.\n\n")
        w(f"Scope: `src/neurobrix/kernels/ops/` — "
          f"{meta['modules_scanned']} modules, "
          f"{meta['kernels_found']} `@triton.jit` kernels discovered.\n\n")

        w("## Result\n\n")
        w("| outcome | kernels |\n|---|---:|\n")
        w(f"| compiled through `{meta['through']}` | **{len(compiled)}** |\n")
        w(f"| refused by the backend | **{len(refused)}** |\n")
        w(f"| not censused (harness could not build a valid call) "
          f"| {len(harness)} |\n")
        w(f"| **total** | **{len(records)}** |\n\n")

        if refused:
            w("## Refusals by category\n\n")
            w("Categories are the backend's own error taxonomy "
              "(`triton_msl/errors.py`), not ours.\n\n")
            w("| category | kernels |\n|---|---:|\n")
            for cat, rows in sorted(by_category.items(),
                                    key=lambda kv: -len(kv[1])):
                w(f"| `{cat}` | {len(rows)} |\n")
            w("\n## Every refusal, by kernel\n\n")
            for cat, rows in sorted(by_category.items(),
                                    key=lambda kv: -len(kv[1])):
                w(f"### `{cat}` — {len(rows)} kernel(s)\n\n")
                for r in sorted(rows, key=lambda x: (x["module"], x["kernel"])):
                    w(f"**`{r['module']}::{r['kernel']}`** — "
                      f"failed at stage `{r['stage']}` "
                      f"(`{r.get('exception', '?')}`)\n\n")
                    w("```\n" + r["reason"].strip() + "\n```\n\n")

        if harness:
            w("## Not censused — the harness, not the backend\n\n")
            w("These kernels were rejected by Triton's own frontend before "
              "any Metal stage ran, because this tool had to invent a "
              "signature or a `tl.constexpr` value and the kernel rejected "
              "it. They are a limit of the census, **not** a Metal gap, and "
              "are excluded from the refusal count above.\n\n")
            w("| module | kernel | reason (first line) |\n|---|---|---|\n")
            for r in sorted(harness, key=lambda x: (x["module"], x["kernel"])):
                first = (r.get("reason") or "").strip().splitlines()
                first = first[0][:110] if first else ""
                w(f"| `{r['module']}` | `{r['kernel']}` | {first} |\n")
            w("\n")

        if import_failures:
            w("## Modules that would not import\n\n")
            w("| module | error |\n|---|---|\n")
            for f in import_failures:
                w(f"| `{f['module']}` | {f['error'][:140]} |\n")
            w("\n")

        w("## Provenance of the constexpr values\n\n")
        w("A census compiles kernels nobody called, so every "
          "`tl.constexpr` needed a value. Per-kernel provenance is in the "
          "JSON beside this file.\n\n")
        counts = {}
        for r in records:
            for src in (r.get("constexpr_provenance") or {}).values():
                counts[src] = counts.get(src, 0) + 1
        w("| source | parameters |\n|---|---:|\n")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            w(f"| `{k}` | {v} |\n")
        w(f"\nMachine-readable record: `{json_path.name}`\n")

    return md_path, json_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--through", default="msl", choices=_STAGE_ORDER,
                    help="last pipeline stage to run (default: msl, the "
                         "coverage stage; metallib additionally needs "
                         "Apple's offline shader compiler)")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the dated report "
                         "(default: validation_outputs/metal_first_light_<date>)")
    ap.add_argument("--limit", type=int, default=None,
                    help="census only the first N kernels (smoke runs)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y_%m_%d")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "validation_outputs" / f"metal_first_light_{stamp}")

    import triton
    import triton_msl
    from triton.backends.compiler import GPUTarget
    import triton.runtime.driver as driver_mod

    target = driver_mod.driver.active.get_current_target()
    if target.backend != "metal":
        print(f"REFUSING: the active Triton target is '{target.backend}', "
              f"not 'metal'. A census run anywhere but on the Metal backend "
              f"would measure the wrong compiler.", file=sys.stderr)
        return 2

    kernels, import_failures = discover_kernels()
    if args.limit:
        kernels = kernels[:args.limit]
    print(f"discovered {len(kernels)} @triton.jit kernels in "
          f"{OPS_DIR.relative_to(REPO_ROOT)}")
    print(f"target: {target}   through stage: {args.through}\n")

    tracer = _StageTracer(stop_after=args.through
                          if args.through != _STAGE_ORDER[-1] else None)
    tracer.install()
    records = []
    try:
        for i, kernel in enumerate(kernels, 1):
            rec = compile_one(kernel, tracer, target, None)
            records.append(rec)
            mark = {"compiled": "ok", "refused": "REFUSED",
                    "harness": "--"}[rec["status"]]
            print(f"  [{i:3}/{len(kernels)}] {rec['module']:32} "
                  f"{rec['kernel']:44} {mark}")
    finally:
        tracer.restore()

    meta = {
        "date": stamp,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "triton_version": getattr(triton, "__version__", "unknown"),
        "triton_msl_version": getattr(triton_msl, "__version__", "unknown"),
        "triton_msl_codegen": getattr(triton_msl, "CODEGEN_VERSION", "unknown"),
        "target": f"{target.backend}/{target.arch}",
        "through": args.through,
        "modules_scanned": len(sorted(OPS_DIR.glob("*.py"))) - 1,
        "kernels_found": len(kernels),
        "metal_toolchain_present": bool(
            os.system("xcrun -f metal >/dev/null 2>&1") == 0),
    }
    md_path, json_path = write_report(records, import_failures, meta, out_dir)
    print(f"\ncensus written:\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
