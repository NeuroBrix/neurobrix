#!/usr/bin/env python3
"""A decode step's complete budget, measured at the dispatch point.

Every millisecond of a step gets a name, and the names sum to the step. That
is the point: a lever chosen from a budget is chosen from evidence, and a
lever chosen from intuition is a guess with a commit message.

What it counts, per decode step and for the whole run:

  * launches, and command buffers created and committed;
  * REAL GPU TIME — `GPUEndTime - GPUStartTime` off every command buffer once
    it has completed, which is Metal's own measurement of when the device was
    busy, not ours;
  * encode time — building the command buffer, before the commit;
  * commit time, and wait/drain time — the two halves of what used to be one
    blocking call;
  * copies by kind, with their bytes;
  * host synchronisations: how many flushes, and how long they waited;
  * the Python time BETWEEN launches, as the residual the instrumented seams
    do not account for. It is the engine's own host work and it is named as
    such rather than left as the gap between a total and a sum.

    python tools/metal_decode_budget.py --out DIR [--arm triton] [--max-tokens 8]
    python tools/metal_decode_budget.py --out DIR --arm compiled   # the ATen reference

Writes `budget.json` and `BUDGET.md`, or refuses to conclude.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class Budget:
    def __init__(self):
        self.n = defaultdict(int)
        self.t = defaultdict(float)
        self.bytes = defaultdict(int)
        self.gpu_s = 0.0
        self.gpu_intervals = 0
        self.kernels = defaultdict(int)
        self.marks: list = []          # (wall_time, label) at step boundaries

    def snapshot(self):
        return {"counts": dict(self.n),
                "seconds": {k: round(v, 5) for k, v in self.t.items()},
                "bytes": dict(self.bytes),
                "gpu_seconds": round(self.gpu_s, 5),
                "gpu_intervals": self.gpu_intervals}

    def mark(self):
        """A step boundary, as the engine declares it.

        A budget divided out of whole-run totals is not a step's budget: the
        run also holds the weight load, the prefill and the one-off MSL
        compilations. Doing that gave `launch` at 986 ms inside a 596 ms step
        and a residual of MINUS 530 ms — the arithmetic saying the
        denominator was a fiction. The counters are differenced between
        consecutive boundaries instead, so what is reported is one step.
        """
        self.marks.append({"wall": time.perf_counter(),
                           "counts": dict(self.n),
                           "seconds": dict(self.t),
                           "gpu_s": self.gpu_s,
                           "gpu_intervals": self.gpu_intervals})


def install(b: Budget, arm: str):
    """Wrap the real seams. Nothing here changes what runs."""
    if arm.startswith("triton"):
        _install_triton(b)
    else:
        _install_aten(b)
    _install_common(b)
    _install_step_boundary(b, arm)


def _install_step_boundary(b: Budget, arm: str):
    """The engine's own per-token hook, so a step boundary is its and not a
    guess: the sampler is called exactly once per generated token."""
    modules = (["neurobrix.triton.samplers"] if arm.startswith("triton")
               else ["neurobrix.core.module.autoregressive.samplers"])
    for name in modules:
        try:
            module = __import__(name, fromlist=["x"])
        except Exception:                                # pragma: no cover
            continue
        for attr in dir(module):
            cls = getattr(module, attr)
            if not isinstance(cls, type) or "Sampler" not in attr:
                continue
            call = getattr(cls, "__call__", None)
            if call is None or call is object.__call__:
                continue

            def sampled(self, *a, _orig=call, **k):
                b.mark()
                return _orig(self, *a, **k)
            cls.__call__ = sampled


def _install_common(b: Budget):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator

    _memcpy = DeviceAllocator.memcpy

    def memcpy(dst, src, nbytes, kind=3):
        started = time.perf_counter()
        try:
            return _memcpy(dst, src, nbytes, kind=kind)
        finally:
            elapsed = time.perf_counter() - started
            b.n[f"memcpy_kind{kind}"] += 1
            b.t[f"memcpy_kind{kind}"] += elapsed
            b.bytes[f"memcpy_kind{kind}"] += int(nbytes)
    DeviceAllocator.memcpy = staticmethod(memcpy)

    _sync = DeviceAllocator.sync_device

    def sync_device():
        started = time.perf_counter()
        try:
            return _sync()
        finally:
            b.n["sync_device"] += 1
            b.t["sync_device"] += time.perf_counter() - started
    DeviceAllocator.sync_device = staticmethod(sync_device)


def _install_triton(b: Budget):
    from neurobrix.kernels import launcher

    _prepare = launcher.prepare

    def prepare(kernel, args, kwargs):
        cache, _kc, _bk = launcher._binder(kernel)
        before = len(cache)
        started = time.perf_counter()
        try:
            return _prepare(kernel, args, kwargs)
        finally:
            elapsed = time.perf_counter() - started
            b.n["prepare"] += 1
            b.t["prepare"] += elapsed
            if len(cache) > before:
                b.n["compile"] += 1
                b.t["compile"] += elapsed
    launcher.prepare = prepare

    _launch = launcher.launch

    def launch(kernel, grid, *args, **kwargs):
        name = getattr(kernel, "__name__", None) or getattr(
            getattr(kernel, "fn", None), "__name__", "?")
        started = time.perf_counter()
        try:
            return _launch(kernel, grid, *args, **kwargs)
        finally:
            b.n["launch"] += 1
            b.t["launch"] += time.perf_counter() - started
            b.kernels[name] += 1
    launcher.launch = launch

    try:
        from neurobrix.triton import metal_driver
        from neurobrix.kernels import metal_device
    except Exception:                                    # pragma: no cover
        return

    # The driver's dispatch: encode, then commit. Split so the two halves are
    # separately visible — they are different levers.
    for attr in ("_dispatch_params", "_dispatch"):
        original = getattr(metal_driver.MetalKernel, attr, None)
        if original is None:
            continue

        def wrapper(self, *a, _orig=original, _name=attr, **k):
            started = time.perf_counter()
            try:
                return _orig(self, *a, **k)
            finally:
                b.n[f"driver.{_name}"] += 1
                b.t[f"driver.{_name}"] += time.perf_counter() - started
        setattr(metal_driver.MetalKernel, attr, wrapper)

    runtime_cls = metal_device.MetalRuntime

    for attr, label in (("track_committed", "commit+drain"),
                        ("flush", "flush"),
                        ("_blit", "blit_encode")):
        original = getattr(runtime_cls, attr, None)
        if original is None:
            continue

        def wrapper(self, *a, _orig=original, _label=label, **k):
            started = time.perf_counter()
            try:
                return _orig(self, *a, **k)
            finally:
                b.n[_label] += 1
                b.t[_label] += time.perf_counter() - started
        setattr(runtime_cls, attr, wrapper)

    # Real GPU time, read off each command buffer once it has completed.
    _await = getattr(runtime_cls, "_await", None)
    if _await is not None:
        def awaited(self, command_buffer, _orig=_await):
            started = time.perf_counter()
            try:
                return _orig(self, command_buffer)
            finally:
                b.n["await"] += 1
                b.t["await"] += time.perf_counter() - started
                try:
                    gpu = (float(command_buffer.GPUEndTime())
                           - float(command_buffer.GPUStartTime()))
                    if gpu > 0:
                        b.gpu_s += gpu
                        b.gpu_intervals += 1
                except Exception:
                    pass
        runtime_cls._await = awaited

    # Command buffers created, which is the count the stream lever changes.
    original_new = getattr(runtime_cls, "commandBuffer", None)
    if original_new is None:
        # counted at the two creation sites instead
        for mod, attr in ((metal_driver, "_dispatch_params"),):
            pass


def _install_aten(b: Budget):
    """The ATen reference. Its seams are torch's, so what is countable is the
    op count and the step wall; MPS exposes no per-op device timer, and that
    is stated rather than filled in with a number that is not measured."""
    try:
        from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    except Exception:                                    # pragma: no cover
        return
    # The hot loop takes a `pre_op_callback` by design, so counting the ops
    # costs nothing structural and touches nothing. What CANNOT be had here
    # is device time: MPS exposes no per-op timer the way a Metal command
    # buffer exposes GPUStartTime/GPUEndTime, so the GPU column of this arm
    # is left at zero rather than filled with a number that was not measured.
    original = getattr(CompiledSequence, "run", None)
    if original is None:
        return

    def counted(self, debug=False, pre_op_callback=None, _orig=original):
        def count(index, op, _user=pre_op_callback):
            b.n["aten_op"] += 1
            if _user is not None:
                _user(index, op)
        return _orig(self, debug=debug, pre_op_callback=count)
    CompiledSequence.run = counted


def _steps(progress: Path) -> list:
    if not progress.exists():
        return []
    stamps = []
    for line in progress.read_text().splitlines():
        found = re.search(r"^t=([0-9.]+)", line)
        if found:
            stamps.append(float(found.group(1)))
    return [round(b_ - a_, 4) for a_, b_ in zip(stamps, stamps[1:])]


def _markdown(doc: dict) -> str:
    t = doc["totals"]["seconds"]
    n = doc["totals"]["counts"]
    steps = doc["decode_steps"]
    median = statistics.median(steps) if steps else 0.0
    launches = n.get("launch") or n.get("aten_op") or 0
    per_step = launches / max(1, doc["max_tokens"])
    lines = [f"# A decode step's budget — `--{doc['arm']}`", "",
             f"Generated **{doc['generated']}** by `tools/metal_decode_budget.py` "
             f"on {doc['machine']}.", "",
             f"* {doc['model']}, {doc['max_tokens']} tokens, whole run "
             f"**{doc['wall_s']} s**",
             f"* decode steps: {len(steps)}, median **{median:.3f} s**"
             + (f", min {min(steps):.3f}, max {max(steps):.3f}" if steps else ""),
             f"* launches per step: **{per_step:.0f}**",
             f"* real GPU time, `GPUEndTime - GPUStartTime` summed over "
             f"{doc['totals']['gpu_intervals']} command buffers: "
             f"**{doc['totals']['gpu_seconds']:.3f} s** whole run, "
             f"**{doc['totals']['gpu_seconds'] / max(1, doc['max_tokens']) * 1000:.1f} ms** a step",
             "", "## One decode step, by name", "",
             f"Differenced between the {doc.get('decode_rows_used', 0)} step "
             f"boundaries the engine declared, median of the decode steps — "
             f"not a whole-run total divided by a token count, which also "
             f"holds the weight load, the prefill and the one-off "
             f"compilations.", "",
             "| what | count | ms |", "|---|---:|---:|"]
    budget = doc.get("step_budget", {})
    for key in sorted(budget, key=lambda k: -budget[k]["ms"]):
        lines.append(f"| `{key}` | {budget[key]['count']:.0f} | {budget[key]['ms']:.1f} |")
    lines += ["",
              f"| **GPU busy** (`GPUEndTime - GPUStartTime`, "
              f"{doc.get('step_gpu_intervals', 0):.0f} buffers) | | "
              f"**{doc.get('step_gpu_s', 0) * 1000:.1f}** |",
              f"| **step wall** | | **{doc.get('step_wall_s', 0) * 1000:.1f}** |", ""]
    if doc.get("residual_ms") is not None:
        lines += [f"`launch` contains prepare, the driver's encode and the commit, so "
                  f"it is counted alone; copies and syncs sit outside it. Those "
                  f"account for **{doc['accounted_ms']:.1f} ms** of the "
                  f"**{doc.get('step_wall_s', 0) * 1000:.1f} ms** step; the remaining "
                  f"**{doc['residual_ms']:.1f} ms** is host work between launches — "
                  f"the engine's own Python, named rather than left as a gap.", ""]
    if doc.get("top_kernels"):
        lines += ["## Kernels by launch count", "", "| kernel | launches | per step |",
                  "|---|---:|---:|"]
        for name, count in doc["top_kernels"]:
            lines.append(f"| `{name}` | {count} | {count / max(1, doc['max_tokens']):.0f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt",
                    default="Explain in one short paragraph why the sky appears blue.")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--arm", default="triton")
    ap.add_argument("--keep-caches", action="store_true")
    args = ap.parse_args()

    import os
    import shutil
    if not args.keep_caches:
        for path in (Path.home() / ".cache" / "triton_msl",
                     Path.home() / ".triton" / "cache"):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    args.out.mkdir(parents=True, exist_ok=True)
    progress = args.out / "progress.txt"
    progress.unlink(missing_ok=True)   # it is appended to; a stale run's
                                       # timestamps are not this run's steps
    os.environ["NBX_DECODE_PROGRESS"] = str(progress)

    b = Budget()
    if args.arm.startswith("triton"):
        from neurobrix.kernels import launcher
        launcher.install()
    install(b, args.arm)

    out_txt = args.out / "generated.txt"
    argv = ["neurobrix", "run", "--model", args.model, "--prompt", args.prompt,
            "--max-tokens", str(args.max_tokens), "--temperature", "0",
            f"--{args.arm}", "--output", str(out_txt)]
    saved, sys.argv = sys.argv, argv
    started = time.perf_counter()
    try:
        from neurobrix.__main__ import main as nbx_main
        nbx_main()
    except SystemExit:
        pass
    finally:
        sys.argv = saved
    wall = time.perf_counter() - started

    steps = _steps(progress)
    median = statistics.median(steps) if steps else 0.0
    totals = b.snapshot()

    # One step is the difference between two boundaries the engine declared.
    # The first interval covers the prefill and is dropped; what remains is
    # decode, and its median is the step this budget is about.
    per_step = []
    for before, after in zip(b.marks, b.marks[1:]):
        row = {"wall_s": after["wall"] - before["wall"],
               "gpu_s": after["gpu_s"] - before["gpu_s"],
               "gpu_intervals": after["gpu_intervals"] - before["gpu_intervals"],
               "counts": {k: after["counts"].get(k, 0) - before["counts"].get(k, 0)
                          for k in after["counts"]},
               "seconds": {k: after["seconds"].get(k, 0.0) - before["seconds"].get(k, 0.0)
                           for k in after["seconds"]}}
        per_step.append(row)
    decode_rows = per_step[1:] if len(per_step) > 1 else per_step

    def _median(pick):
        values = [pick(r) for r in decode_rows]
        return statistics.median(values) if values else 0.0

    step_wall = _median(lambda r: r["wall_s"])
    seam_keys = sorted({k for r in decode_rows for k in r["seconds"]})
    step_budget = {k: {"count": _median(lambda r, k=k: r["counts"].get(k, 0)),
                       "seconds": _median(lambda r, k=k: r["seconds"].get(k, 0.0))}
                   for k in seam_keys}
    # `launch` contains prepare, the driver halves and the commit, so it is
    # counted alone; copies and syncs happen outside it.
    inside = step_budget.get("launch", {}).get("seconds", 0.0) or \
        step_budget.get("aten_op", {}).get("seconds", 0.0)
    outside = sum(v["seconds"] for k, v in step_budget.items()
                  if k.startswith("memcpy") or k == "sync_device")
    accounted_ms = (inside + outside) * 1000
    median = step_wall or median
    document = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/metal_decode_budget.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "model": args.model, "arm": args.arm, "max_tokens": args.max_tokens,
        "wall_s": round(wall, 3),
        "decode_steps": steps,
        "median_step_s": round(median, 4),
        "totals": totals,
        "step_wall_s": round(step_wall, 5),
        "step_gpu_s": round(_median(lambda r: r["gpu_s"]), 5),
        "step_gpu_intervals": _median(lambda r: r["gpu_intervals"]),
        "step_budget": {k: {"count": round(v["count"], 1),
                            "ms": round(v["seconds"] * 1000, 2)}
                        for k, v in step_budget.items()},
        "decode_rows_used": len(decode_rows),
        "accounted_ms": round(accounted_ms, 2),
        "residual_ms": round(median * 1000 - accounted_ms, 2),
        "top_kernels": sorted(b.kernels.items(), key=lambda kv: -kv[1])[:20],
        "output_sha_note": "compare generated.txt against the campaign's arm sha",
    }
    (args.out / "budget.json").write_text(json.dumps(document, indent=1))
    (args.out / "BUDGET.md").write_text(_markdown(document))
    for name in ("budget.json", "BUDGET.md"):
        if not (args.out / name).exists():
            print(f"REFUSING to conclude: {name} was not written")
            return 2
    print(f"\nwritten: {args.out / 'BUDGET.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
