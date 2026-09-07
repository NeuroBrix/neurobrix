#!/usr/bin/env python3
"""Where a decode step's milliseconds go, counted at the dispatch point.

    python tools/metal_decode_profile.py --out <dir> [--max-tokens 6]

A decode step on this machine costs ~0.83 s against ~22 ms on the ATen path
for the same model on the same GPU. That is not a tuning gap, it is a
structural one, and this counts the things that could make it: how many
kernels are launched, how many are COMPILED (a cache that misses its key
recompiles every step), how many autotune sweeps run, how many host
synchronisations happen, and how many bytes cross between host and device —
each with the wall time it took.

Counted by wrapping the real call sites, not by sampling: the launcher's
`prepare` (which compiles and loads), its `launch`, the Metal driver's
dispatch (which is where a synchronisation would sit), the allocator's
`memcpy`, and Triton's autotuner.

Writes its files or refuses to conclude.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class Counters:
    def __init__(self):
        self.n = defaultdict(int)
        self.t = defaultdict(float)
        self.bytes = defaultdict(int)
        self.kernels = defaultdict(int)
        self.compiled = []
        self.step = 0
        self.per_step = []

    def snapshot(self):
        return {"counts": dict(self.n), "seconds": {k: round(v, 4) for k, v in self.t.items()},
                "bytes": dict(self.bytes)}

    def mark_step(self):
        self.per_step.append(self.snapshot())


def install(counters: Counters):
    """Wrap the real call sites. Nothing here changes what runs."""
    from neurobrix.kernels import launcher
    from neurobrix.kernels.nbx_tensor import DeviceAllocator

    # prepare(): the compile + load path. A miss is a compilation.
    _prepare = launcher.prepare

    def prepare(kernel, args, kwargs):
        name = getattr(kernel, "__name__", "?")
        cache, _key_cache, _backend = launcher._binder(kernel)
        before = len(cache)
        started = time.perf_counter()
        out = _prepare(kernel, args, kwargs)
        elapsed = time.perf_counter() - started
        counters.n["prepare"] += 1
        counters.t["prepare"] += elapsed
        if len(cache) > before:
            counters.n["compile"] += 1
            counters.t["compile"] += elapsed
            counters.compiled.append({"kernel": name, "seconds": round(elapsed, 4),
                                      "step": counters.step})
        return out
    launcher.prepare = prepare

    # launch(): one per kernel dispatch.
    _launch = launcher.launch

    def launch(kernel, grid, *args, **kwargs):
        name = getattr(kernel, "__name__", None) or getattr(
            getattr(kernel, "fn", None), "__name__", "?")
        started = time.perf_counter()
        try:
            return _launch(kernel, grid, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            counters.n["launch"] += 1
            counters.t["launch"] += elapsed
            counters.kernels[name] += 1
    launcher.launch = launch

    # The driver's dispatch: this is where a per-kernel host synchronisation
    # lives, if there is one.
    try:
        from neurobrix.triton import metal_driver

        _dispatch = metal_driver.MetalKernel._dispatch_params

        def dispatch(self, *a, **k):
            started = time.perf_counter()
            try:
                return _dispatch(self, *a, **k)
            finally:
                elapsed = time.perf_counter() - started
                counters.n["gpu_dispatch"] += 1
                counters.t["gpu_dispatch"] += elapsed
        metal_driver.MetalKernel._dispatch_params = dispatch
    except Exception:
        pass

    # Host <-> device copies.
    _memcpy = DeviceAllocator.memcpy

    def memcpy(dst, src, nbytes, kind=3):
        started = time.perf_counter()
        try:
            return _memcpy(dst, src, nbytes, kind=kind)
        finally:
            elapsed = time.perf_counter() - started
            counters.n[f"memcpy_kind{kind}"] += 1
            counters.t[f"memcpy_kind{kind}"] += elapsed
            counters.bytes[f"memcpy_kind{kind}"] += int(nbytes)
    DeviceAllocator.memcpy = staticmethod(memcpy)

    # Autotune sweeps.
    try:
        from triton.runtime.autotuner import Autotuner

        _bench = Autotuner._bench

        def bench(self, *a, **k):
            started = time.perf_counter()
            try:
                return _bench(self, *a, **k)
            finally:
                elapsed = time.perf_counter() - started
                counters.n["autotune_bench"] += 1
                counters.t["autotune_bench"] += elapsed
        Autotuner._bench = bench
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt",
                        default="Explain in one short paragraph why the sky appears blue.")
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--arm", default="triton")
    args = parser.parse_args()

    import shutil
    for path in (Path.home() / ".cache" / "triton_msl",
                 Path.home() / ".triton" / "cache"):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    counters = Counters()
    from neurobrix.kernels import launcher
    launcher.install()
    install(counters)

    # The per-token hook the engine already has, so a step boundary is the
    # engine's own and not a guess.
    import os
    progress = args.out / "progress.txt"
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["NBX_DECODE_PROGRESS"] = str(progress)
    os.environ["NBX_FORCE_RAND_SEED"] = "1234"

    import sys
    out_txt = args.out / "generated.txt"
    argv = ["neurobrix", "run", "--model", args.model, "--prompt", args.prompt,
            "--max-tokens", str(args.max_tokens), "--temperature", "0",
            f"--{args.arm}", "--output", str(out_txt)]
    started = time.perf_counter()
    saved = sys.argv
    sys.argv = argv
    try:
        from neurobrix.__main__ import main as nbx_main
        nbx_main()
    except SystemExit:
        pass
    finally:
        sys.argv = saved
    wall = time.perf_counter() - started

    steps = _read_steps(progress)
    document = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/metal_decode_profile.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "model": args.model, "arm": args.arm, "max_tokens": args.max_tokens,
        "wall_s": round(wall, 3),
        "decode_steps": steps,
        "totals": counters.snapshot(),
        "top_kernels_by_launch_count": sorted(
            counters.kernels.items(), key=lambda kv: -kv[1])[:25],
        "compilations": counters.compiled[:200],
        "compilation_count": len(counters.compiled),
    }
    (args.out / "profile.json").write_text(json.dumps(document, indent=1))
    (args.out / "PROFILE.md").write_text(_markdown(document))
    missing = [n for n in ("profile.json", "PROFILE.md")
               if not (args.out / n).exists()]
    if missing:
        print(f"REFUSING to conclude: {missing} not written")
        return 2
    print(f"\nwritten: {args.out / 'PROFILE.md'}")
    return 0


def _read_steps(progress: Path) -> list:
    import re
    steps = []
    try:
        for line in progress.read_text().splitlines():
            m = re.search(r"t=([0-9.]+) step=(\d+)", line)
            if m:
                steps.append((int(m.group(2)), float(m.group(1))))
    except OSError:
        return []
    out = []
    for i in range(1, len(steps)):
        out.append({"step": steps[i][0],
                    "seconds": round(steps[i][1] - steps[i - 1][1], 4)})
    return out


def _markdown(d) -> str:
    lines = []
    lines.append(f"# Where a decode step's time goes — `--{d['arm']}`")
    lines.append("")
    lines.append(f"Generated **{d['generated']}** by `{d['tool']}` on {d['machine']}.")
    lines.append("**No public claim is made from any number here.**")
    lines.append("")
    lines.append(f"* model {d['model']}, {d['max_tokens']} tokens, whole run "
                 f"**{d['wall_s']} s**")
    steps = d["decode_steps"]
    if steps:
        per = [s["seconds"] for s in steps]
        lines.append(f"* decode steps measured: {len(per)}, median "
                     f"**{sorted(per)[len(per)//2]:.3f} s**, min {min(per):.3f}, "
                     f"max {max(per):.3f}")
    lines.append("")
    lines.append("## Counted at the dispatch point, whole run")
    lines.append("")
    lines.append("| what | count | seconds | bytes |")
    lines.append("|---|---:|---:|---:|")
    t = d["totals"]
    for key in sorted(set(t["counts"]) | set(t["seconds"])):
        lines.append(f"| {key} | {t['counts'].get(key, 0)} | "
                     f"{t['seconds'].get(key, 0)} | "
                     f"{t['bytes'].get(key, '') if t['bytes'].get(key) else ''} |")
    lines.append("")
    lines.append(f"* MSL compilations: **{d['compilation_count']}**")
    lines.append("")
    lines.append("## Kernels by launch count")
    lines.append("")
    lines.append("| kernel | launches |")
    lines.append("|---|---:|")
    for name, n in d["top_kernels_by_launch_count"]:
        lines.append(f"| `{name}` | {n} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
