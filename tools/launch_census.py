#!/usr/bin/env python3
"""Count kernel launches without a profiler, and say who issued them.

The audio timeline of 2026-09-03 had to retract its headline: it reported
triton as 92x slower than compiled, and that number was the INSTRUMENT. nsys
charges roughly 230 us per launch; the triton path issues ~2 M launches, so
the profiler's own overhead was the 460 s it reported. The count survived —
1,985,476 against compiled's 9,856 — because a count is not distorted by the
cost of counting.

So this tool measures the thing that survived, and nothing else. A dict
increment per launch is ~0.1 us against nsys's ~230 us: three orders of
magnitude cheaper, which is what makes the wall-clock still meaningful while
it runs.

Attribution is sampled, deliberately. Walking the Python stack costs ~10 us,
which at two million launches would be 20 s of pure instrument — the exact
mistake being corrected. So the COUNT is exact for every kernel and the CALL
SITE is captured only for the first few occurrences of each kernel. You get
an exact census plus a truthful witness of where each kernel comes from.

It also separates two things the nsys report could not:

  * launches issued by OUR code, versus
  * launches issued by upstream Triton's AUTOTUNER benchmarking configs.

That distinction is the open question left by the timeline: the report showed
879,317 `cudaLaunchKernel` calls into an ATen elementwise kernel inside a
`--triton` run, while a dispatcher-level provenance audit of the same path
intercepted ZERO torch operations. Both cannot be true of the same execution
unless the two runs differed — and they did, in whether the autotune cache was
warm. `--cold-autotune` reproduces the difference on purpose.

    python3 tools/launch_census.py --model whisper-large-v3-turbo \\
        --audio benchmarks/assets/jfk_11s.wav --engine triton
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OURS = "src/neurobrix/"
# Frames inside these modules mean the launch came from upstream Triton's
# autotuner benchmarking a config, not from our execution of the model.
AUTOTUNER = ("triton/runtime/autotuner.py", "triton/testing.py")
STACKS_PER_KERNEL = 3


def _site(stack) -> tuple[str, bool]:
    """(deepest NeuroBrix frame, whether an autotuner frame is above it)."""
    from_autotuner = any(a in f.filename for f in stack for a in AUTOTUNER)
    for frame in reversed(stack):
        if OURS in frame.filename:
            rel = frame.filename.split("src/neurobrix/")[-1]
            return f"src/neurobrix/{rel}:{frame.lineno}", from_autotuner
    return "<outside neurobrix>", from_autotuner


class Census:
    def __init__(self):
        self.counts: collections.Counter = collections.Counter()
        self.by_autotuner: collections.Counter = collections.Counter()
        self.sites: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        self.stacks_taken: collections.Counter = collections.Counter()
        self.torch_ops: collections.Counter = collections.Counter()

    def install(self):
        """Wrap the two launch entry points and the ATen dispatcher.

        `JITFunction.run` is where every Triton launch lands, including the
        autotuner's own benchmark launches — which is exactly why it is the
        right hook: a census that could not see them would attribute the
        autotuner's work to the model.
        """
        import triton
        from triton.runtime.jit import JITFunction

        original = JITFunction.run
        census = self

        def counting_run(self, *args, **kwargs):
            name = getattr(self, "__name__", "<jit>")
            census.counts[name] += 1
            if census.stacks_taken[name] < STACKS_PER_KERNEL:
                census.stacks_taken[name] += 1
                site, auto = _site(traceback.extract_stack()[:-1])
                census.sites[name][("autotuner:" if auto else "") + site] += 1
                if auto:
                    census.by_autotuner[name] += 1
            return original(self, *args, **kwargs)

        JITFunction.run = counting_run

    def install_torch(self):
        """Count ATen operations too, so the two questions are answered by ONE
        run rather than by comparing two runs that may differ."""
        from torch.utils._python_dispatch import TorchDispatchMode

        census = self

        class Counter(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                census.torch_ops[getattr(func, "__name__", str(func))] += 1
                return func(*args, **(kwargs or {}))

        return Counter()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--engine", default="triton", choices=["triton", "compiled"])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--cold-autotune", action="store_true",
                    help="clear the Triton autotune cache first, so the "
                         "autotuner's benchmarking launches are included")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cold_autotune:
        import shutil
        for cache in (Path.home() / ".triton" / "cache",
                      Path.home() / ".neurobrix" / "autotune"):
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
                print(f"[cold] cleared {cache}")

    sys.argv = ["neurobrix", "run", "--model", args.model,
                "--max-tokens", str(args.max_tokens)]
    if args.audio:
        sys.argv += ["--audio", args.audio]
    else:
        sys.argv += ["--prompt", args.prompt]
    if args.engine == "triton":
        sys.argv.append("--triton")

    census = Census()
    census.install()

    from neurobrix.cli import main as cli_main

    rc, t0 = 0, time.perf_counter()
    with census.install_torch():
        try:
            cli_main()
        except SystemExit as exc:
            rc = exc.code or 0
        except Exception as exc:                          # noqa: BLE001
            print(f"\n[run failed: {type(exc).__name__}: {exc}]", file=sys.stderr)
            rc = 1
    wall = time.perf_counter() - t0

    total = sum(census.counts.values())
    torch_total = sum(census.torch_ops.values())
    print("\n" + "=" * 78)
    print(f"LAUNCH CENSUS — {args.model}  engine={args.engine}"
          f"{'  COLD autotune' if args.cold_autotune else '  warm autotune'}")
    print("=" * 78)
    print(f"wall (instrumented, no profiler) : {wall:8.2f} s")
    print(f"triton kernel launches           : {total:>10,}")
    print(f"ATen operations dispatched       : {torch_total:>10,}")
    print()
    print(f"{'launches':>12}  {'kernel':<34} issued from")
    print("-" * 78)
    for name, n in census.counts.most_common(20):
        site = census.sites[name].most_common(1)
        where = site[0][0] if site else "?"
        print(f"{n:>12,}  {name:<34} {where}")
    if torch_total:
        print(f"\n{'count':>12}  ATen operation")
        print("-" * 78)
        for name, n in census.torch_ops.most_common(12):
            print(f"{n:>12,}  {name}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "launch_census.json").write_text(json.dumps({
            "model": args.model, "engine": args.engine,
            "cold_autotune": args.cold_autotune, "wall_s": wall,
            "triton_launches": total, "aten_ops": torch_total,
            "by_kernel": dict(census.counts),
            "aten_by_op": dict(census.torch_ops),
            "sites": {k: dict(v) for k, v in census.sites.items()},
        }, indent=2))
        print(f"\nwritten: {out/'launch_census.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
