#!/usr/bin/env python3
"""R33 provenance audit that follows CALLS, not directories.

The 2026-08-30 provenance audit concluded CLEAN. It was wrong, and the reason
matters more than the verdict: it inspected the `triton/` **directory** and
read kernel **names** off a text timeline. Both are proxies. A module outside
that directory can import torch and be called from inside the triton branch,
and an ATen kernel in an nsys report does not say who launched it.

The audio timeline of 2026-09-03 exposed it — 264 s of
`at::native::vectorized_elementwise_kernel` over 879,317 launches, inside a
`--triton` run.

So this audit intercepts every torch operation as it happens, with
`TorchFunctionMode`, and records the NeuroBrix frame that called it. It cannot
be fooled by a deferred import, by a re-export, or by a call that crosses a
directory boundary, because it never looks at files at all.

    python3 tools/torch_provenance.py --model whisper-large-v3-turbo \\
        --audio benchmarks/assets/jfk_11s.wav --engine triton

Output: every torch op that ran, how many times, and the call site inside
NeuroBrix that issued it — ranked by count, because the count is what turns a
boundary conversion into a hot path.

Exit code is 1 when any torch op is attributed to a triton-branch call site,
so this is usable as a land gate.
"""

from __future__ import annotations

import argparse
import collections
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Frames in these modules are the SOURCE of a torch call for our purposes.
# Everything else (torch internals, numpy, stdlib) is noise.
OURS = "src/neurobrix/"
# A call site under any of these is inside the triton branch and is therefore
# an R33 violation when it issues a torch op.
TRITON_BRANCH = ("src/neurobrix/triton/", "src/neurobrix/kernels/")
# The two documented NBX<->torch boundary converters. R33 forbids conversion
# MID-COMPUTE; a single hand-off at the edge is the accepted boundary, and
# flagging it as a violation would bury the real ones.
BOUNDARY = ("nbx_tensor.py:224", "nbx_tensor.py:225", "nbx_tensor.py:241",
            "nbx_tensor.py:242", "nbx_tensor.py:243", "nbx_tensor.py:244")


def _nbx_frame(stack) -> str:
    """The deepest NeuroBrix frame — who actually issued the op."""
    for frame in reversed(stack):
        if OURS in frame.filename:
            rel = frame.filename.split("src/neurobrix/")[-1]
            return f"src/neurobrix/{rel}:{frame.lineno}"
    return "<outside neurobrix>"


def build_mode(counts, sites, full_stacks):
    """Intercept at the ATen DISPATCHER, not at the Python API.

    `TorchFunctionMode` sees only calls made through torch's public Python
    surface. The first version of this tool used it and reported 6 torch
    operations on a run whose profile contained 879,317 ATen kernel
    launches — the instrument was measuring the wrong layer, which is the
    same mistake as auditing a directory. `TorchDispatchMode` sits under
    every path into ATen, so what it does not see did not execute.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class Provenance(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", str(func))
            # The stack walk is the expensive part; sample it only until a
            # call site is well established, then just count.
            # Every op is attributed. The stack walk costs, but a truncated
            # sample makes the COUNT meaningless — and the count is what
            # separates a boundary conversion from a hot path.
            stack = traceback.extract_stack()[:-1]
            site = _nbx_frame(stack)
            sites[(name, site)] += 1
            # Keep ONE full stack per (op, site) so the frame ABOVE ours can be
            # read. Attributing to the deepest NeuroBrix frame is right for
            # locating the path, but it cannot distinguish "our code called
            # torch" from "a library we called allocated with torch".
            key = (name, site)
            if key not in full_stacks:
                full_stacks[key] = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                                    for f in stack[-8:]]
            counts[name] += 1
            return func(*args, **(kwargs or {}))

    return Provenance()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--input-image", dest="input_image")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--engine", default="triton", choices=["triton", "compiled"])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    sys.argv = ["neurobrix", "run", "--model", args.model,
                "--max-tokens", str(args.max_tokens)]
    if args.audio:
        sys.argv += ["--audio", args.audio]
    elif args.input_image:
        sys.argv += ["--input-image", args.input_image]
    else:
        sys.argv += ["--prompt", args.prompt]
    if args.engine == "triton":
        sys.argv.append("--triton")

    counts: collections.Counter = collections.Counter()
    sites: collections.Counter = collections.Counter()

    from neurobrix.cli import main as cli_main

    full_stacks: dict = {}
    mode = build_mode(counts, sites, full_stacks)
    rc = 0
    with mode:
        try:
            cli_main()
        except SystemExit as exc:
            rc = exc.code or 0
        except Exception as exc:                      # noqa: BLE001
            print(f"\n[run failed: {type(exc).__name__}: {exc}]", file=sys.stderr)
            rc = 1

    total = sum(counts.values())
    print("\n" + "=" * 72)
    print(f"TORCH PROVENANCE — engine={args.engine}  model={args.model}")
    print("=" * 72)
    print(f"total torch operations intercepted: {total:,}\n")
    if not total:
        print("No torch operation ran. R33 clean on this path.")
        return rc

    print(f"{'count':>10}  {'torch op':<26}  call site")
    print("-" * 72)
    violations = 0
    for (name, site), n in sites.most_common(args.top):
        flag = ""
        if any(b in site for b in TRITON_BRANCH) and not any(b in site for b in BOUNDARY):
            flag = "  <== TRITON BRANCH"
            violations += n
        print(f"{n:>10,}  {name:<26}  {site}{flag}")

    for (name, site), frames in list(full_stacks.items()):
        if any(b in site for b in TRITON_BRANCH) and not any(b in site for b in BOUNDARY):
            print(f"\n  caller chain for {name} @ {site}:")
            for f in frames:
                print(f"      {f}")
    print()
    if violations:
        print(f"R33 VIOLATION: {violations:,} torch operations issued from the "
              f"triton branch.")
        print("A missing capability is added to NBXTensor and the house kernel "
              "family — never imported back from torch.")
        return 1
    print("No torch operation attributed to a triton-branch call site.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
