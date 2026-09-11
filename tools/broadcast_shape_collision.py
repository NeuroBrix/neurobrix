#!/usr/bin/env python3
"""Which of our kernels carry the shape that upstream issue #9 mis-lowers.

The defect is NOT in the bitonic sort. Measured 2026-09-10: the middle-axis
index of a `tt.broadcast` is derived from the SOURCE shape without the target
shape, so two broadcasts from a common source shape to different targets get
one index expression and the last one emitted decides for both. The sort is
where it was found; its scope is every kernel with that shape.

We have no idea how many of ours carry it, so this counts rather than assumes.
Two passes, because they answer different questions and only the pair is
useful:

  --scan     an UPPER BOUND over the tree, and nothing better. AST over the
             kernel sources, counting bodies with two or more
             broadcast-producing expressions. It does NOT discriminate and
             must not be read as a candidate list: `offs_m[:, None]` and
             `offs_n[None, :]` are two broadcasts in almost every 2-D kernel
             we have, and their source shapes DIFFER, which is precisely the
             case that is fine. The trigger needs identical source shapes,
             and shapes are constexpr-dependent, so no reading of the source
             can decide it. The bound is worth printing only to say how large
             the question is.

  --census   what a MODEL reaches. Wraps triton's compiler in-process, runs
             the model, and reports, for every kernel actually compiled,
             whether its TTIR holds two `tt.broadcast` ops with the SAME
             source shape and DIFFERENT result shapes. That is the exact
             trigger, read from the IR rather than guessed from the source.

A pattern in a kernel nothing executes and a pattern on the decode path are
not the same finding, and only the count tells them apart.

    tools/broadcast_shape_collision.py --scan
    tools/broadcast_shape_collision.py --census <model> [--arm triton]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ir_census import compile_census, refuse_if_empty        # noqa: E402

KERNELS = Path(__file__).resolve().parents[1] / "src" / "neurobrix" / "kernels"

#: Every way a Triton body can produce a broadcast. `broadcast_to` is the
#: explicit one; `expand_dims`/`[None]` followed by an arithmetic op against a
#: wider tensor produces an implicit `tt.broadcast` just the same, which is why
#: the scan cannot be limited to the explicit call.
_EXPLICIT = ("broadcast_to",)


def _is_triton_jit(node: ast.FunctionDef) -> bool:
    for d in node.decorator_list:
        src = ast.unparse(d)
        if "triton.jit" in src or src.endswith("jit"):
            return True
    return False


def _broadcast_sites(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    sites = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if any(name.endswith(e) for e in _EXPLICIT):
                sites.append((node.lineno, ast.unparse(node)[:90]))
        # `x[None, :, None]` — an expand that only exists to be broadcast
        elif isinstance(node, ast.Subscript):
            sub = ast.unparse(node.slice)
            if sub.count("None") >= 1 and ":" in sub:
                sites.append((node.lineno, ast.unparse(node)[:90]))
    return sites


def scan(verbose: bool = False) -> int:
    hits: dict[str, list] = {}
    total = 0
    for path in sorted(KERNELS.rglob("*.py")):
        if "triton_kernels_ref" in path.parts:
            continue                      # vendored reference tree, not ours
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not _is_triton_jit(node):
                continue
            total += 1
            sites = _broadcast_sites(node)
            if len(sites) >= 2:
                rel = path.relative_to(KERNELS.parents[2])
                hits[f"{rel}::{node.name}"] = sites
    print(f"@triton.jit kernels in the tree      : {total}")
    print(f"with 2+ broadcast-producing sites    : {len(hits)}")
    print()
    print("This is an UPPER BOUND and it does not discriminate. Two broadcasts")
    print("collide only if their SOURCE shapes are identical and their targets")
    print("differ; `offs_m[:, None]` with `offs_n[None, :]` is two broadcasts")
    print("from DIFFERENT source shapes and is the ordinary, correct case that")
    print("almost every 2-D kernel here contains. Shapes are constexpr-")
    print("dependent, so the source cannot settle it — only the compiled IR")
    print("can. Use --census.")
    if verbose:
        print()
        for name, sites in sorted(hits.items()):
            print(f"{name}   ({len(sites)} sites)")
    return 0


_BCAST = re.compile(
    r"tt\.broadcast\s+\S+\s*:\s*tensor<([0-9x]+)x[a-z0-9]+>\s*->\s*tensor<([0-9x]+)x[a-z0-9]+>")


def _expanded_axes(src: str, dst: str) -> tuple:
    """Which axes this broadcast expands. `1x2x1 -> 1x2x4` expands axis 2."""
    a, b = src.split("x"), dst.split("x")
    if len(a) != len(b):
        return ("rank", src, dst)          # a rank change: not comparable
    return tuple(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


def collisions_in(ttir: str) -> list[tuple[str, list[str]]]:
    """(source shape, [result shapes]) for two broadcasts of ONE source shape
    that expand DIFFERENT axes — the trigger, not merely the resemblance.

    Keying on "same source, different target" alone is far too loose, and the
    first census run proved it: TinyLlama's decode path reported three carriers
    whose pairs were `32x1 -> 32x32` with `32x1 -> 32x64`. Both expand axis 1,
    whose source extent is 1, so the index contribution is zero in both and a
    shared expression is harmless. Reporting those as carriers would have been
    a false non-zero on the path that matters most — worse than reporting
    nothing, because it would have been acted on.

    What made the upstream case collide is that `1x2x1` was expanded along
    axis 2 for one target (`1x2x4`) and along axis 0 for the other (`4x2x1`):
    two different index expressions are owed, and one was emitted.
    """
    by_src: dict[str, dict] = defaultdict(dict)
    for src, dst in _BCAST.findall(ttir):
        by_src[src][dst] = _expanded_axes(src, dst)
    out = []
    for src, dsts in sorted(by_src.items()):
        if len(set(dsts.values())) > 1:
            out.append((src, sorted(dsts)))
    return out


def census(model: str, arm: str, out: Path | None) -> int:
    """The wrapper is `ir_census.compile_census`; only the predicate is ours.

    It was a copy of the same forty lines until the `other` census needed them
    a second time. Widening the brick rather than adding one means the
    "patch BOTH compiler bindings" lesson and the "a run that compiled nothing
    is not a count of zero" refusal are now written once and inherited here,
    instead of being re-derived — and re-forgotten — per tool.
    """
    seen = compile_census(model, arm, lambda ttir: [
        {"source": src, "targets": dsts} for src, dsts in collisions_in(ttir)])
    rc = refuse_if_empty(seen, model, arm, "the collision")
    if rc is not None:
        return rc
    carriers = {k: v for k, v in seen.items() if v["findings"]}
    print()
    print(f"kernels compiled by {model} ({arm}) : {len(seen)}")
    print(f"carrying the collision shape        : {len(carriers)}")
    for name, row in sorted(carriers.items()):
        print(f"  {name}  ({row['compilations']} compilation(s))")
        for c in row["findings"]:
            print(f"      source {c['source']} -> {', '.join(c['targets'])}")
    if out:
        out.write_text(json.dumps(
            {"model": model, "arm": arm, "kernels": seen}, indent=1))
        print(f"\nwritten: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="list the bodies behind the bound")
    ap.add_argument("--census", metavar="MODEL")
    ap.add_argument("--arm", default="triton")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.scan:
        return scan(args.verbose)
    if args.census:
        return census(args.census, args.arm, args.out)
    ap.error("give --scan or --census MODEL")


if __name__ == "__main__":
    sys.exit(main())
