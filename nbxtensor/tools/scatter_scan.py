#!/usr/bin/env python3
"""Where does tensor semantics live outside the nbxtensor library?

NBXTensor is meant to be ONE library: the dtype table, shapes, strides and
views, broadcasting, type promotion, allocators and memory plans, device
backends, kernels and their cost. The engine (Prism, the DtypeEngine's policy,
the flows, the execution modes) should only IMPORT it. This tool finds the
definitions that still live in the engine, so the migration can be measured and
the list of exceptions can only shrink.

It reads code with `ast`, never with grep: a comment or a string that mentions
"stride" is not a definition. It walks every function and class (nested ones
included) under the engine roots and reports those whose name is tensor
semantics, plus the whole modules the design moves into the library.

    python3 nbxtensor/tools/scatter_scan.py            # markdown table
    python3 nbxtensor/tools/scatter_scan.py --tsv      # the migration list format

The migration list (docs/nbxtensor/migration_list.tsv) is the committed output
of `--tsv`, reviewed by hand. The guard test compares the two: a definition
found but not listed fails (the scatter grew), a definition listed but no
longer found fails (it moved: delete its line).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "src" / "neurobrix"

# The engine roots the Triton branch runs through. The PyTorch branch
# (core/runtime/graph, core/flow, kernels/metadata_ops.py ...) is a separate
# product by design and is NOT a migration target: torch is its tensor
# library. Its files are skipped.
ROOTS = ("triton", "kernels", "core/prism", "core/memory", "core/module",
         "core/runtime", "core/state")
SKIP_PARTS = ("triton_kernels_ref", "__pycache__")
PYTORCH_BRANCH_PREFIXES = ("core/runtime/graph/",)
PYTORCH_BRANCH_FILES = {"kernels/metadata_ops.py", "core/runtime/torch_allocator.py",
                        "core/runtime/graph_executor.py",
                        # the compiled halves split out of the kernel files: they stay with the engine (M1)
                        "kernels/ops/residual_chain_torch.py", "kernels/ops/fused_upsample_conv_torch.py",
                        "kernels/torch_interop.py"}

# Names the pattern catches that are NOT tensor semantics (reviewed by hand;
# a line here says why, and the list only grows with a reason).
NOT_SEMANTIC = {
    ("core/prism/chunked_region.py", "region_carriers"): "a region's carrier tensors, Prism's tiling plan",
    ("core/module/tiling_engine.py", "_detect_pixel_shuffle_broadcast_chains"): "graph pattern detection",
    ("triton/promotion.py", "_promote_int"): "symbolic seq_len promotion of trace constants (a graph pass)",
    ("triton/promotion.py", "_promote_int_value"): "symbolic seq_len promotion of trace constants (a graph pass)",
    ("core/prism/solver.py", "_arena_over_allocation"): "Prism's decision on a fact of the library (the arena's "
        "bytes, nbxtensor.memory.layout; the device's largest allocation, the profile) — COST.md section 2",
    ("core/prism/solver.py", "_store_arena_over_allocation"): "Prism's decision on a fact of the library (an expert "
        "store's dense arena, nbxtensor.memory.layout) — COST.md section 2",
    ("core/prism/solver.py", "_arena_door"): "Prism's policy over a plan, reading the two facts above — COST.md "
        "section 2",
    ("core/prism/solver.py", "_arena_mb_of"): "reads the arena bytes the library laid out (ComponentMemory.arena_bytes) "
        "and refuses where none was",
    ("triton/weight_loader.py", "_load_shard_into_arenas"): "the engine's loading policy over one shard (which weight on "
        "which device or the host, at which dtype, widened or not; DESIGN.md section 3: the engine keeps its loading "
        "policy); every read, conversion and placement is nbxtensor.io.safetensors",
}

# Names that are tensor semantics wherever they are defined.
SEMANTIC = re.compile(
    r"(stride|broadcast|itemsize|dtype_size|bytes_per_elem|result_type"
    r"|promote_(int|types?|dtype|func)|contiguous|numel|nbytes"
    r"|view_strides|can_view|bf16_|float32_to_bf16|carrier"
    r"|malloc|allocator|arena|memory_pool|pinned|host_slab"
    r"|copy_nd|strided_copy|strided_scatter|fill_constant|slice_bounds)",
    re.IGNORECASE,
)

# Whole modules the design moves into the library (docs/nbxtensor/DESIGN.md §3).
MODULES_TO_LIBRARY = {
    "kernels/nbx_tensor.py": "tensor + memory + backend (split in M2)",
    "kernels/metal_device.py": "memory/metal + backend/metal",
    "kernels/quantized_tensor.py": "quant",
    "kernels/launch_keys.py": "ops: launch decisions as pure functions",
    "kernels/launcher.py": "backend: kernel launcher",
    "kernels/autotune_bucket.py": "ops/autotune",
    "kernels/autotune_certified.py": "ops/autotune",
    "kernels/autotune_refusals.py": "ops/autotune",
    "kernels/screen_oracle.py": "ops/autotune",
    "nbxtensor/random/rng_stream.py": "random",
    "nbxtensor/random/seeded_draw.py": "random",
    "nbxtensor/random/mt_randn.py": "random",
    "nbxtensor/random/rng_pin.py": "random",
    "kernels/oom_advice.py": "memory",
    "kernels/wrappers.py": "ops (split by op family in M2)",
    "kernels/dispatch.py": "ops: the op table (its view rule goes to shape/)",
    "triton/itemsize.py": "dtype: the one table reader",
    "triton/arena.py": "memory",
    "triton/memory_pool.py": "memory",
    "triton/device_transfer.py": "memory/transfer",
    "triton/launcher_contract.py": "backend",
    "triton/triton_ext_driver.py": "backend/metal",
    "triton/metal_backend.py": "backend/metal",
    "triton/cpu_backend.py": "backend/cpu",
    "triton/replay_graph.py": "backend: graph capture primitive (the engine keeps FrozenPlan)",
}


def _files():
    for root in ROOTS:
        base = ENGINE / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if any(s in p.parts for s in SKIP_PARTS):
                continue
            yield p


def scan():
    rows = []
    ops_left = False
    for p in _files():
        rel = str(p.relative_to(ENGINE))
        if rel in PYTORCH_BRANCH_FILES or rel.startswith(PYTORCH_BRANCH_PREFIXES):
            continue
        if rel in MODULES_TO_LIBRARY:
            rows.append((rel, "*", MODULES_TO_LIBRARY[rel]))
            continue
        if rel.startswith("kernels/ops/"):
            if p.name != "__init__.py":
                ops_left = True   # a Triton kernel file still in the engine: the directory is still to move
            continue  # the kernel files move as a directory (ops/)
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (SEMANTIC.search(node.name) and node.name not in seen
                        and (rel, node.name) not in NOT_SEMANTIC):
                    seen.add(node.name)
                    rows.append((rel, node.name, f"line {node.lineno}"))
    if ops_left:
        rows.append(("kernels/ops/", "*", "ops: every kernel file"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="store_true")
    a = ap.parse_args()
    rows = scan()
    if a.tsv:
        for r in rows:
            print("\t".join(r))
        return 0
    mods = [r for r in rows if r[1] == "*"]
    defs = [r for r in rows if r[1] != "*"]
    print(f"# Tensor semantics outside nbxtensor: {len(mods)} modules to move, "
          f"{len(defs)} definitions scattered in the engine\n")
    print("| file | definition | where / destination |\n|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
