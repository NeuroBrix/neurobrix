"""The engine's Triton branch reaches a device only through nbxtensor (Hocine 2026-10-29 22:39): on Metal as anywhere,
memory, launches, device files, costs and precision go through the library — the branch imports no Metal module, no
Objective-C bridge and no native binding of a backend itself, and it calls the library's PUBLIC API only.

Read with `ast` over the Triton branch (`src/neurobrix/triton`) and its dispatch layer (`src/neurobrix/kernels`), never
imported, so it runs on any machine. The PyTorch branch (MPS) is apart, as decided.

1. No import of a Metal or native-binding module (`Metal`, `objc`, `Foundation`, `triton_apple_backend`,
   `metal_native`, ...) — none today; any is refused.
2. No import of a private name of nbxtensor (a name, or a module on its path, with a leading underscore): such an
   import says the library lacks a public API the engine needs — the library gains it (README, API.md), the engine
   calls it. TODAY'S SITES are listed below and the list SHRINKS ONLY (an exact match both ways).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOTS = (REPO / "src" / "neurobrix" / "triton", REPO / "src" / "neurobrix" / "kernels")
NATIVE = {"Metal", "objc", "Foundation", "CoreFoundation", "MetalPerformanceShaders", "triton_apple_backend",
          "metal_native"}

# What is left: each site names a public API nbxtensor must gain. Shrinks only. (file relative to src/neurobrix, name)
PRIVATE_SITES = {
    ("kernels/census.py", "nbxtensor.ops._configs"),
    ("kernels/ops/fused_upsample_conv_torch.py", "nbxtensor.ops.fused_upsample_conv._fused_upsample_conv2d_nbx"),
    ("kernels/ops/fused_upsample_conv_torch.py", "nbxtensor.ops.fused_upsample_conv._pair"),
    ("kernels/ops/fused_upsample_conv_torch.py", "nbxtensor.ops.fused_upsample_conv._tiled_conv2d_spatial_nbx"),
    ("triton/flow/autoregressive.py", "nbxtensor.ops._configs.active_vendor_profile"),
    ("triton/state_fence.py", "nbxtensor.backend.triton_ext_driver._native"),
}


def _files(roots=ROOTS):
    for root in roots:
        for p in sorted(root.rglob("*.py")):
            if "triton_kernels_ref" not in p.parts:
                yield p


def _rel(p: Path) -> str:
    parts = p.parts
    return "/".join(parts[parts.index("neurobrix") + 1:])


def native_imports(roots=ROOTS) -> list:
    found = []
    for p in _files(roots):
        for n in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            found += [f"{_rel(p)}:{n.lineno} {m}" for m in names if m.split(".")[0] in NATIVE]
    return found


def private_sites(roots=ROOTS) -> set:
    found = set()
    for p in _files(roots):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("nbxtensor"):
                private_module = any(part.startswith("_") for part in n.module.split("."))
                for a in n.names:
                    if private_module or a.name.startswith("_"):
                        found.add((_rel(p), f"{n.module}.{a.name}"))
    return found


def test_the_triton_branch_imports_no_metal_or_native_binding():
    found = native_imports()
    assert not found, "the Triton branch reaches a device only through nbxtensor:\n  " + "\n  ".join(found)


def test_no_new_private_name_of_the_library_is_reached():
    new = sorted(private_sites() - PRIVATE_SITES)
    assert not new, ("the engine reaches a PRIVATE name of nbxtensor — give the library the public API it lacks "
                     "(README, API.md) and call that:\n  " + "\n  ".join(f"{f}: {n}" for f, n in new))


def test_the_list_of_private_sites_only_shrinks():
    gone = sorted(PRIVATE_SITES - private_sites())
    assert not gone, "these sites now call public API — delete them from PRIVATE_SITES:\n  " + "\n  ".join(map(str, gone))


def test_the_guard_sees_what_it_refuses(tmp_path):
    pkg = tmp_path / "neurobrix" / "triton"
    pkg.mkdir(parents=True)
    (pkg / "x.py").write_text("import objc\nfrom triton_apple_backend import metal_native\n"
                              "from nbxtensor.backend.runtime import _private_helper\nfrom nbxtensor.shape import numel\n")
    assert [s.split(" ", 1)[1] for s in native_imports([pkg])] == ["objc", "triton_apple_backend"]
    assert private_sites([pkg]) == {("triton/x.py", "nbxtensor.backend.runtime._private_helper")}
