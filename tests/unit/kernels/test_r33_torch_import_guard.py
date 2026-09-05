"""R33 land gate: no torch reachable from the triton branch, and the exception
list can only shrink.

The 2026-08-30 provenance audit concluded CLEAN by inspecting a directory. It
was wrong. This guard is the executable form of the 2026-09-03 inventory
(`docs/audits/r33_torch_reachability_inventory_2026_09_03.md`) and it runs at
every land, like the soft-warning lint.

Two rules, and the second is what makes the debt decrease:

1. **A torch import under `kernels/` fails the gate** unless it is on the
   exception list below.
2. **An exception that is no longer needed ALSO fails the gate.** The list
   cannot quietly outlive its reason, so it shrinks by construction and never
   grows by habit — the failure message tells you to delete the entry.

What this does NOT cover, deliberately: torch pulled in by *upstream Triton*.
Touching `triton.runtime.driver.active` imports torch inside Triton's own CUDA
backend, and the autotuner allocates its benchmark buffers with torch. Verified
2026-09-03 by caller chain. That is a dependency of the tool, not of our code,
and no kernel we write can remove it. The runtime half of the audit
(`tools/torch_provenance.py`) attributes those correctly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
KERNELS = _SRC / "kernels"

# R33 is about the TRITON BRANCH, not about one directory. `kernels/` was the
# whole of it until the Metal port added `triton/metal_backend.py` and
# `kernels/metal_device.py`; scanning only `kernels/` would have let a torch
# import into the Metal driver unseen. Apple gets no exception: the container
# is NBXTensor there exactly as it is on CUDA, and a capability Metal needs
# and NBXTensor lacks is added to NBXTensor, never borrowed from torch.
TRITON_BRANCH = (KERNELS, _SRC / "triton")

# Every file under kernels/ still allowed to import torch, with the reason it
# is allowed and what would remove it. Adding a line here is a deliberate act
# that must survive review; removing one is the goal.
ALLOWED = {
    "nbx_tensor.py": (
        "the two documented NBX<->torch BOUNDARY converters "
        "(nbx_dtype_to_torch / nbx_to_torch), used at the edge to hand a "
        "result to the compiled pipeline. R33 forbids conversion mid-compute, "
        "not a hand-off at the boundary. Deferred imports, inside those two "
        "functions only."
    ),
    "metadata_ops.py": (
        "the COMPILED-mode metadata op set (view/reshape/gather/slice). Called "
        "only from core/runtime/graph_executor; the triton dispatcher has its "
        "own pure-Python metadata section. Since 2026-09-03 the kernels "
        "package no longer re-exports it eagerly, so importing a kernel no "
        "longer pulls it. Removing it entirely means porting those ops, which "
        "the triton path does not need."
    ),
    "ops/residual_chain_torch.py": (
        "the compiled half of the band-streamed residual chain, split out of "
        "residual_chain.py on 2026-09-03 precisely so the NBX half imports no "
        "torch. Imported only on the compiled branch of tiling_engine."
    ),
    "ops/fused_upsample_conv.py": (
        "three torch-only implementations (_fused_upsample_conv2d_torch, "
        "_tiled_conv2d_spatial_torch, _rms_norm_direct) plus the torch branch "
        "of tiled_rms_norm_spatial. All imports are DEFERRED and sit behind a "
        "type test that asks whether the input is an NBXTensor, so the triton "
        "path never reaches them. Removing this entry means moving those "
        "functions to a companion module as residual_chain did."
    ),
}


def _torch_importers() -> dict[str, list[int]]:
    """Every file under kernels/ that imports torch, and where.

    Parsed with `ast`, not grepped: a string or a comment mentioning torch is
    not an import, and the difference matters when the gate is what stops a
    regression.
    """
    found: dict[str, list[int]] = {}
    paths = sorted(q for root in TRITON_BRANCH for q in root.rglob("*.py"))
    for path in paths:
        if "triton_kernels_ref" in str(path):
            continue                      # reference tree, never executed
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:               # pragma: no cover
            continue
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "torch" or a.name.startswith("torch.")
                       for a in node.names):
                    lines.append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "torch"
                                    or node.module.startswith("torch.")):
                    lines.append(node.lineno)
        if lines:
            key = (str(path.relative_to(KERNELS))
                   if KERNELS in path.parents or path.parent == KERNELS
                   else str(path.relative_to(_SRC)))
            found[key] = sorted(lines)
    return found


# The Metal port's own modules. Named rather than merely covered by the sweep
# above, because a rule everyone agrees with is still worth an assertion that
# says which files it was written for.
METAL_TRITON_PATH = (
    "triton/metal_backend.py",
    "triton/metal_driver.py",
    "metal_device.py",
)


@pytest.mark.parametrize("filename", METAL_TRITON_PATH)
def test_the_metal_path_imports_no_torch(filename):
    """Apple is not an exception to R33.

    triton-msl's own driver imports torch in eight places and its zero-copy
    dispatch is built on `torch.mps`; taking any of it would put torch back in
    the execution path through the side door. We take its LOWERER — TTGIR to
    MSL, which is text — and compile, load and dispatch ourselves, on
    NBXTensor buffers from the Metal allocator.

    A file listed here that does not exist yet is not a failure: the pin is
    "if it exists, it is torch-free", so it guards the modules as they land.
    """
    path = _SRC / filename
    if not path.exists():
        pytest.skip(f"{filename} does not exist yet")
    found = _torch_importers()
    key = (str(path.relative_to(KERNELS))
           if path.parent == KERNELS else str(path.relative_to(_SRC)))
    assert key not in found, (
        f"{filename} imports torch (lines {found.get(key)}).\n"
        f"The Metal path is the triton branch: its container is NBXTensor and "
        f"its device work goes through the allocator. A capability it needs "
        f"and NBXTensor lacks is added to NBXTensor, in the seam, with pins — "
        f"never borrowed from torch."
    )


def test_the_gate_itself_catches_an_injected_import(tmp_path):
    """A green gate is worth nothing until it has been seen to go red.

    R33, 2026-09-05: *"un gate vert ne vaut que s'il a été vu échouer sur un
    import torch injecté."* This file spent its whole life passing on an empty
    subprocess — PYTHONPATH pointed at the repo root, the import raised
    ModuleNotFoundError, the assertion compared an empty string, and two
    failures sat classified as "pre-existing" on both machines for weeks. The
    lesson is not "fix the path", it is that a guard nobody has watched fail
    is a decoration.

    So the guard is exercised against a file that violates it, every run: a
    module carrying a torch import is written into the scanned tree, the
    scanner must name it, and it is removed again. If this ever passes while
    the scanner is blind, the whole file is worthless and this says so.
    """
    injected = _SRC / "triton" / "_r33_negative_control.py"
    assert not injected.exists(), (
        "the negative control's file already exists — a previous run did not "
        "clean up, and the gate may have been scanning it as real code"
    )
    injected.write_text(
        "# Written and deleted by test_the_gate_itself_catches_an_injected_import.\n"
        "# If you are reading this in a working tree, that test died mid-run.\n"
        "import torch  # noqa: F401\n"
    )
    try:
        found = _torch_importers()
        key = str(injected.relative_to(_SRC))
        assert key in found, (
            "THE GATE IS BLIND. A file importing torch was placed under "
            f"{injected.parent} and the scanner did not report it. Every "
            "green run of this file until now proved nothing."
        )
        assert found[key] == [3], (
            f"the scanner found the import but at the wrong line: {found[key]}"
        )
    finally:
        injected.unlink(missing_ok=True)


def test_the_gate_covers_the_dispatch_layer():
    """R33 names the dispatch layer explicitly, because that is where the
    vendor-agnostic launcher replacing Triton's `kernel[grid]` will live.

    A perimeter that stops at today's files would let the launcher land
    outside it on the day it is written, which is exactly the day it matters.
    """
    scanned = {q.resolve() for root in TRITON_BRANCH for q in root.rglob("*.py")}
    for required in (KERNELS / "dispatch.py", KERNELS / "wrappers.py",
                     KERNELS / "nbx_tensor.py"):
        assert required.resolve() in scanned, (
            f"{required.name} is not inside the R33 scan perimeter"
        )


def test_no_unlisted_torch_import_under_kernels():
    """The gate. A new torch import here is a regression of R33."""
    found = _torch_importers()
    unlisted = {f: n for f, n in found.items() if f not in ALLOWED}
    assert not unlisted, (
        "torch imported under kernels/ outside the R33 exception list:\n"
        + "\n".join(f"  {f} (lines {n})" for f, n in unlisted.items())
        + "\n\nA missing capability is added to NBXTensor and the house kernel "
          "family — never imported back from torch. If this import is genuinely "
          "compiled-only, split it into a companion module the way "
          "ops/residual_chain_torch.py was, and add it to ALLOWED with its "
          "reason and what would remove it."
    )


@pytest.mark.parametrize("filename", sorted(ALLOWED))
def test_every_exception_is_still_needed(filename):
    """The list shrinks by construction.

    An entry whose file no longer imports torch has been fixed, and leaving it
    here would let the list outlive its reasons and slowly become decoration."""
    found = _torch_importers()
    assert filename in found, (
        f"'{filename}' no longer imports torch — the exception is obsolete.\n"
        f"Delete its entry from ALLOWED in this file. The list is meant to "
        f"shrink; keeping a stale entry hides progress and invites new ones."
    )


def test_the_kernels_package_does_not_pull_torch():
    """Importing the package must not import torch.

    This is the check that actually caught the widest surface: removing torch
    from `ops/_configs.py` changed nothing, because `kernels/__init__` was
    eagerly re-exporting `execute_metadata_op` and pulling torch anyway. A
    static import list would not have seen it — only asking what a real import
    loads does.
    """
    import subprocess
    import sys

    src_root = KERNELS.parents[1]
    code = (
        "import sys, importlib;"
        "importlib.import_module('neurobrix.kernels');"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(src_root), "PATH": "/usr/bin:/bin"}, timeout=120,
    )
    assert out.stdout.strip().endswith("False"), (
        f"importing neurobrix.kernels pulled torch into the process.\n"
        f"stdout: {out.stdout!r}\nstderr: {out.stderr[-400:]!r}"
    )


def test_pure_kernel_modules_stay_torch_free():
    """The hot kernels, imported in isolation, must not pull torch.

    `ops/_configs` is the widest: rmsnorm, softmax, conv2d, matmul, baddbmm and
    depthwise_conv2d all import it. `matmul` is excluded — it queries the
    Triton target at module level, and `triton.runtime.driver.active` imports
    torch inside upstream Triton's own CUDA backend. That is the tool's
    dependency, not ours.
    """
    import subprocess
    import sys

    src_root = KERNELS.parents[1]
    for module in ("neurobrix.kernels.ops._configs",
                   "neurobrix.kernels.ops.rmsnorm",
                   "neurobrix.kernels.ops.residual_chain"):
        code = (f"import sys, importlib;"
                f"importlib.import_module('{module}');"
                f"print('torch' in sys.modules)")
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONPATH": str(src_root), "PATH": "/usr/bin:/bin"}, timeout=120,
        )
        assert out.stdout.strip().endswith("False"), (
            f"importing {module} pulled torch.\n"
            f"stderr: {out.stderr[-400:]!r}"
        )
