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

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
KERNELS = SRC / "kernels"          # the dispatch layer: dispatch, wrappers, adapter, classification, ops/
TRITON = SRC / "triton"            # the whole triton branch, Metal backend included

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


def _torch_importers(root: Path = KERNELS) -> dict[str, list[int]]:
    """Every file under `root` that imports torch, and where.

    Parsed with `ast`, not grepped: a string or a comment mentioning torch is
    not an import, and the difference matters when the gate is what stops a
    regression.
    """
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
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
            found[str(path.relative_to(root))] = sorted(lines)
    return found


# The triton branch: a pin per file, and the pinned set is EMPTY. Any file
# listed here would be a file the owner's 2026-09-05 rule forbids outright.
TRITON_ALLOWED: dict[str, str] = {}
METAL_FILES = ("metal_backend.py",)


def test_no_torch_import_anywhere_under_triton():
    """The owner's universal R33 (2026-09-05): zero torch under src/neurobrix/triton,
    the Metal backend included, conditional and dead-branch imports counted."""
    found = _torch_importers(TRITON)
    unlisted = {f: n for f, n in found.items() if f not in TRITON_ALLOWED}
    assert not unlisted, (
        "torch imported under src/neurobrix/triton:\n"
        + "\n".join(f"  {f} (lines {n})" for f, n in unlisted.items())
        + "\n\nA missing capability is added to NBXTensor and the allocator in the seam, "
          "never substituted with torch — on every backend."
    )


@pytest.mark.parametrize("filename", METAL_FILES)
def test_the_metal_files_exist_and_carry_no_torch(filename):
    path = TRITON / filename
    assert path.exists(), f"{filename} is not where the gate expects it — update METAL_FILES"
    assert filename not in _torch_importers(TRITON)


def test_the_gate_is_seen_failing_on_an_injected_import(tmp_path):
    """A green gate is worth something only once it has been seen failing:
    the same scanner over a tree carrying an injected torch import must name
    it — a conditional one, a deferred one inside a function, and a dead
    branch alike (the owner's rule counts all three)."""
    (tmp_path / "clean.py").write_text("x = 1\n")
    (tmp_path / "conditional.py").write_text("if False:\n    import torch\n")
    (tmp_path / "deferred.py").write_text("def f():\n    from torch import cuda\n    return cuda\n")
    (tmp_path / "dead.py").write_text("try:\n    pass\nexcept Exception:\n    import torch.nn as nn\n")
    found = _torch_importers(tmp_path)
    assert set(found) == {"conditional.py", "deferred.py", "dead.py"}, found
    assert found["conditional.py"] == [2] and found["deferred.py"] == [2]


def test_the_runtime_probe_is_seen_failing_on_an_injected_import():
    """The dynamic half: the same subprocess probe, given a module that
    imports torch, must report True — otherwise a False from it means nothing."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c", "import sys, torch; print('torch' in sys.modules)"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "True", out.stderr[-300:]


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

    src = KERNELS.parents[1]          # the package lives under src/, not at the repo root
    code = (
        "import sys, importlib;"
        "importlib.import_module('neurobrix.kernels');"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"}, timeout=120,
    )
    # A harness that cannot import the package proves nothing: the 2026-09-05
    # addendum found this gate dying on ModuleNotFoundError at every land
    # (PYTHONPATH on the repo root) and its assertion failing on an empty
    # string — two "pre-existing" failures that were the harness itself.
    assert out.returncode == 0, f"the probe process failed:\n{out.stderr[-600:]}"
    assert out.stdout.strip() == "False", (
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

    src = KERNELS.parents[1]          # the package lives under src/, not at the repo root
    for module in ("neurobrix.kernels.ops._configs",
                   "neurobrix.kernels.ops.rmsnorm",
                   "neurobrix.kernels.ops.residual_chain"):
        code = (f"import sys, importlib;"
                f"importlib.import_module('{module}');"
                f"print('torch' in sys.modules)")
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"}, timeout=120,
        )
        assert out.returncode == 0, f"the probe process failed for {module}:\n{out.stderr[-600:]}"
        assert out.stdout.strip() == "False", (
            f"importing {module} pulled torch.\n"
            f"stderr: {out.stderr[-400:]!r}"
        )
