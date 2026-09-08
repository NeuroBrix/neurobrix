"""The vendorless layer must not branch on one vendor's device prefix.

`core/prism/structure.py` states the rule at the top of the file: *VENDORLESS:
nvidia/amd/intel -> cuda/hip/xpu device strings*. Device strings therefore
appear in `core/` legitimately — but they are supposed to arrive from
`get_device_prefix(vendor, architecture)`, not from a literal comparison
against one vendor.

A literal comparison is not a style problem. `startswith("cuda")` in a sum
over devices returns 0 on an Apple or AMD machine, and the branch it guards
then takes the "no GPU" path on hardware that has one — silently, with no
error anywhere. This walks the AST and fails on that shape, so the next one
is caught when it is written rather than when a machine behaves oddly.

Scope: `src/neurobrix/core/`, the layer whose whole contract is to be
vendorless. Vendor-specific trees (`config/vendors/`, the per-vendor kernel
reference directories, backend packages) are where vendor names belong.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Device-string prefixes. A literal comparison against one of these in the
# vendorless layer is what this test exists to catch.
VENDOR_PREFIXES = {"cuda", "hip", "rocm", "xpu", "mps", "metal", "npu", "musa"}

# Comparison shapes that branch on a device string.
_COMPARE_METHODS = {"startswith", "endswith"}

# The accelerator prefixes a guard has to name to be vendorless. A prefix
# check listing all of these treats every backend alike; one listing fewer
# takes a different branch depending on whose machine it runs on.
_COMPLETE_SET = {"cuda", "hip", "xpu", "mps"}

ROOT = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "core"

# A guard that is CORRECT as written, with the reason. An unexplained waiver
# is how a rule stops meaning anything, so every entry says why.
ALLOWED: dict[str, str] = {
    "neurobrix/core/io/weight_loader.py:834":
        "gates pinned-memory DMA. The very next branch reads `elif device != "
        "'cpu'` with the comment 'MPS/XPU: direct .to() transfer (no pinned "
        "DMA - unified memory)', so every backend is handled and the CUDA "
        "name only picks the transfer mechanism CUDA needs.",
    "neurobrix/core/io/weight_loader.py:941":
        "same shape as :834, for the pytorch-format loader.",
    "neurobrix/core/io/weight_loader.py:700":
        "batch sync after non-blocking DMA. Only the CUDA/HIP path issues "
        "non-blocking transfers; the MPS path uses a blocking .to(), so there "
        "is nothing outstanding to synchronise.",
    "neurobrix/core/prism/solver.py:3981":
        "unreachable. The branch is the `else` of a chain over _EAGER_VALUES "
        "and _LAZY_VALUES, whose union is every AllocationStrategy value, so "
        "no valid strategy reaches it. Latent, not live.",
}

# A guard that is WRONG on non-CUDA hardware, recorded with what it costs.
# These do not fail the test -- they are known and tracked -- but a NEW one
# will, which is the point: the list may shrink, never silently grow.
KNOWN_OPEN: dict[str, str] = {
    "neurobrix/core/strategies/zero3.py:257":
        "`if not self.exec_device.startswith('cuda'): return` - the zero3 "
        "offload install is a NO-OP on Apple. Prism still selects zero3 (a "
        "31 GB component was assigned it on mps:0), so the strategy is chosen "
        "and then does nothing.",
    "neurobrix/core/runtime/executor.py:1155":
        "`dev.startswith(('cuda','hip','xpu'))` omits mps, so seen_gpu is "
        "never true on Apple and the mixed cpu/gpu detection returns False.",
    "neurobrix/core/prism/solver.py:3270":
        "collects device strings for cuda:/hip:/xpu: only, so n_devices "
        "counts 0 on Apple.",
    "neurobrix/core/io/loader.py:360":
        "guards torch.cuda.synchronize before timing the transfer. Correct as "
        "a guard - the call is CUDA-only - but no MPS equivalent is issued, so "
        "the elapsed time is measured before the transfer has completed.",
    "neurobrix/core/io/loader.py:378":
        "same shape as :360.",
    "neurobrix/core/io/weight_loader.py:422":
        "same shape as loader.py:360.",
    "neurobrix/core/strategies/base.py:253":
        "CUDA-only device transfer helper living under the vendorless layer.",
    "neurobrix/core/strategies/triton/base.py:38":
        "to_cuda/to_cpu only - the Triton strategy family under core/ is "
        "CUDA-only by construction.",
    "neurobrix/core/strategies/triton/base.py:65":
        "CUDA-only synchronize_device.",
    "neurobrix/core/strategies/triton/lazy_sequential.py:67":
        "raises ZERO FALLBACK unless a CUDA device resolves, so lazy_sequential "
        "cannot run on Apple at all on the Triton branch - and Prism assigns "
        "that strategy to models this machine is expected to serve.",
    "neurobrix/core/strategies/triton/lazy_sequential.py:71":
        "second half of the same guard.",
}


def _vendor_literal(node: ast.AST) -> str | None:
    """The vendor prefix this node is a literal for, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        head = node.value.split(":")[0].strip().lower()
        if head in VENDOR_PREFIXES:
            return head
    return None


def _findings_in(path: Path) -> list[tuple[int, str, str]]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:  # not ours to police
        return []
    out: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        # x.startswith("cuda") / x.endswith("cuda")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _COMPARE_METHODS:
                for arg in node.args:
                    if isinstance(arg, ast.Tuple):
                        # A tuple naming every accelerator prefix is exactly
                        # the vendorless form -- it is the SINGLE-vendor guard
                        # that silently takes the wrong branch elsewhere.
                        covered = {v for e in arg.elts if (v := _vendor_literal(e))}
                        if covered and not covered >= _COMPLETE_SET:
                            out.append((node.lineno, f".{node.func.attr}() covering only "
                                        f"{sorted(covered)}", "partial set"))
                        continue
                    v = _vendor_literal(arg)
                    if v:
                        out.append((node.lineno, f".{node.func.attr}()", v))
        # An equality dispatch (`kind == "cuda"` inside a chain that also
        # handles hip/xpu/mps) is legitimate and extremely common, and an AST
        # cannot cheaply tell a complete chain from a partial one. Equality is
        # therefore NOT failed on here -- claiming it would make the rule noise
        # and the waiver list meaningless. The prefix-guard shape above is the
        # one that silently mis-branches, and it is what this enforces.
    return out


def test_vendorless_layer_does_not_branch_on_one_vendor():
    assert ROOT.is_dir(), f"vendorless layer not found at {ROOT}"
    findings = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT.parents[1]).as_posix()
        for lineno, shape, vendor in _findings_in(path):
            key = f"{rel}:{lineno}"
            if key in ALLOWED or key in KNOWN_OPEN:
                continue
            line = path.read_text().splitlines()[lineno - 1].strip()
            findings.append(f"{key}  {shape} on {vendor!r}\n      {line}")

    assert not findings, (
        "the vendorless layer branches on a single vendor's device prefix.\n"
        "Device strings belong here, but they come from "
        "get_device_prefix(vendor, architecture) -- a literal comparison "
        "silently takes the wrong branch on every other vendor's hardware:\n\n"
        + "\n".join(findings)
    )


def test_every_waiver_and_known_item_still_points_at_a_real_line():
    """A stale entry silently re-permits whatever moved into that line.

    Both lists key on file:line, so a shifted line would keep waiving while
    pointing at something else entirely. This fails when an entry no longer
    corresponds to a flagged site, forcing it to be re-adjudicated rather than
    inherited.
    """
    live = set()
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT.parents[1]).as_posix()
        for lineno, _shape, _vendor in _findings_in(path):
            live.add(f"{rel}:{lineno}")

    stale = sorted((set(ALLOWED) | set(KNOWN_OPEN)) - live)
    assert not stale, (
        "these entries no longer match a flagged site — re-adjudicate them "
        "instead of leaving them to waive whatever moved there:\n  "
        + "\n  ".join(stale)
    )
