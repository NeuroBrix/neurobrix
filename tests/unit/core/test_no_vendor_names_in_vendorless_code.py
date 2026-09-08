"""The vendorless layer must not branch on one vendor's device prefix.

`core/prism/structure.py` states the rule at the top of the file: *VENDORLESS:
nvidia/amd/intel -> cuda/hip/xpu device strings*. Device strings therefore
appear in `core/` legitimately — but they are supposed to arrive from the
brand, not from a literal comparison against one vendor.

A literal comparison is not a style problem. `startswith("cuda")` in a sum
over devices returns 0 on an Apple or AMD machine, and the branch it guards
then takes the "no GPU" path on hardware that has one — silently, with no
error anywhere. This walks the AST and fails on that shape, so the next one
is caught when it is written rather than when a machine behaves oddly.

TWO NAMING LAYERS, AND WHY THE TEST HAS TO KNOW WHICH
-----------------------------------------------------
The same literal excludes different hardware depending on where the string
it compares came from:

* **prism** — a string built by `DeviceSpec.brand.to_device_prefix()`:
  NVIDIA `cuda:N`, AMD **`hip:N`**, Intel `xpu:N`, Apple `mps:N`. A guard
  naming only `cuda` here excludes **AMD and Apple**.
* **kernel** — a string from the vendor YAML's `device_prefix`, where nvidia
  **and amd** both map to `"cuda"` and only apple to `"mps"` (an NBXTensor
  reports `_device == 'cuda'` on Metal). The same guard here excludes
  **Apple only**.

Without that distinction a reviewer states the impact backwards, which has
already happened twice in this work — first claiming AMD was excluded
everywhere, then claiming it never was. So every entry below DECLARES its
layer, and the test COMPUTES which vendors the site excludes instead of
anyone asserting it in prose.
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

# What each vendor's device string looks like, per layer. This is the whole
# point of declaring a layer: the same literal excludes different hardware.
LAYERS: dict[str, dict[str, str]] = {
    # DeviceSpec.brand.to_device_prefix()
    "prism": {"nvidia": "cuda", "amd": "hip", "intel": "xpu", "apple": "mps"},
    # config/vendors/*/*.yml `device_prefix`
    "kernel": {"nvidia": "cuda", "amd": "cuda", "apple": "mps"},
}

ROOT = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "core"


def excluded_vendors(layer: str, named_prefixes: set[str]) -> list[str]:
    """Which vendors a guard naming `named_prefixes` silently skips."""
    table = LAYERS[layer]
    return sorted(v for v, prefix in table.items() if prefix not in named_prefixes)


# Keys are "<file>::<the line's own text>", never a line number: a waiver keyed
# on a number silently follows whatever moves into that line, and the first
# edit elsewhere in the file invalidates every entry below it.
#
# Each value is (layer, reason). The layer says which naming scheme the
# compared string comes from, so the impact is computed rather than claimed.

# A guard that is CORRECT as written.
ALLOWED: dict[str, tuple[str, str]] = {
    'neurobrix/core/io/weight_loader.py::is_cuda = device.startswith("cuda") or device.startswith("hip")':
        ("prism",
         "gates pinned-memory DMA. The next branch reads `elif device != 'cpu'` "
         "with the comment 'MPS/XPU: direct .to() transfer (no pinned DMA - "
         "unified memory)', so every backend is handled and the vendor names "
         "only pick the transfer mechanism CUDA and HIP need. Covers both "
         "loaders (identical line)."),
    'neurobrix/core/io/weight_loader.py::if any(d.startswith("cuda") or d.startswith("hip") for d in devices_used):':
        ("prism",
         "batch sync after non-blocking DMA. Only the CUDA/HIP path issues "
         "non-blocking transfers; the MPS path uses a blocking .to(), so there "
         "is nothing outstanding to synchronise."),
    'neurobrix/core/prism/solver.py::total_gpu_mb = sum(d.capacity_mb for d in devices if d.device_string.startswith("cuda"))':
        ("prism",
         "unreachable: it is the `else` of a chain over _EAGER_VALUES and "
         "_LAZY_VALUES, whose union is every AllocationStrategy value. Latent."),
}

# A guard that is WRONG on hardware it excludes, recorded with what it costs.
# These do not fail the test — they are known and tracked — but a NEW one
# will: the list may shrink, never silently grow.
KNOWN_OPEN: dict[str, tuple[str, str]] = {
    "neurobrix/core/runtime/executor.py::elif dev.startswith(('cuda', 'hip', 'xpu')):":
        ("prism",
         "omits mps, so seen_gpu is never true on Apple and the mixed cpu/gpu "
         "detection returns False."),
    'neurobrix/core/prism/solver.py::if d.startswith("cuda:") or d.startswith("hip:") or d.startswith("xpu:"):':
        ("prism",
         "collects device strings for cuda:/hip:/xpu: only, so n_devices "
         "counts 0 on Apple."),
    'neurobrix/core/io/loader.py::if device.startswith("cuda"):':
        ("prism",
         "guards torch.cuda.synchronize before timing the transfer. Correct as "
         "a guard, but no MPS equivalent is issued, so elapsed is measured "
         "before the transfer has completed."),
    'neurobrix/core/io/loader.py::if device.startswith("cuda") or device.startswith("hip"):':
        ("prism", "same shape as the line above."),
    'neurobrix/core/io/weight_loader.py::if device.startswith("cuda") or device.startswith("hip"):':
        ("prism", "same shape as loader.py's sync guard."),
    'neurobrix/core/strategies/base.py::if target_device.startswith("cuda:"):':
        ("prism",
         "CUDA-only device transfer helper living under the vendorless layer."),
    'neurobrix/core/strategies/triton/base.py::if isinstance(device, str) and device.startswith("cuda"):':
        ("prism",
         "sets the device index before syncing only when the string starts with "
         "\"cuda\". This is NOT only a non-NVIDIA bug: on any multi-GPU box a "
         "component on index 1 whose prefix is not matched syncs whatever "
         "device was current instead of its own, so a multi-card NVIDIA run "
         "hits it the moment the string carries another prefix."),
}


def _vendor_literal(node: ast.AST) -> str | None:
    """The vendor prefix this node is a literal for, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        head = node.value.split(":")[0].strip().lower()
        if head in VENDOR_PREFIXES:
            return head
    return None


def _findings_in(path: Path) -> list[tuple[int, str, set[str]]]:
    """(line, shape, the prefixes this site names) for each flagged guard."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:  # not ours to police
        return []
    out: list[tuple[int, str, set[str]]] = []
    per_line: dict[int, set[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _COMPARE_METHODS:
                for arg in node.args:
                    elts = arg.elts if isinstance(arg, ast.Tuple) else [arg]
                    named = {v for e in elts if (v := _vendor_literal(e))}
                    if named:
                        per_line.setdefault(node.lineno, set()).update(named)
    # An `or` chain writes several calls on one line; they are ONE guard, so
    # their prefixes are unioned before the impact is computed. Scoring them
    # separately is what made an early version report a complete
    # cuda-or-hip-or-xpu check as three single-vendor violations.
    for lineno, named in per_line.items():
        out.append((lineno, ".startswith()", named))
    return out


def _live_sites() -> dict[str, tuple[Path, int, set[str]]]:
    sites: dict[str, tuple[Path, int, set[str]]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT.parents[1]).as_posix()
        lines = path.read_text().splitlines()
        for lineno, _shape, named in _findings_in(path):
            sites[f"{rel}::{lines[lineno - 1].strip()}"] = (path, lineno, named)
    return sites


def test_vendorless_layer_does_not_branch_on_one_vendor():
    assert ROOT.is_dir(), f"vendorless layer not found at {ROOT}"
    findings = []
    for key, (_path, lineno, named) in sorted(_live_sites().items()):
        if key in ALLOWED or key in KNOWN_OPEN:
            continue
        # A guard naming every accelerator excludes nobody under either
        # layer. That IS the vendorless form, so it is not a violation and
        # does not need a waiver — requiring one would make the lists grow
        # with entries that say "this is fine".
        if not any(excluded_vendors(L, named) for L in LAYERS):
            continue
        rel, _, line = key.partition("::")
        # An unadjudicated site: report what it would exclude under BOTH
        # layers, since nobody has yet declared which one it reads.
        both = {L: excluded_vendors(L, named) for L in LAYERS}
        findings.append(
            f"{rel}:{lineno}  names {sorted(named)}\n"
            f"      {line}\n"
            f"      excludes {both['prism']} if it reads a Prism device string, "
            f"{both['kernel']} if a kernel one — declare which."
        )

    assert not findings, (
        "the vendorless layer branches on a single vendor's device prefix.\n"
        "Device strings belong here, but a literal comparison silently takes "
        "the wrong branch on the hardware it does not name:\n\n"
        + "\n".join(findings)
    )


def test_every_entry_declares_a_layer_the_test_knows():
    """A reason without a layer cannot say which hardware it costs."""
    bad = []
    for name, table in (("ALLOWED", ALLOWED), ("KNOWN_OPEN", KNOWN_OPEN)):
        for key, value in table.items():
            if (not isinstance(value, tuple) or len(value) != 2
                    or value[0] not in LAYERS):
                bad.append(f"{name}[{key[:60]}...] -> {value!r}")
    assert not bad, (
        "every entry must be (layer, reason) with layer in "
        f"{sorted(LAYERS)}:\n  " + "\n  ".join(bad)
    )


def test_every_known_open_site_still_excludes_the_hardware_it_claims():
    """The impact is computed from the site's own prefixes, not asserted.

    If someone widens a guard to name every accelerator, it stops excluding
    anyone — and then it belongs in ALLOWED or nowhere, not in a list of open
    defects. This fails rather than letting the list rot into fiction.
    """
    live = _live_sites()
    fixed = []
    for key, (layer, _reason) in KNOWN_OPEN.items():
        if key not in live:
            continue  # the staleness test owns that case
        _path, _lineno, named = live[key]
        if not excluded_vendors(layer, named):
            fixed.append(f"{key}  names {sorted(named)} — excludes nobody now")
    assert not fixed, (
        "these are listed as open but no longer exclude any vendor:\n  "
        + "\n  ".join(fixed)
    )


def test_every_waiver_and_known_item_still_points_at_a_real_line():
    """A stale entry silently re-permits whatever moved into that line."""
    live = set(_live_sites())
    stale = sorted((set(ALLOWED) | set(KNOWN_OPEN)) - live)
    assert not stale, (
        "these entries no longer match a flagged site — re-adjudicate them "
        "instead of leaving them to waive whatever moved there:\n  "
        + "\n  ".join(stale)
    )


def test_the_open_list_reports_what_it_excludes():
    """The list's value is the impact it names; print it with the report."""
    live = _live_sites()
    rows = []
    for key, (layer, _r) in sorted(KNOWN_OPEN.items()):
        if key not in live:
            continue
        _p, _l, named = live[key]
        rows.append(f"{key.split('::')[0]}  [{layer}]  names {sorted(named)}"
                    f"  -> excludes {excluded_vendors(layer, named)}")
    assert rows, "the open list is empty — nothing left to track"
    print("\n" + "\n".join(rows))
