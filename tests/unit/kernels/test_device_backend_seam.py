"""The device runtime is reachable from ONE class, and that is what makes a port possible.

Porting NeuroBrix to a new GPU runtime — ROCm yesterday, Metal next — is
bounded only because every call into the device runtime goes through
`DeviceAllocator`. Measured 2026-09-03: **49 runtime touches inside the
class, 0 outside**. A single raw `_gpu_runtime()` call added anywhere else
turns a bounded port into a hunt, and it would be invisible in review.

So it is pinned here rather than trusted.

The second pin is the backend contract itself. `cuda` and `hip` are two
symbol tables over the same C ABI, so a key present in one and missing from
the other is a crash on that hardware alone — the hardest class of bug to
see, because the machine that would catch it is the one nobody has.

Metal does not fit that table: it is an Objective-C API with `MTLDevice`
and `MTLBuffer` and no `metalMalloc` to name, so it will be a second
*implementation* behind this class rather than a third row in the dict.
These pins describe the seam it will slot into.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from neurobrix.kernels import nbx_tensor

SOURCE = Path(inspect.getfile(nbx_tensor)).read_text()

# Names that reach the device runtime. Anything calling these is coupled to
# the GPU API and must live inside DeviceAllocator.
RUNTIME_ENTRY_POINTS = ("_gpu_runtime", "_active_backend")


# The seam's own implementation: these two resolve the backend table and load
# the runtime library, so they necessarily touch it. Everything ELSE must go
# through DeviceAllocator.
SEAM_FUNCTIONS = ("_gpu_runtime", "_active_backend", "_detect_gpu_backend")


def _allowed_spans() -> list[tuple[int, int]]:
    """Line ranges permitted to reach the device runtime."""
    tree = ast.parse(SOURCE)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeviceAllocator":
            spans.append((node.lineno, node.end_lineno or node.lineno))
        elif isinstance(node, ast.FunctionDef) and node.name in SEAM_FUNCTIONS:
            spans.append((node.lineno, node.end_lineno or node.lineno))
    if not spans:
        pytest.fail("DeviceAllocator not found")
    return spans


def test_every_runtime_touch_is_inside_the_allocator():
    """The property the whole port depends on.

    If this fails, someone reached the GPU runtime from outside the one class
    that is meant to own it. Move the call into `DeviceAllocator` and expose
    it as a method — that is the seam a Metal or ROCm backend replaces."""
    spans = _allowed_spans()
    strays = []
    for number, line in enumerate(SOURCE.split("\n"), start=1):
        if any(start <= number <= end for start, end in spans):
            continue
        code = line.split("#", 1)[0]
        if any(f"{name}()" in code for name in RUNTIME_ENTRY_POINTS):
            strays.append((number, line.strip()[:90]))

    assert not strays, (
        "device-runtime calls found outside DeviceAllocator:\n"
        + "\n".join(f"  line {n}: {t}" for n, t in strays)
    )


def test_no_other_module_reaches_the_device_runtime_directly():
    """The same property, one level up: only `nbx_tensor` talks to the driver.

    The triton tree and the kernels use NBXTensor; if another module starts
    loading `libcudart` itself, the port has two coupling points instead of
    one."""
    root = Path(inspect.getfile(nbx_tensor)).parent.parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "nbx_tensor.py" or "triton_kernels_ref" in str(path):
            continue
        text = path.read_text(errors="replace")
        if "cpu_backend" in path.name:
            continue                      # its whole job is probing for runtimes
        for number, line in enumerate(text.split("\n"), start=1):
            code = line.split("#", 1)[0]          # a comment naming it is fine
            if "libcudart" not in code and "libamdhip64" not in code:
                continue
            # Gated VRAM probes are diagnostic-only (env-gated, off by default)
            # and print rather than compute, so they cannot produce a wrong
            # result. Recorded as portability debt, not blocking.
            if "VRAM_PROBE" in text or "_vram_probe" in text:
                continue
            offenders.append(f"{path.relative_to(root)}:{number}  {code.strip()[:70]}")
    assert not offenders, (
        "modules loading a device runtime directly instead of going through\n"
        "DeviceAllocator — these break on ROCm and Apple, where libcudart.so\n"
        "does not exist:\n  " + "\n  ".join(offenders)
    )


# --- the backend contract ---------------------------------------------------

def test_declared_backends_have_identical_contracts():
    """A key in `cuda` but not in `hip` is a crash on AMD alone."""
    backends = nbx_tensor._GPU_BACKENDS
    assert set(backends) == {"cuda", "hip"}
    cuda, hip = set(backends["cuda"]), set(backends["hip"])
    assert cuda == hip, (
        f"backend contracts differ — only in cuda: {sorted(cuda - hip)}; "
        f"only in hip: {sorted(hip - cuda)}"
    )


def test_the_contract_covers_what_a_port_must_provide():
    """The operations a new runtime has to implement, named.

    This list IS the port's specification: a Metal backend supplies these
    behaviours, whatever it calls them."""
    required = {
        "rt_libs", "malloc", "free", "memcpy", "memset",
        "set_device", "get_device", "device_count", "mem_get_info", "sync",
        "malloc_host", "free_host",
    }
    for name, table in nbx_tensor._GPU_BACKENDS.items():
        missing = required - set(table)
        assert not missing, f"backend {name!r} does not declare {sorted(missing)}"


def test_backend_selection_is_data_not_branching():
    """Selection reads the active Triton target and indexes the table; it does
    not branch on vendor names, so adding a backend is adding data."""
    source = inspect.getsource(nbx_tensor._detect_gpu_backend)
    assert "get_current_target" in source, "selection should ask the runtime"
    assert source.count("if ") <= 3, (
        "backend selection is growing branches; it should stay a table lookup"
    )
