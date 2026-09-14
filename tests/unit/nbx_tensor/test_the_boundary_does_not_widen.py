"""`nbx_tensor` is the replacement of torch on the Triton branch and a future
separate library (owner, 2026-09-14): no engine import leaks into it from
now on. This is the ratchet — the imports it holds TODAY are recorded here
by scope, and any import of `neurobrix.*` or `torch` not in the record fails
with the rule. The record can only shrink; shrinking it is the work the
page `docs/reference/nbx-tensor-boundary.md` names.

Injection (2026-09-14 00:21 UTC): an engine import added inside a function of
nbx_tensor.py made this RED; removed, green.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
NBX_TENSOR = REPO / "src/neurobrix/kernels/nbx_tensor.py"

# Every engine-or-torch import nbx_tensor.py carried on 2026-09-14, by the scope
# that carries it. Each is a debt of the separation; none may be joined by another.
RECORDED = {
    ("nbx_dtype_to_torch", "torch"),
    ("set_torch_device", "torch"),
    ("nbx_to_torch", "torch"),
    ("DeviceAllocator._pool_parked_cap", "neurobrix.kernels.wrappers"),
    ("_copy_nd", "neurobrix.kernels.ops.strided_copy"),
    ("_strided_copy", "neurobrix.kernels.ops.strided_copy"),
    ("_strided_scatter", "neurobrix.kernels.ops.strided_copy"),
    ("_fill_constant", "neurobrix.kernels.ops.fill_op"),
    ("NBXTensor.to", "neurobrix.kernels.ops.copy_op"),
    ("NBXTensor.cat", "neurobrix.kernels.ops.cat_op"),
    ("NBXTensor.__add__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__radd__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__sub__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__rsub__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__mul__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__rmul__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__truediv__", "neurobrix.kernels.wrappers"),
    ("NBXTensor.__neg__", "neurobrix.kernels.wrappers"),
}
ALLOWED_ROOTS = {"triton", "numpy"}          # the library's own dependencies


def boundary_imports(path: Path = NBX_TENSOR):
    """{(scope, module)} of every import that is not stdlib, not a sibling
    (relative), and not one of the library's own dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    std = set(sys.stdlib_module_names)
    found = set()

    def walk(node, scope):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(ch, scope + [ch.name])
                continue
            mods = []
            if isinstance(ch, ast.Import):
                mods = [a.name for a in ch.names]
            elif isinstance(ch, ast.ImportFrom) and ch.level == 0:
                mods = [ch.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if root in std or root in ALLOWED_ROOTS:
                    continue
                found.add((".".join(scope) or "<module>", m))
            walk(ch, scope)
    walk(tree, [])
    return found


def test_no_engine_or_torch_import_joined_the_record():
    new = boundary_imports() - RECORDED
    assert not new, (
        "nbx_tensor.py gained an import of the engine or of torch: " + ", ".join(f"{s}: {m}" for s, m in sorted(new))
        + " — nbx_tensor is a library the engine imports, never the reverse (owner, 2026-09-14). "
          "Give the missing capability to nbx_tensor itself; do not fetch it from the engine or from torch.")


def test_the_record_only_shrinks():
    """A recorded import that is gone is progress; the record must then lose the
    line, so the ratchet stays tight (a record wider than the file lets an
    import come back unnoticed)."""
    gone = RECORDED - boundary_imports()
    assert not gone, ("these recorded imports no longer exist — remove them from RECORDED so the ratchet "
                      "tightens: " + ", ".join(f"{s}: {m}" for s, m in sorted(gone)))


def test_the_ratchet_reads_the_file_it_claims_to(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("import os\nimport numpy as np\ndef f():\n    from neurobrix.core.runtime import x\n    import torch\n")
    assert boundary_imports(p) == {("f", "neurobrix.core.runtime"), ("f", "torch")}
