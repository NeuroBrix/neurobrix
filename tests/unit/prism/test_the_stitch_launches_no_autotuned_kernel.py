"""The stitch that joins a sliced stretch launches no autotuned kernel: it holds no key of the census.

A stretch run in slices of a token axis (`core/strategies/chunked_piece.ChunkedPiece.run`) executes
ops graph.json does not carry: the slice taken of each input (`take_slice`: reshape, narrow,
contiguous), the slice written into the whole output (`put_slice`: view, narrow, copy_), the copy an
arena cannot overwrite (`own`: new_empty, copy_) and the sum of a contraction's partials
(`accumulate` / `store`: new_empty, copy_, to, +). The census derives keys from graph.json
(tools/derived_census.py), so it can only be complete if none of these launches a kernel the
certified directory keys. This proves it BY NAME, on both engines:

  * compiled: every ATen kind the stitch dispatches is outside the census's keyed kinds
    (`derived_census._KEYED_KINDS`, the one list `_op_launches` keys);
  * Triton: every kernel the NBXTensor methods the stitch calls can reach — walked through the
    source, call by call, inside `neurobrix.kernels` — is a plain `@triton.jit` function, never a
    `triton.runtime.autotuner.Autotuner` (the object the certified directory seeds).

The stitch's own method calls are read from its source too, so a stitch that starts calling
another method is walked without editing this file.

Injection (seen red, then restored green — STATE.md of the op_tiler campaign, 2026-10-05):
`add_forward_kernel` wrapped in `triton.autotune` -> the Triton test names it.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap
from pathlib import Path

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.prism import chunked_region as CR
from neurobrix.kernels.nbx_tensor import NBXTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derived_census as D  # noqa: E402

STITCH = (CR.take_slice, CR.put_slice, CR.own, CR.accumulate, CR.store)

# The ATen kind each tensor method of the stitch dispatches in the compiled engine.
ATEN_OF = {"reshape": "aten::reshape", "narrow": "aten::narrow", "contiguous": "aten::clone",
           "view": "aten::view", "copy_": "aten::copy_", "new_empty": "aten::new_empty",
           "to": "aten::_to_copy", "__add__": "aten::add"}


def _tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _stitch_methods():
    """The tensor methods the stitch calls (`x.m(...)`, `a + b`), read from its source."""
    out = set()
    for fn in STITCH:
        for node in ast.walk(_tree(fn)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                out.add("__add__")
    return out


def _local_imports(tree):
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                names[a.asname or a.name] = (node.module, a.name)
    return names


def _resolve(name, fn, local):
    if name in local:
        mod, attr = local[name]
        try:
            return getattr(importlib.import_module(mod), attr, None)
        except Exception:
            return None
    return (getattr(fn, "__globals__", {}) or {}).get(name)


def _in_kernels(obj):
    return (getattr(obj, "__module__", "") or "").startswith("neurobrix.kernels")


def _launches(entry_methods):
    """{kernel name: kernel object} reachable from the NBXTensor methods `entry_methods`."""
    seen, kernels = set(), {}
    stack = [getattr(NBXTensor, m) for m in entry_methods if hasattr(NBXTensor, m)]
    while stack:
        fn = stack.pop()
        fn = getattr(fn, "__func__", fn)
        if not callable(fn) or id(fn) in seen or not _in_kernels(fn):
            continue
        seen.add(id(fn))
        try:
            tree = _tree(fn)
        except (OSError, TypeError):
            continue
        local = _local_imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name):   # K[grid](...)
                k = _resolve(f.value.id, fn, local)
                if k is not None and type(k).__module__.startswith("triton"):
                    kernels[f.value.id] = k
            elif isinstance(f, ast.Name):
                target = _resolve(f.id, fn, local)
                if target is not None:
                    stack.append(target)
            elif isinstance(f, ast.Attribute) and hasattr(NBXTensor, f.attr):
                stack.append(getattr(NBXTensor, f.attr))
    return kernels


def test_the_stitch_calls_only_the_methods_this_proof_maps():
    assert _stitch_methods() <= set(ATEN_OF), sorted(_stitch_methods() - set(ATEN_OF))


def test_the_compiled_stitch_dispatches_no_keyed_kind():
    kinds = {ATEN_OF[m] for m in _stitch_methods()}
    assert kinds and not kinds & D._KEYED_KINDS, sorted(kinds & D._KEYED_KINDS)


def test_the_triton_stitch_launches_no_autotuned_kernel():
    from triton.runtime.autotuner import Autotuner
    kernels = _launches(_stitch_methods())
    # not vacuous: the copy and the sum are reached
    assert {"strided_copy_nd_kernel", "add_forward_kernel"} <= set(kernels), sorted(kernels)
    tuned = sorted(n for n, k in kernels.items() if isinstance(k, Autotuner))
    assert not tuned, f"the stitch reaches autotuned kernels the census cannot key: {tuned}"
