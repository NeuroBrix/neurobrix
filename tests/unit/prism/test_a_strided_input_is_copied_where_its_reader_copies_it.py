"""An op is priced with the copies its executor makes of what it reads.

Two copies the walk did not hold, each at a decoder's measured peak (campaigns/2026_10_09_transient_proof):

* TRITON, the wrappers' contiguity guard (`op_transients.contiguous_copy_bytes`): a reader of
  `CONTIGUOUS_COPY_READERS` copies a strided input dense before its kernel. Sana_1600M_4Kpx_BF16's
  VAE `aten.relu::18` reads the [0,3,1,2] permute of a [1,3072,4096,128] fp32 tensor and held
  6 144 MB above its output (card 2, 2026-10-09, allocator pool off, walk_B_sana4k_vae_nopool.jsonl).
  The set is held here to the dispatched functions' own source: each member's Triton function
  copies its first input contiguous in an unconditional statement.
* COMPILED, the vendor library's layout (`op_transients.library_layout_transient_bytes`): cuDNN
  runs a channels-first convolution on Tensor Cores through channels-last copies of the input, the
  weight and the result (https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional,
  "NCHW ... incur[s] automatic transposes"), at the dtypes the vendor file declares
  (`conv.library_layout_copy_dtypes`). mochi-1-preview's VAE `aten.convolution::33` held
  636 + 540 + 0.8 MB of them (card 3, 2026-10-09).

SEEN RED (2026-10-09): `contiguous_copy_bytes` returning 0, `strided_view_tensors` returning the
empty set, `library_layout_transient_bytes` returning 0, and "add" added to the reader set — each
turns its cell red.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
     tests/unit/prism/test_a_strided_input_is_copied_where_its_reader_copies_it.py
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

import pytest

from neurobrix.core.prism import op_transients as T


def _op(uid, op_type, ins, outs, **attrs):
    return {"op_uid": uid, "op_type": op_type, "input_tensor_ids": list(ins),
            "output_tensor_ids": list(outs), "attributes": attrs}


def test_a_view_that_moves_an_axis_is_strided_and_one_that_does_not_is_not():
    ops = {o["op_uid"]: o for o in [
        _op("p::0", "aten::permute", ["x"], ["moved"], dims=[0, 3, 1, 2]),
        _op("p::1", "aten::permute", ["x"], ["kept"], dims=[0, 1, 2, 3]),
        _op("tr::0", "aten::transpose", ["x"], ["swapped"], dim0=1, dim1=2),
        _op("tr::1", "aten::transpose", ["x"], ["same"], dim0=1, dim1=1),
        _op("t::0", "aten::t", ["w"], ["wt"]),
        _op("v::0", "aten::view", ["x"], ["flat"]),
    ]}
    assert T.strided_view_tensors(ops) == {"moved", "swapped", "wt"}


def test_a_reader_that_copies_holds_the_copy_of_a_strided_input_on_triton_only():
    op = _op("relu::18", "aten::relu", ["moved"], ["y"])
    shape = [1, 128, 3072, 4096]
    triton = T.TransientContext(engine="triton", strided_views=frozenset({"moved"}))
    assert T.contiguous_copy_bytes(op, [shape], ["float32"], triton) == 6_144 * 2 ** 20
    # a dense input is read in place
    assert T.contiguous_copy_bytes(op, [shape], ["float32"],
                                   T.TransientContext(engine="triton", strided_views=frozenset({"other"}))) == 0
    # the compiled library reads by strides
    assert T.contiguous_copy_bytes(op, [shape], ["float32"],
                                   T.TransientContext(engine="compiled", strided_views=frozenset({"moved"}))) == 0
    # a binary elementwise wrapper reads its strided operand by its strides
    add = _op("add::0", "aten::add", ["moved", "z"], ["y"])
    assert T.contiguous_copy_bytes(add, [shape, shape], ["float32", "float32"], triton) == 0
    # and the walk adds it to the op's transient
    assert T.op_transient_bytes("relu::18", op, [shape], [shape], ["float32"], "float32", triton) \
        >= 6_144 * 2 ** 20


def _callable_node(f):
    try:
        src = textwrap.dedent(inspect.getsource(f))
    except (OSError, TypeError):
        return None, ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, src          # a lambda inside a table: read by its callee below
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.Lambda)):
            return n, src
    return None, src


def _copies_first_input(node) -> bool:
    """An UNCONDITIONAL statement of `node`'s body copies its first parameter contiguous."""
    p = node.args.args[0].arg if node.args.args else None
    if p is None:
        return False
    for st in node.body if isinstance(node.body, list) else [node.body]:
        if isinstance(st, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            continue
        for sub in ast.walk(st):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "contiguous"
                    and any(isinstance(x, ast.Name) and x.id == p for x in ast.walk(sub.func.value))):
                return True
            if isinstance(sub, ast.ListComp) and any(isinstance(g.iter, ast.Name) and g.iter.id == p
                                                     for g in sub.generators):
                e = sub.elt
                if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "contiguous":
                    return True
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "_prepare_unary"
                    and sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == p):
                return True
    return False


def _copies(f, hop=0) -> bool:
    from neurobrix.kernels import wrappers as w
    from neurobrix.kernels.nbx_tensor import NBXTensor
    node, src = _callable_node(f)
    if node is not None and _copies_first_input(node):
        return True
    if hop == 0 and len(src.strip().splitlines()) <= 3:       # a one-line forwarder
        for name in re.findall(r"(?:\bw\.|NBXTensor\.)(\w+)\(", src):
            g = getattr(w, name, None) or getattr(NBXTensor, name, None)
            if g is not None and _copies(g, 1):
                return True
    return False


def test_every_reader_in_the_set_copies_its_input_in_its_triton_function():
    from neurobrix.kernels import dispatch
    from neurobrix.kernels.classification import canonical_aten
    table = {}
    for name, f in dispatch._build_op_map().items():
        table.setdefault(canonical_aten(name), f)
    missing = sorted(r for r in T.CONTIGUOUS_COPY_READERS if r not in table)
    assert not missing, f"readers with no Triton function: {missing}"
    not_copying = sorted(r for r in T.CONTIGUOUS_COPY_READERS if not _copies(table[r]))
    assert not not_copying, f"in CONTIGUOUS_COPY_READERS but their function copies nothing: {not_copying}"


def test_the_compiled_library_holds_its_layout_copies_at_the_declared_dtypes():
    op = _op("convolution::33", "aten::convolution", ["x", "w", "b"], ["y"])
    x, w, b, y = [1, 128, 14, 322, 578], [128, 128, 3, 3, 3], [128], [1, 128, 12, 320, 576]
    ctx = T.TransientContext(engine="compiled", conv_layout_copy_dtypes=frozenset({"float16"}))
    got = T.library_layout_transient_bytes("convolution::33", op, [x, w, b], [y], ["float16"] * 3, ctx)
    assert got == 2 * (T._num(x) + T._num(w) + T._num(y)), got
    # the Triton engines run the graph's layout
    tri = T.TransientContext(engine="triton", conv_layout_copy_dtypes=frozenset({"float16"}))
    assert T.library_layout_transient_bytes("convolution::33", op, [x, w, b], [y], ["float16"] * 3, tri) == 0
    # a dtype the vendor file does not declare is run in the graph's layout
    assert T.library_layout_transient_bytes("convolution::33", op, [x, w, b], [y], ["float32"] * 3, ctx) == 0
    assert T.op_transient_bytes("convolution::33", op, [x, w, b], [y], ["float16"] * 3, "float16", ctx) >= got


@pytest.mark.parametrize("arch, dtypes", [("volta", {"float16"}),
                                          ("ampere", {"float16", "bfloat16", "float32"}),
                                          ("hopper", {"float16", "bfloat16", "float32"})])
def test_the_dtypes_are_the_vendor_file_s(arch, dtypes):
    from neurobrix.core.config.loader import get_vendor_config
    assert set(get_vendor_config("nvidia", arch)["conv"]["library_layout_copy_dtypes"]) == dtypes
