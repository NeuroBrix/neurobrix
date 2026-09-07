"""The recorded K/V layout must reach whatever function executes the op.

Recording the layout on the op is only half of it: an interceptor installed
AFTER compile() replaces `op.func`, and the wrapper carrying the layout went
with it. Measured on CUDA 2026-09-07 with the head_dim cell: the KV
interceptor refused every attention of a 64-token TinyLlama request on two
of the four arms, because the layout never reached it — the compiled arm,
whose interceptor was bound before compile, ran.

So there are two invariants here, and the second is the one that keeps this
closed: the helper attaches the layout, and EVERY assignment to `op.func` in
both engines goes through the helper. A future hot-patch site that forgets
fails this file rather than a model at one length.
"""
import ast
import inspect
from pathlib import Path

import pytest

from neurobrix.triton.sequence import TritonSequence
from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence


class _Stub:
    """The smallest thing the helper needs: a layout map."""
    def __init__(self, cls, attr, value):
        setattr(self, attr, {"sdpa::0": value})
        self._bind = cls._bind_sdpa_layout.__get__(self, cls)


def _probe():
    seen = {}

    def probe(*args, **kwargs):
        seen.update(kwargs)
        return "ran"
    return probe, seen


def test_the_triton_helper_attaches_the_recorded_layout():
    stub = _Stub(TritonSequence, "_sdpa_k_layout", True)
    probe, seen = _probe()
    assert stub._bind("sdpa::0", probe)() == "ran"
    assert seen["k_pre_transposed"] is True


def test_the_triton_helper_leaves_other_ops_alone():
    stub = _Stub(TritonSequence, "_sdpa_k_layout", True)
    probe, seen = _probe()
    assert stub._bind("mm::7", probe) is probe
    probe()
    assert "k_pre_transposed" not in seen


def test_the_compiled_helper_attaches_both_layouts():
    stub = _Stub(CompiledSequence, "_sdpa_kv_layout", (True, False))
    probe, seen = _probe()
    assert stub._bind("sdpa::0", probe)() == "ran"
    assert seen["k_pre_transposed"] is True
    assert seen["v_pre_transposed"] is False


def test_a_caller_that_already_said_so_is_not_overridden():
    """`setdefault`, not assignment: an explicit answer at the call site
    wins over the bound one."""
    stub = _Stub(TritonSequence, "_sdpa_k_layout", True)
    probe, seen = _probe()
    stub._bind("sdpa::0", probe)(k_pre_transposed=False)
    assert seen["k_pre_transposed"] is False


def test_the_layout_is_only_offered_to_a_callable_that_can_receive_it():
    """A per-op_uid interceptor is usually a tiling hook with a fixed
    signature. Offering it a keyword it does not declare would turn a
    correct layout into a TypeError, so the parameter is offered only when
    the callable declares it or takes **kwargs."""
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor.__new__(GraphExecutor)
    attrs = {"nbx_k_pre_transposed": True, "nbx_v_pre_transposed": False}
    sdpa = "aten::scaled_dot_product_attention"

    def takes_it(q, k, v, k_pre_transposed=None):
        return None

    def takes_kwargs(q, k, v, **kw):
        return None

    def takes_neither(q, k, v):
        return None

    assert ex._sdpa_layout_kwargs(sdpa, attrs, takes_it) == {"k_pre_transposed": True}
    assert ex._sdpa_layout_kwargs(sdpa, attrs, takes_kwargs) == {
        "k_pre_transposed": True, "v_pre_transposed": False}
    assert ex._sdpa_layout_kwargs(sdpa, attrs, takes_neither) == {}
    assert ex._sdpa_layout_kwargs("aten::mm", attrs, takes_kwargs) == {}
    assert ex._sdpa_layout_kwargs(sdpa, {}, takes_kwargs) == {}


def test_every_interceptor_call_site_offers_the_layout():
    """The three places an interceptor is CALLED without a compile step to
    bind onto — the ATen sequential engine's two branches and the
    triton-sequential loop's — must each pass through the helper."""
    import inspect as _inspect
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    src = Path(_inspect.getsourcefile(GraphExecutor)).read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        holder = node.func.value
        if not (isinstance(holder, ast.Attribute)
                and holder.attr in ("_op_interceptors", "_op_uid_interceptors")):
            continue
        offenders.append(node.lineno)
    assert not offenders, (
        "an interceptor is called directly at these lines instead of through "
        "a local bound with _sdpa_layout_kwargs, so an attention op reaching "
        "it would lose its recorded layout: " + ", ".join(map(str, offenders)))


@pytest.mark.parametrize("module", [TritonSequence, CompiledSequence])
def test_every_assignment_to_op_func_goes_through_the_helper(module):
    """The invariant that keeps this closed as the engines grow."""
    path = Path(inspect.getsourcefile(module))
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Attribute) and target.attr == "func"):
                continue
            value = node.value
            through_helper = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "_bind_sdpa_layout")
            if not through_helper:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "these assignments to op.func bypass _bind_sdpa_layout, so an "
        "attention op patched there would lose its recorded K/V layout and "
        "fall back to reading it off a shape that cannot express it at "
        "seq_len == head_dim: " + ", ".join(offenders))


def _slots_of(cls_node: ast.ClassDef) -> set | None:
    """The names a class declares in __slots__, or None when it declares none."""
    for node in cls_node.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__slots__"
                   for t in node.targets):
            continue
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return {e.value for e in value.elts if isinstance(e, ast.Constant)}
        return set()
    return None


@pytest.mark.parametrize("module", [TritonSequence, CompiledSequence])
def test_no_attribute_is_assigned_that_the_class_cannot_hold(module):
    """A `__slots__` class refuses an attribute that is not one of its slots,
    and refuses it where the assignment happens — which for an engine is
    inside compile(), on every model of the hub at once.

    Measured 2026-09-07 on CUDA: a layout map created lazily on
    CompiledSequence raised AttributeError in `compile()` for every model
    whose sequence holds an attention op, and the compiled arm of the whole
    campaign stopped. The class had said so at line 7 of its own file.
    """
    path = Path(inspect.getsourcefile(module))
    tree = ast.parse(path.read_text())
    offenders = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        slots = _slots_of(cls)
        if slots is None:                     # no __slots__: anything goes
            continue
        for node in ast.walk(cls):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr not in slots):
                    offenders.append(f"{cls.name}.{target.attr} "
                                     f"({path.name}:{node.lineno})")
    assert not offenders, (
        "these attributes are assigned on a class that declares __slots__ "
        "and are not among them, so the assignment raises where it runs: "
        + ", ".join(sorted(set(offenders))))


def test_a_real_compiled_sequence_holds_the_layout_map():
    """The AST check says the slot is declared; this says the object can be
    built and the map assigned on it. The crash it stands for happened at
    `compile()`, not at import."""
    import torch
    empty = {"ops": {}, "tensors": {}, "execution_order": [],
             "weight_tensor_ids": [], "input_tensor_ids": [],
             "output_tensor_ids": []}
    seq = CompiledSequence(empty, torch.device("cpu"), torch.float32)
    assert seq._sdpa_kv_layout == {}
    seq._sdpa_kv_layout["sdpa::0"] = (True, False)
    probe, seen = _probe()
    seq._bind_sdpa_layout("sdpa::0", probe)()
    assert seen == {"k_pre_transposed": True, "v_pre_transposed": False}
