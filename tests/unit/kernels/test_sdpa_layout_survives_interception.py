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
