"""A key an interceptor forms in triton-sequential is named by its op in the census record.

The triton-sequential loop runs op_uid / op_type interceptors (Prism's tiled convs, the KV cache's
attention) outside the dispatcher, which alone named the op: their keys reached the census with no
`.ops` pair (the 2026-09-28 walks: TinyLlama's attention keys, 2 of 7 unpaired). `_call_named`
names the op while keys are recorded and clears it on return. What the test would do if it did not:
the op seen during the call is None (red with `_call_named` reduced to `fn(*args, **kwargs)`), or it
leaks past the return (the second assertion).
"""
from __future__ import annotations

from neurobrix.core.runtime import graph_executor as GE
from neurobrix.kernels import census


def test_the_interceptor_runs_with_its_op_named_and_leaves_none(monkeypatch):
    monkeypatch.setattr(census, "recording", lambda: True)
    seen = []
    GE._call_named("aten.scaled_dot_product_attention::3",
                   lambda: seen.append(census._OP[0]), (), {})
    assert seen == ["aten.scaled_dot_product_attention::3"]
    assert census._OP[0] is None


def test_without_a_record_the_interceptor_is_called_plainly(monkeypatch):
    monkeypatch.setattr(census, "recording", lambda: False)
    assert GE._call_named("aten.mm::1", lambda a, b=0: a + b, (2,), {"b": 3}) == 5
