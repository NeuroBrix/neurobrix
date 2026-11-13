"""A Triton sequence compiled again (a served model's weights reloaded between requests) is bound EVERY per-op_uid
interceptor its executor holds, as the compiled sequence's rebuild is — not only the ones still pending. It kept only
the pending ones: the decoder's KV interceptor, registered while the first sequence existed, was lost, and
Ming-Lite-Omni served by Triton prefilled request 2 into no cache and decoded garbage ("AADADA", 2026-11-13)."""
from neurobrix.core.runtime.graph_executor import GraphExecutor


class _Seq:
    def __init__(self):
        self.bound = {}

    def register_op_uid_interceptor(self, uid, fn):
        self.bound[uid] = fn

    def update_op_uid_interceptors(self, fns):
        self.bound.update(fns)


def _executor():
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._op_uid_followers, ex._compiled_seq = {}, None
    ex._triton_seq = _Seq()
    return ex


def test_an_interceptor_registered_on_the_first_sequence_is_bound_on_the_next():
    ex = _executor()
    kv = object()
    ex.register_op_uid_interceptors({"aten.scaled_dot_product_attention::0": kv})   # bound on the live sequence
    assert ex._triton_seq.bound == {"aten.scaled_dot_product_attention::0": kv}
    ex._triton_seq = _Seq()                                                            # dropped, compiled again
    ex._bind_triton_uid_interceptors()
    assert ex._triton_seq.bound == {"aten.scaled_dot_product_attention::0": kv}


def test_the_pending_ones_are_bound_too():
    ex = _executor()
    ex._triton_seq = None
    tile = object()
    ex.register_op_uid_interceptors({"aten.convolution::1": tile})                   # no sequence yet: pending
    ex._triton_seq = _Seq()
    ex._bind_triton_uid_interceptors()
    assert ex._triton_seq.bound == {"aten.convolution::1": tile}
