"""Zero3 drives triton-sequential through a view of its weight store (SequentialWeightView).

Before it, the ratchet never built in that engine (no arena sequence), so every host weight
crossed to the card at every use (Janus-Pro-7B on a 16 GB card, stage B).
"""
from neurobrix.core.strategies.zero3 import Zero3Strategy
from neurobrix.triton.sequential import SequentialWeightView


class _T:
    def __init__(self, device="cpu", base=None):
        self._device = device
        self._base = base

    def contiguous(self):
        return _T(self._device)


def _ref(tid):
    return {"type": "tensor", "tensor_id": tid}


def _graph():
    order = ["emb", "q0", "act0", "q1", "head"]
    ops = {
        "emb": {"attributes": {"args": [_ref("param::embed.weight"), _ref("input::ids")]}},
        "q0": {"attributes": {"args": [_ref("t0"), _ref("param::model.layers.0.q.weight")]}},
        "act0": {"attributes": {"args": [_ref("t1")]}},
        "q1": {"attributes": {"args": [_ref("t2"), _ref("param::model.layers.1.q.weight")]}},
        "head": {"attributes": {"args": [_ref("t3"), _ref("param::lm_head.weight")]}},
    }
    return order, ops


def test_blocks_are_indexed_by_execution_position():
    view = SequentialWeightView(*_graph())
    blocks = view.get_op_blocks()
    assert blocks[0] == {"first_op": 1, "last_op": 2,
                         "weight_tensor_ids": ["param::model.layers.0.q.weight"]}
    assert blocks[1]["first_op"] == 3 and blocks[-1]["last_op"] == 4


def test_a_rebind_lands_in_the_pass_store():
    view = SequentialWeightView(*_graph())
    store = {"param::model.layers.0.q.weight": _T("cpu")}
    view.bind_store(store)
    gpu = _T("cuda")
    assert view.rebind_partial({"param::model.layers.0.q.weight": gpu, "t9": gpu}) == [
        "param::model.layers.0.q.weight"]
    assert store["param::model.layers.0.q.weight"] is gpu and "t9" not in store
    assert view.mark_cpu_weighted_ops_for_transfer(0) == 0


def test_a_view_of_an_evicted_weight_is_copied_out():
    view = SequentialWeightView(*_graph())
    w = _T("cuda")
    alias = _T("cuda", base=w)
    store = {"param::model.layers.0.q.weight": w, "t1": alias, "t2": _T("cuda")}
    view.bind_store(store)
    assert view.materialize_slots_depending_on(["param::model.layers.0.q.weight"]) == 1
    assert store["t1"] is not alias and store["t1"]._base is None


def test_zero3_finds_the_view_when_no_sequence_exists():
    class _Exec:
        _triton_seq = None
        _compiled_seq = None
    ex = _Exec()
    view = SequentialWeightView(*_graph())
    ex._triton_seq_view = ((0, 5), view)
    seq, is_triton = Zero3Strategy._seq_handle(None, ex)
    assert seq is view and is_triton is True
