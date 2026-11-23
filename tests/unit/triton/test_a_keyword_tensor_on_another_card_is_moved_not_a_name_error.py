"""`TritonSequence._transfer_kwargs` moves a keyword NBXTensor that lives on another card, as `_transfer_args` moves a
positional one. Red on main 320ecaa7f: it called `needs_move` / `transfer_tensor`, which only `_transfer_args` imported
(locally), so the first keyword tensor needing a cross-device move raised NameError."""
from nbxtensor.tensor import NBXTensor

from neurobrix.triton import sequence
from neurobrix.triton.sequence import TritonSequence


def test_a_keyword_tensor_needing_a_move_is_moved(monkeypatch):
    t = object.__new__(NBXTensor)                    # an NBXTensor as the walker sees it, no device needed
    moved = object.__new__(NBXTensor)
    monkeypatch.setattr(sequence, "needs_move", lambda tensor, dev: tensor is t)      # the names both walkers call
    monkeypatch.setattr(sequence, "transfer_tensor", lambda tensor, dev: moved)
    out = TritonSequence._transfer_kwargs(None, {"x": t, "n": 3}, 1)
    assert out == {"x": moved, "n": 3}
