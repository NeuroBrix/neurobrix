"""A replayed step restores the arena from its snapshot; the arena's epoch — which makes the next bind re-bind every
weight — moves only when the restored weight slots are other objects than the bound ones (S3, 2026-10-28: an epoch
bumped at every restore re-bound ~20k weights, 24 ms a token, on Qwen3-30B-A3B's decode). No card."""
from nbxtensor.memory.arena import Arena


def test_the_same_weights_keep_the_epoch_and_other_weights_move_it():
    a = Arena(6, num_weights=3, num_inputs=1)
    w = [object(), object(), object()]
    for i, t in enumerate(w):
        a[i] = t
    a[4] = object()
    snap = list(a._slots)
    snap[4], snap[5] = object(), object()          # intermediates differ: they always do
    e = a.epoch
    a.restore_from(snap)
    assert a.epoch == e and a[4] is snap[4]
    other = list(snap)
    other[1] = object()                             # a weight slot replaced
    a.restore_from(other)
    assert a.epoch == e + 1 and a[1] is other[1]


def test_the_views_go_with_the_weights():
    a = Arena(2, num_weights=1)
    a.views["w"] = (object(), object())
    a.clear_all()
    assert a.views == {}
