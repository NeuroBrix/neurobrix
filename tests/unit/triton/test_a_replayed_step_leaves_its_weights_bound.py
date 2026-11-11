"""After a replayed stretch restored its recorded arena, the next step does not bind every weight again.

A served request's first step binds afresh (the cleanup emptied the arena). The replayed stretch then restores the
arena its recording held: other weight OBJECTS than the fresh bind's, so the arena's epoch moves — and the next
step's bind, memoised against the epoch, bound all the weights again, whose fresh objects the next restore replaced
again. From a served model's second request on, every token paid a full bind and the strided copies of the weights
the scan found replaced (Qwen3-Coder-30B-A3B on a V100-32GB, 2026-11-11: 0.20 s a token against 0.08; 30 binds in
30 tokens, read by a probe). No card.

What this test would do if the code were wrong: the second step calls `_bind_weights` again — RED (seen, with the
call after the replay removed)."""
from nbxtensor.memory.arena import Arena

from neurobrix.triton import replay as _replay
from neurobrix.triton.sequence import TritonSequence


def _sequence(monkeypatch, binds):
    seq = TritonSequence.__new__(TritonSequence)
    seq._arena = Arena(4, num_weights=2, num_inputs=1)
    seq._compiled = True
    recorded = [object(), object(), None, None]          # the weights the recording bound

    def bind(weights):
        binds.append(1)
        seq._arena[0], seq._arena[1] = object(), object()     # a bind puts fresh objects (views, host originals)

    def replayed(s, skip_kills, cb):
        s._arena.restore_from(recorded)
        return True
    monkeypatch.setattr(seq, "_bind_weights", bind)
    monkeypatch.setattr(_replay, "maybe_run", replayed)
    monkeypatch.setattr(seq, "_expert_segments", lambda: [("tail", 0, 1, (), [False])])
    seq._ops = [object()]
    return seq


def test_the_second_step_of_a_request_binds_nothing(monkeypatch):
    binds, weights = [], {"w0": object(), "w1": object()}
    seq = _sequence(monkeypatch, binds)
    for _ in range(3):                                   # three decode steps of one request
        seq.bind_weights(weights)
        seq._run_expert_tail(False, None)
    assert binds == [1], f"{len(binds)} binds in three steps: every step bound the weights again"


def test_other_weights_are_still_bound(monkeypatch):
    binds = []
    seq = _sequence(monkeypatch, binds)
    first, other = {"w0": object()}, {"w0": object()}    # both kept alive: the memo reads identities
    seq.bind_weights(first)
    seq._run_expert_tail(False, None)
    seq.bind_weights(other)                              # really other weights
    assert binds == [1, 1]
