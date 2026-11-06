"""A decode step replays when its positions are read on the device and its kept invariant state is named by address.

whisper-large's decoder (2026-11-06, B2 audit): its table-slice interceptor narrowed the positional table at the host's
cache length — a recorded step baked one step's rows and the plan was rejected at verify — and its kept invariant state
(the cross-attention projections, state engine step 4) declined the replay outright. Positions read from the cache's
device length and the kept state's addresses in the bucket: the plan verifies byte-equal and replays, 216 -> 57 ms a token
on a V100, the same transcript. What this test does if the code were wrong: the identity given while nothing is kept, or
kept for an input that changed (a new window) — red; a cache without a device length still signing a position-
interceptor step (a baked position replayed) — red. Card-less."""
from types import SimpleNamespace

from neurobrix.core.state.invariant import InvariantState


def _state():
    plan = {"op_indices": [0], "frontier_slots": [5], "input_slots": [1]}
    return InvariantState(plan, lambda a, b: a == b, lambda v: v, lambda a, b: False, lambda v: id(v))


def test_the_identity_holds_only_while_the_kept_state_serves_the_same_inputs():
    st, enc, kv = _state(), object(), object()
    arena = {1: enc, 5: None}
    assert st.replay_identity(arena) is None                    # nothing kept yet: the run evaluates it
    assert st.begin(arena) is False
    arena[5] = kv
    st.end(arena)
    assert st.replay_identity(arena) == ((id(kv),), (id(enc),))
    arena[1] = object()                                           # a new window's encoder output
    assert st.replay_identity(arena) is None
    st.release()
    assert st.replay_identity({1: enc, 5: kv}) is None


def test_a_position_step_without_a_device_length_is_not_signed():
    from neurobrix.triton.kv_cache import TritonAttentionInterceptor as I
    it = I.__new__(I)
    it.cache = SimpleNamespace(lens=None, batch=1, replay_key=lambda b: ("k", b))
    it._is_prefill = False
    assert it.replay_signature([it.intercept_position_slice]) is None
    assert it.replay_signature([it.intercept]) == ("k", 256)
