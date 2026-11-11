"""A replayed run mirrors onto a state owner's host state what its recorded normal run moved there — nothing more.

The attention interceptor numbers the layers by the calls it counts in a step, and the KV cache's host length moves
when the LAST layer decodes. A layer-streamed model replays the one piece that frees no input its caller hands over:
its first piece, holding the first layer's attention. Its replay advanced the length (no last layer ran) and counted
no call, so every later piece's attention took the number of the layer before it — keys and values written into the
wrong layer, the last layer's never coming (TinyLlama, Triton, a 1 600 MB card, 2026-11-11: "The capital of
Francephстутяш86488шutelyxt"; at 2 000 MB no piece replays and the text is right). No card.

What this test would do if the code were wrong: after one recorded and one replayed run of a first piece, the call
count stays behind (the next layer is numbered 0) and the length moved — RED (seen, with the replay mirroring a fixed
step: one position, no call)."""
from neurobrix.triton import replay as R
from neurobrix.triton.kv_cache import TritonAttentionInterceptor


class _Cache:
    num_kv_heads = 4
    num_layers = 22

    def __init__(self):
        self.length = 24

    def replay_host_mark(self):
        return self.length

    def replay_advance(self):
        self.length += 1


def _recorded(owner, run):
    plan = R.FrozenPlan.__new__(R.FrozenPlan)
    before = owner.replay_host_mark()
    run()
    plan.advances = {0: (before, owner.replay_host_mark())}
    return plan


def test_a_first_piece_counts_its_attention_call_and_moves_no_position():
    att = TritonAttentionInterceptor(_Cache())
    att._call_count = 0

    def first_layer():                       # the piece's one attention call: layer 0, not the last
        att._call_count += 1
    plan = _recorded(att, first_layer)
    att._call_count = 0                      # the next step begins (update_position_offset)
    R._advance_what_the_recording_moved(plan, [att])
    assert (att._call_count, att.cache.length) == (1, 24), "the next piece's first attention must be layer 1"


def test_a_whole_step_moves_one_position_and_all_its_calls():
    att = TritonAttentionInterceptor(_Cache())
    att._call_count = 0

    def whole_step():
        att._call_count += 22
        att.cache.length += 1
    plan = _recorded(att, whole_step)
    att._call_count = 0
    R._advance_what_the_recording_moved(plan, [att])
    assert (att._call_count, att.cache.length) == (22, 26)
