"""A layer-streamed piece frees the inputs its caller hands over at their last use (`_consumed_input_tids`): a frozen plan
can neither keep the buffer its launches read nor refill it, so such a run is not replayed.

granite-speech-3.3-8b triton on the M4 Pro (the Mac's Metal gate, 2026-11-06): replayed, the verify's normal re-run found
the freed input None ("Failed at aten.bmm::0: None args at positions [1]", red since 2026-10-03); a plan that kept it
transcribed garbage after four tokens (the freed memory reused); with the replay off, the native transcription (880038ec).

What this test does if the code were wrong: a piece handing over inputs given a signature — red."""
from neurobrix.triton import replay


def test_a_run_that_frees_handed_inputs_declines_the_replay():
    seq = type("Piece", (), {"_is_multi_device": False, "_replay_ineligible": False,
                             "_consumed_input_tids": frozenset({"input::hidden"})})()
    assert replay.signature(seq) is None
    assert any("frees the inputs" in r for r in seq.__dict__["_replay_declined"])
