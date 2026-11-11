"""The text front end of a phoneme model (Kokoro) hands ONE utterance at its own length: the ids are [1, n], the
lengths say n, and the padding mask is [1, n] with no position padded — on both engines, whatever the text.

Why it is pinned. The model's graphs carry ONE symbol for the text's extent: the capture declares that the padded
extent (the mask's, the encoder's) and the real length (`input_lengths`, which the vendor's packed LSTM stops at) are
one quantity — true for a text that is not padded beyond its own length, and only then. Measured against the vendor's
module (2026-11-11): equal at 17, 23 and 31 tokens; a text padded to 31 with 23 real tokens gets a DIFFERENT answer
(max abs diff 0.056) and nothing refuses it. No request can reach that today because these two functions build the
lengths from the text itself.

What this test would do if the code were wrong: the day a batch or a padding is added here, a padded request would
get a wrong answer in silence; this goes red first, and the change must bring the refusal by name with it.
"""
from types import SimpleNamespace

import numpy as np
import pytest

LENGTHS = [1, 5, 23, 31, 200]          # 23 is the length the graphs were captured at; the others are not


def _engine(device):
    resolved = {}
    ctx = SimpleNamespace(primary_device=device, nbx_path_str="/nowhere", pkg=SimpleNamespace(defaults={}),
                          variable_resolver=SimpleNamespace(resolved=resolved))
    return SimpleNamespace(ctx=ctx), resolved


def _a_text_of(monkeypatch, n):
    from neurobrix.core.module.audio import g2p
    monkeypatch.setattr(g2p, "phoneme_ids", lambda *a, **k: ("x" * n, list(range(1, n + 1))))
    monkeypatch.setattr(g2p, "request_voice", lambda *a, **k: "af_heart")


def _never_padded(n, ids, lengths, mask, resolved):
    assert ids.shape == (1, n) and ids.reshape(-1).tolist() == list(range(1, n + 1))
    assert lengths.reshape(-1).tolist() == [n], "the lengths are the text's own extent"
    assert mask.shape == (1, n) and not mask.any(), "one utterance: no position is padding"
    for same in (("global.text_lengths", "text_lengths", "input_lengths"), ("global.text_mask", "text_mask", "m"),
                 ("global.input_ids", "input_ids")):
        assert len({id(resolved[k]) for k in same}) == 1, f"{same} name one tensor"


@pytest.mark.parametrize("n", LENGTHS)
def test_the_triton_front_end_hands_the_text_at_its_own_length(monkeypatch, n):
    from neurobrix.triton import audio_frontend as fe

    class Host:                        # stands for NBXTensor.from_numpy: the test needs no device
        @staticmethod
        def from_numpy(a):
            return np.asarray(a)

    monkeypatch.setattr(fe, "NBXTensor", Host)
    monkeypatch.setattr(fe, "_set_device_for", lambda ctx: None)
    monkeypatch.setattr(fe, "_load_voicepack_np", lambda engine, count: None)
    _a_text_of(monkeypatch, n)
    engine, r = _engine("cuda:0")
    fe.preprocess_phonemizer_input_np(engine, "any text", {})
    _never_padded(n, r["input_ids"], r["input_lengths"], r["text_mask"], r)


@pytest.mark.parametrize("n", LENGTHS)
def test_the_pytorch_front_end_hands_the_text_at_its_own_length(monkeypatch, n):
    pytest.importorskip("torch")
    from neurobrix.core import audio_frontend as fe
    monkeypatch.setattr(fe, "_load_voicepack", lambda engine, count, device: None)
    _a_text_of(monkeypatch, n)
    engine, r = _engine("cpu")
    fe.preprocess_phonemizer_input(engine, "any text", {})
    _never_padded(n, r["input_ids"].numpy(), r["input_lengths"].numpy(), r["text_mask"].numpy(), r)
