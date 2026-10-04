"""A confirmation cell (and the census that must cover it) runs the family's smallest judging request.

The owner's method (2026-09-28, point 4): a matrix, gate or verification cell CONFIRMS a certified
model — steps, frames, resolution and tokens are the family's `confirmation:` section
(config/families/<family>.yml), data the owner changes, never a production request. One derivation
(`tools/trace_request.derived_request`) serves the matrix and the census, so the census covers
exactly the keys a confirmation run forms.

What each test would do if the code were wrong: a derivation that ignored the section keeps the
family bound (`--steps 20`) and the first test fails (seen red with the section not applied); one
that appended instead of replacing leaves two `--steps` and the second fails; the size the fake
container's trace gives under the section is checked by test_the_census_request_comes_from_the_
container_s_trace.py (320x512 from 960x1088, by hand).
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import trace_request as TR  # noqa: E402


def test_the_confirmation_values_replace_the_family_bound(monkeypatch):
    monkeypatch.setattr(TR.Z, "request_args", lambda model, family, extra: ["--prompt", "x", "--steps", "20"])
    monkeypatch.setattr(TR, "confirmation", lambda family: {"steps": 8, "size_fraction": 0.5})
    monkeypatch.setattr(TR, "container_topology", lambda model: {})      # a container that declares none
    monkeypatch.setattr(TR, "off_trace_size", lambda model, family: (320, 512))
    req = TR.derived_request("M", "image")
    assert req == ["--prompt", "x", "--steps", "8", "--height", "320", "--width", "512"], req


def test_a_flag_the_request_lacks_is_added_once():
    assert TR._with_flags(["--prompt", "x"], {"max_tokens": 32, "temperature": 0.0}) == \
        ["--prompt", "x", "--max-tokens", "32", "--temperature", "0.0"]
    assert TR._with_flags(["--max-tokens", "64"], {"max_tokens": 32}) == ["--max-tokens", "32"]


def test_every_lattice_family_names_its_confirmation_size():
    for family in TR.LATTICE:
        assert "size_fraction" in TR.confirmation(family), family


def test_a_speech_models_confirmation_is_one_short_sentence():
    """tts: the derived request is the family's confirmation sentence, shorter than its calibration
    sentence — a TTS decode is as long as the text it speaks (orpheus past 2 400 s on the Mac, 2026-10-04)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import trace_request as T
    from neurobrix.core.config import get_family_config
    cal = get_family_config("tts")["calibration"]["prompt"]
    req = T.derived_request("orpheus-3b-0.1-ft", "tts")
    said = req[req.index("--prompt") + 1]
    assert len(said.split()) < len(cal.split()), (said, cal)
