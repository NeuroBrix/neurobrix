"""The precision campaign composes the tts request with the same brick as the
regression cells: an artefact that ships voices and declares none gets
`--speaker <first voice>`; one that declares its voice gets nothing.
2026-09-14 10:25: the campaign's Kokoro cell produced no output on both arms
("ships 54 voices and declares none as its default") — a harness composing
its own request excluded a family.

Injection: `speaker_args` returning [] made the first test RED; restored, green.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import artefact_voice as AV  # noqa: E402


def _artefact(root, name, voices, declared=None):
    d = root / name; (d / "modules" / "voices").mkdir(parents=True); (d / "runtime").mkdir()
    for v in voices:
        (d / "modules" / "voices" / f"{v}.pt").write_bytes(b"")
    (d / "runtime" / "defaults.json").write_text(json.dumps({"voice": declared} if declared else {"phoneme_lang": "a"}))


def test_an_artefact_with_voices_and_no_default_gets_the_first_voice(tmp_path):
    _artefact(tmp_path, "k", ["bf_emma", "af_heart", "am_adam"])
    assert AV.speaker_the_artefact_requires("k", tmp_path) == "af_heart"
    assert AV.speaker_args("k", tmp_path) == ["--speaker", "af_heart"]


def test_an_artefact_that_declares_its_voice_gets_nothing(tmp_path):
    _artefact(tmp_path, "v", ["a", "b"], declared="b")
    assert AV.speaker_args("v", tmp_path) == []
    assert AV.speaker_args("absent", tmp_path) == []


def test_the_campaign_request_for_tts_carries_the_speaker(tmp_path, monkeypatch):
    import precision_zoo_campaign as Z
    _artefact(tmp_path, "k", ["af_heart"])
    monkeypatch.setattr(Z, "CACHE", tmp_path)
    monkeypatch.setattr(Z, "family_stimulus", lambda fam: ["--prompt", "x"])
    args = Z.request_args("k", "tts", [])
    assert "--speaker" in args and args[args.index("--speaker") + 1] == "af_heart"
