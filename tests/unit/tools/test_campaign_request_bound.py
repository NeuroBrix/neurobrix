"""The campaign's bound on a request whose length the vendor leaves open. A decode loop gets a
token budget; a denoiser's loop gets a step count — Allegro's vendor default of 100 steps at
3.5 min each is 5.8 h for one arm and 23 h for a retrace gate's four, and the row timed out at
7200 s twice (2026-09-07/08) without ever producing a verdict. Both arms run the SAME request,
so the bound costs the comparison nothing; a family whose own stimulus names the flag keeps it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def _args(family, model="M", stimulus=(), extra=(), monkeypatch=None):
    monkeypatch.setattr(C, "family_stimulus", lambda f: list(stimulus))
    monkeypatch.setattr(C, "_declares_image_input", lambda m: False)
    return C.request_args(model, family, list(extra))


def test_a_video_request_carries_a_bounded_step_count(monkeypatch):
    args = _args("video", stimulus=["--prompt", "a red apple", "--seed", "42"], monkeypatch=monkeypatch)
    assert "--steps" in args and args[args.index("--steps") + 1] == "4"
    assert args[:4] == ["--prompt", "a red apple", "--seed", "42"]      # the stimulus is untouched


def test_a_family_that_names_the_flag_itself_keeps_its_own_value(monkeypatch):
    args = _args("video", stimulus=["--prompt", "p", "--steps", "20"], monkeypatch=monkeypatch)
    assert args.count("--steps") == 1 and args[args.index("--steps") + 1] == "20"


def test_the_decode_families_keep_their_token_budget(monkeypatch):
    for family, flag, value in (("llm", "--max-tokens", "64"), ("audio_llm", "--max-tokens", "64")):
        args = _args(family, stimulus=[], monkeypatch=monkeypatch)
        assert args[args.index(flag) + 1] == value


def test_an_unbounded_family_gets_no_bound(monkeypatch):
    args = _args("upscaler", stimulus=[], monkeypatch=monkeypatch)
    assert "--steps" not in args and "--max-tokens" not in args


def test_the_caller_s_extra_arguments_come_last(monkeypatch):
    args = _args("video", stimulus=["--prompt", "p"], extra=["--triton"], monkeypatch=monkeypatch)
    assert args[-1] == "--triton"
