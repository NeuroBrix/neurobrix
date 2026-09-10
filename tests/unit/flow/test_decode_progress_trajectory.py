"""The per-token decode trajectory: one shape, one site, every flow.

A run that succeeds and leaves no trajectory reads as "this model was not measured"
when the truth is "the tool could not see it". The autoregressive and encoder_decoder
flows emit their trajectory; the audio_llm flow keeps its own decode loop and emitted
none, so no audio row of a throughput table could carry a decode rate at all
(2026-09-08). `decode_progress.record` is the one site that shape now comes from.
"""
from __future__ import annotations

import re

from neurobrix import decode_progress


def test_record_writes_the_documented_line_and_feeds_the_listener(tmp_path, monkeypatch):
    path = tmp_path / "prog.txt"
    monkeypatch.setenv("NBX_DECODE_PROGRESS", str(path))
    seen = []
    decode_progress.set_listener(lambda *a: seen.append(a))
    try:
        decode_progress.record(0, 1, 4998, False)
        decode_progress.record(1, 2, 2, True)
    finally:
        decode_progress.clear_listener()

    lines = path.read_text().splitlines()
    assert len(lines) == 2, "one line per token, appended"
    m = re.fullmatch(r"t=(\d+\.\d{3}) step=0 n=1 last=4998 done=False", lines[0])
    assert m, f"the harness parses `t=` at millisecond resolution: {lines[0]!r}"
    assert lines[1].endswith("step=1 n=2 last=2 done=True")
    assert seen == [(0, 1, 4998, False), (1, 2, 2, True)]


def test_record_without_the_env_writes_nothing_and_still_emits(tmp_path, monkeypatch):
    monkeypatch.delenv("NBX_DECODE_PROGRESS", raising=False)
    seen = []
    decode_progress.set_listener(lambda *a: seen.append(a))
    try:
        decode_progress.record(3, 4, 77, False)
    finally:
        decode_progress.clear_listener()
    assert seen == [(3, 4, 77, False)], "the in-process channel is not gated by the file"
    assert not list(tmp_path.iterdir())


def test_both_audio_llm_loops_record_their_trajectory():
    """R30: the same event at the same site on both engines."""
    from pathlib import Path
    src = Path(decode_progress.__file__).parent
    for mod in ("core/flow/audio_llm.py", "triton/flow/audio_llm.py"):
        text = (src / mod).read_text()
        assert "decode_progress.record(" in text, f"{mod} emits no decode trajectory"
