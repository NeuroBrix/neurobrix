"""`neurobrix run --turns` states each turn's own timings: its time to first token, its rate, its time per token.

The Dell's campaign (D10): from the second turn on, the [Timings] line repeated the first turn's time to first token.
The run ledger keeps the FIRST instant of each mark since it was opened (core/runtime/phase_trace), and the turns of
one process shared one ledger: every turn read turn 1's request start and first token, beside its own token tally.

The real `play_turns` runs here on a stand-in executor that emits its tokens through the engines' one emit site, on a
clock the test sets, so each turn's true figures are known. What this test does if the code is wrong: the second turn
says turn 1's 1.000 s to its first token instead of its own 3.000 s (red on main).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from neurobrix import decode_progress
from neurobrix.cli.commands.run import play_turns
from neurobrix.core.runtime import run_timings

NOW = [0.0]
# per turn: (seconds before its first token, its token count); one second between tokens
TURNS = [(1.0, 2), (3.0, 3)]


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **_):
        return [ord(c) for m in messages for c in m["content"]] + [1]


class _Executor:
    modules = {"tokenizer": _Tokenizer()}

    def __init__(self):
        self.turn = 0

    def setup(self):
        pass

    def execute(self, inputs, kv_session=None):
        to_first, n = TURNS[self.turn]
        self.turn += 1
        NOW[0] += to_first
        for i in range(n):
            decode_progress.emit(i, i + 1, 100 + i, i == n - 1)
            NOW[0] += 1.0
        NOW[0] -= 1.0                                   # the request ends at its last token
        ids = inputs["global.input_token_ids"]
        kv_session.record(ids, 0, len(ids), len(ids), 0)
        return {"global.transcription": "ok"}


def test_each_turn_states_its_own_time_to_first_token(monkeypatch, capsys):
    monkeypatch.setattr(time, "monotonic", lambda: NOW[0])
    NOW[0] = 0.0
    run_timings.open_container()
    NOW[0] = 10.0
    args = SimpleNamespace(model="stand-in", system=None, max_tokens=8)
    pkg = SimpleNamespace(manifest={"family": "llm"}, defaults={"max_tokens": 8})
    plan = SimpleNamespace(kv_cache_plan=SimpleNamespace(session_reach=4096, max_cache_len=4096))

    def drain():
        NOW[0] += 5.0                                   # the time between two turns: never a turn's own

    with pytest.raises(SystemExit) as done:
        play_turns(args, ["hello", "and then?"], _Executor(), pkg, plan, {}, drain)
    assert done.value.code == 0
    lines = [run_timings.parse_line(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith(run_timings.TAG)]
    assert [(t["ttft_s"], t["tokens"]) for t in lines] == [(1.0, 2), (3.0, 3)], lines
    assert lines[1]["tpot_ms"] == pytest.approx(1000.0) and lines[1]["tok_s"] == pytest.approx(3 / 5), lines[1]
    assert lines[1]["load_s"] == pytest.approx(0.0), "a later turn loads nothing: its load time is its own"
