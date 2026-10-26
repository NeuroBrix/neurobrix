"""Every `neurobrix run` states four timings on one parseable line, in both engines.

The line is read from hooks the engines already carry — the phase ledger (container open,
request begin/end, `flow.first_token`, the load spans of setup and weight loads) and the
per-token tally of `decode_progress` — never from a parallel timer. A value the run does not
define is `na`, never a number. On a CLI that stopped printing the line, or a producer whose
line the parser cannot read back, these fail.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from neurobrix import decode_progress
from neurobrix.core.runtime import phase_trace, run_timings as RT

SRC = Path(RT.__file__).resolve().parents[3]


def test_a_decoding_run_states_load_ttft_tpot_and_rate():
    # open 0, begin 10, a 2 s weight load inside the request (10..12), first token 13, last 16
    t = RT.compute(t_open=0.0, t_begin=10.0, t_end=17.0, load_spans=[(10.0, 12.0), (10.5, 11.0)],
                   t_first=13.0, t_last=16.0, n_tokens=4)
    assert t["load_s"] == pytest.approx(12.0), "container to request + the load inside it"
    assert t["ttft_s"] == pytest.approx(1.0), "the load is never time to first token"
    assert t["tpot_ms"] == pytest.approx(1000.0), "three decode steps in three seconds"
    assert t["tok_s"] == pytest.approx(1.0), "four tokens over a four-second generation wall"
    assert (t["tokens"], t["first"]) == (4, "token")


def test_a_run_without_tokens_is_timed_to_its_output_and_says_na():
    t = RT.compute(0.0, 5.0, 9.0, [(5.0, 6.0)], None, None, 0)
    assert t["load_s"] == pytest.approx(6.0)
    assert t["ttft_s"] == pytest.approx(3.0)
    assert t["first"] == "output"
    assert t["tpot_ms"] is None and t["tok_s"] is None and t["tokens"] is None
    line = RT.format_line(t)
    assert line == "[Timings] load_s=6.000 ttft_s=3.000 tpot_ms=na tok_s=na tokens=na first=output"


def test_a_single_token_has_no_per_token_time():
    t = RT.compute(0.0, 1.0, 3.0, [], 2.0, 2.0, 1)
    assert t["tpot_ms"] is None and t["tokens"] == 1 and t["tok_s"] == pytest.approx(1.0)


def test_tokens_without_a_first_token_mark_are_refused():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        RT.compute(0.0, 1.0, 3.0, [], None, 2.0, 3)


def test_the_line_is_read_back_by_the_one_parser():
    t = RT.compute(0.0, 10.0, 17.0, [(10.0, 12.0)], 13.0, 16.0, 4)
    line = RT.format_line(t)
    assert line == "[Timings] load_s=12.000 ttft_s=1.000 tpot_ms=1000.00 tok_s=1.00 tokens=4 first=token"
    log = f"[Timing] Total execution: 7.00s\n{line}\n[Output] Generated 4 tokens\n"
    assert RT.parse_line(log) == {"load_s": 12.0, "ttft_s": 1.0, "tpot_ms": 1000.0, "tok_s": 1.0,
                                  "tokens": 4, "first": "token"}
    assert RT.parse_line("no line here\n") is None
    with pytest.raises(ValueError, match="lacks"):
        RT.parse_line("[Timings] load_s=1.000 ttft_s=na\n")


def test_the_hooks_fill_the_ledger_the_line_is_read_from():
    """The real hooks, end to end: the CLI's marks, a load span, the per-token emit site."""
    RT.open_container()
    with phase_trace.load_span():
        pass
    RT.begin_request()
    with phase_trace.load_span():
        pass
    for i in range(3):
        decode_progress.emit(i, i + 1, 100 + i, i == 2)
    RT.end_request()
    t = RT.measured()
    assert phase_trace.first(RT.FIRST_TOKEN) is not None, "the first emit stamps flow.first_token"
    assert t["tokens"] == 3 and t["first"] == "token"
    assert t["tpot_ms"] is not None and t["tok_s"] is not None
    assert RT.parse_line(RT.format_line(t))["tokens"] == 3
    # a fresh request starts a fresh tally: no token of the previous one is counted
    RT.open_container()
    RT.begin_request()
    RT.end_request()
    assert RT.measured()["first"] == "output"


def _cmd_run_calls():
    tree = ast.parse((SRC / "neurobrix" / "cli" / "commands" / "run.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cmd_run")
    return [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]


def test_the_run_command_prints_the_line_after_the_request():
    calls = _cmd_run_calls()
    printed = [c for c in calls if c.startswith("print(run_timings.format_line(run_timings.measured())")]
    assert printed, "cmd_run must print the [Timings] line, for every engine, after the request"
    for hook in ("run_timings.open_container()", "run_timings.begin_request()", "run_timings.end_request()"):
        assert hook in calls, f"cmd_run lost the {hook} mark the line is read from"


def test_every_decoding_flow_reaches_the_emit_site_on_both_engines():
    """R30: the first-token mark and the tally come from ONE site, reached from each engine."""
    for rel in ("core/module/autoregressive/generator.py", "triton/generator.py",
                "core/flow/encoder_decoder.py", "triton/flow/encoder_decoder.py",
                "core/flow/vlm.py", "triton/flow/vlm.py"):
        text = (SRC / "neurobrix" / rel).read_text()
        assert "decode_progress.emit(" in text, rel
    for rel in ("core/flow/audio_llm.py", "triton/flow/audio_llm.py"):
        assert "decode_progress.record(" in (SRC / "neurobrix" / rel).read_text(), rel


def test_the_peak_line_round_trips_and_says_na():
    """Rule 3 (2026-10-26): every run states its device peak on its own line; producer and parser are one pair.
    What this test would do if the code were wrong: a peak lost or invented — RED."""
    from neurobrix.core.runtime import run_timings as rt
    assert rt.parse_peak("x\n" + rt.format_peak(14290.4) + "\n") == 14290.0
    assert rt.parse_peak(rt.format_peak(None)) is None
    import pytest
    with pytest.raises(ValueError, match="no \\[Peak\\] line"):
        rt.parse_peak("[Timings] load_s=1.0")
