"""A gate row carries the run's four timings, parsed from the cell's own `[Timings]` line.

The harness reads the line with the engine's parser (core/runtime/run_timings.py), never a
second regex. Every row gets every field: a value the run does not define is None, and a log
without the line — a run that died first, a tree older than the line — leaves all of them None
instead of breaking the row. On a harness that stopped reading the line, these fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

FIELDS = ("load_s", "ttft_s", "tpot_ms", "tok_s", "tokens", "first_output")


def test_a_decoding_cell_row_carries_its_timings(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("[Timing] Total execution: 3.10s\n"
                   "[Timings] load_s=12.345 ttft_s=0.210 tpot_ms=31.50 tok_s=24.70 tokens=8 first=token\n"
                   "[Peak] device_mb=14290\n")
    assert R.timings_of(log) == {"load_s": 12.345, "ttft_s": 0.21, "tpot_ms": 31.5, "tok_s": 24.7,
                                 "tokens": 8, "first_output": "token", "device_peak_mb": 14290.0}


def test_a_non_decoding_cell_row_says_none_where_the_run_says_na(tmp_path):
    log = tmp_path / "triton.log"
    log.write_text("[Timings] load_s=40.000 ttft_s=95.125 tpot_ms=na tok_s=na tokens=na first=output\n")
    row = R.timings_of(log)
    assert row["ttft_s"] == 95.125 and row["first_output"] == "output"
    assert row["tpot_ms"] is None and row["tok_s"] is None and row["tokens"] is None


def test_an_older_log_keeps_its_row_with_every_field_none(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("[Timing] Total execution: 3.10s\n")
    for r in (R.timings_of(log), R.timings_of(tmp_path / "absent.log")):
        assert {k: r[k] for k in FIELDS} == {k: None for k in FIELDS}
        assert r["device_peak_mb"] is None and "no [Peak] line" in r["peak_error"]


def test_a_malformed_line_is_said_not_guessed(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("[Timings] load_s=1.000 ttft_s=2.000\n")
    row = R.timings_of(log)
    assert "lacks" in row["timings_error"] and row["load_s"] is None


def test_the_cell_row_is_built_with_the_timings():
    src = Path(R.__file__).read_text()
    body = src[src.index("def _run_cell("):]
    body = body[:body.index("\ndef ")]
    assert "**timings_of(log)" in body, "the cell row must carry the run's [Timings] fields"


def test_a_row_carries_the_runs_device_peak(tmp_path):
    """A cell's row carries `device_peak_mb` from the log's `[Peak]` line, and names why when there is none."""
    import regression_matrix as RM
    log = tmp_path / "c.log"
    log.write_text("[Timings] load_s=1.000 ttft_s=2.000 tpot_ms=na tok_s=na tokens=na first=output\n[Peak] device_mb=9876\n")
    assert RM.timings_of(log)["device_peak_mb"] == 9876.0
    log.write_text("[Timings] load_s=1.000 ttft_s=2.000 tpot_ms=na tok_s=na tokens=na first=output\n")
    r = RM.timings_of(log)
    assert r["device_peak_mb"] is None and "no [Peak] line" in r["peak_error"]
