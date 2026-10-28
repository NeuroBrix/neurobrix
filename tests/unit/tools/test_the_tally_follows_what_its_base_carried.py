"""A cell the base gate did not replay but CARRIED from the gate before it is carried again, from the gate that measured
it — the tally follows the base's scope/carried.tsv back. Before: the lot gate 2026-10-28 read only the stable gate's
rows and reported "carried from base 0, never measured 28" on both classes (the stable gate had carried those 28 from
the lot gate before it). What this test does if the code is wrong: the carried cell is never measured, exit 1."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_tally as T  # noqa: E402
from test_the_gate_tally_reads_time_and_judges_by_sha import _gate, _tally, _works  # noqa: E402


def test_a_cell_the_base_carried_is_carried_from_the_gate_that_measured_it(tmp_path, capsys):
    old = _gate(tmp_path / "old", [("m", "triton", 10.0, "aa"), ("n", "triton", 5.0, "cc")])
    base = _gate(tmp_path / "base", [("m", "triton", 10.0, "aa")])            # n carried, no row of its own
    (base / "scope").mkdir()
    (base / "scope" / "carried.tsv").write_text(
        "class\tmodel\tmode\tcarried_from\trow_date\trc\tverdict\tengine\tlast_row\treason\n"
        f"32g\tn\ttriton\t{old}/gate/matrix_32g\t2026-10-26T20:00:00Z\t0\t-\tx\tgreen\tsame architecture\n")
    gate = _gate(tmp_path / "g", [("m", "triton", 10.0, "aa")])
    (gate / "gate2" / "gate_32g.list").write_text("m triton\nn triton\n")
    assert _tally(tmp_path, gate, base, [_works("aa"), _works("cc")]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "never measured 0" in out and "carried from base 1" in out, out
