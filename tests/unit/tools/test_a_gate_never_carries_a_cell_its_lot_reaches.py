"""A gate never carries a recorded verdict for a cell its own computed scope reaches (supervisor 2026-10-20 07:25).

Qwen3-VL-30B's long cells (over 600 s) were carried by record from 2026-10-16 through every gate after, state step
3 among them, which moved its decoding flow to one token per step and broke its decode: four days without a run.
A carry stands only for a cell the lot does not reach. What this test would do if the door were absent: the record
checked by the harness alone, no refusal -> red."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_run as G  # noqa: E402


def _gate(tmp, scope_lines, carried):
    C = tmp / "gate"
    (C / "scope").mkdir(parents=True)
    (C / "scope" / "queue_32g.txt").write_text("".join(l + "\n" for l in scope_lines))
    root = G.rows_root(C)
    root.mkdir(parents=True, exist_ok=True)
    (root / "carries.yml").write_text("carries:\n" + "".join(
        f"- {{model: {m}, mode: {mo}, source_gate: g, date: d, engine: e, reason: r}}\n" for m, mo in carried))
    return C


def test_a_carry_of_a_cell_in_the_scope_is_refused(tmp_path):
    C = _gate(tmp_path, ["Qwen3-VL-30B-A3B-Thinking native,triton 14400"], [("Qwen3-VL-30B-A3B-Thinking", "native")])
    with pytest.raises(G.Refused, match="a carry never survives a lot that reaches the run"):
        G.check_carries(Path("/nonexistent-tree"), C)


def test_the_scope_reads_every_class_and_every_mode(tmp_path):
    C = _gate(tmp_path, ["A native,triton 60", "B triton 60"], [])
    (C / "scope" / "queue_16g.txt").write_text("C native 60\n")
    assert G.scope_cells(C) == {("A", "native"), ("A", "triton"), ("B", "triton"), ("C", "native")}
