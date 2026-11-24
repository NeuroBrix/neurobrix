"""A gate question is one where the model CHOOSES at every step: the check's margin and verdict, the questions data
file, and the run never gating an OPEN entry. Without a card, a model or the vendor: synthetic logits stand in."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_cell_same_request as G  # noqa: E402

FLOOR = 0.5


def _steps(*rows):
    """The top-2 pairs of synthetic logit rows, as the vendor worker records them."""
    return [G.top2(r) for r in rows]


def test_a_tie_is_open_at_its_position():
    """Two equal top logits give a zero margin: OPEN, at the tie's position, the lower id taken as the greedy pick."""
    steps = _steps([1.0, 24.0, 20.0], [24.0, 3.0, 24.0], [0.0, 10.0, 30.0])
    assert steps[1] == ((0, 24.0), (2, 24.0))
    assert G.min_margin(steps) == (0.0, 1)
    assert G.verdict(0.0, FLOOR) == "OPEN"


def test_a_margin_of_0_4_is_under_the_floor_and_3_0_is_over_it():
    """0.4 vendor logit units sits under the 0.5 floor (OPEN); 3.0 at every step clears it (GATE)."""
    near = _steps([21.0, 24.0, 23.6], [0.0, 27.0, 24.0])
    m, k = G.min_margin(near)
    assert (round(m, 6), k) == (0.4, 0) and G.verdict(m, FLOOR) == "OPEN"
    clear = _steps([21.0, 24.0, 18.0], [3.0, 0.0, 6.0])
    m, k = G.min_margin(clear)
    assert (m, k) == (3.0, 0) and G.verdict(m, FLOOR) == "GATE"
    assert G.verdict(0.5, FLOOR) == "GATE" and G.DEFAULT_FLOOR == 0.5


def test_the_data_file_beside_the_tool_reads_with_its_measures():
    """The file holds the gating set and the open, measured list; each entry its text, status and measures (minimum
    margin, position, token, card class, date of the measure); a GATE entry was measured at or over its floor."""
    data = G.read_questions()
    assert G.gating_questions(data) and all(isinstance(q, str) for q in G.gating_questions(data))
    for e in data["gating"] + data["open_measured"]:
        assert e["measures"], e["text"]
        for m in e["measures"]:
            assert {"min_margin", "position", "token", "card_class", "measured"} <= set(m), e["text"]
        if e["status"] == "GATE":
            assert all(m["min_margin"] >= m["floor"] for m in e["measures"]), e["text"]
    assert G.open_red_question(data) is not None


def test_an_entry_without_a_status_is_refused_by_name(tmp_path):
    """A text with no GATE/OPEN status is refused when the file is read, never sent."""
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"gating": [{"text": "q"}], "open_measured": []}))
    with pytest.raises(SystemExit, match="lacks its text or a GATE/OPEN status"):
        G.read_questions(p)


def test_the_run_never_gates_an_open_entry(tmp_path):
    """An OPEN entry is skipped even inside the gating list; the open red is the unpinned open entry."""
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"gating": [{"text": "a", "status": "GATE"}, {"text": "b", "status": "OPEN"},
                                        {"text": "c", "status": "GATE"}],
                             "open_measured": [{"text": "d", "status": "OPEN"},
                                               {"text": "e", "status": "OPEN", "pinned": False}]}))
    data = G.read_questions(p)
    assert G.gating_questions(data) == ["a", "c"]
    assert G.open_red_question(data) == "e"
    assert [b["messages"][-1]["content"] for b in G.bodies("m", "SYS", G.gating_questions(data))] == ["a", "c"]
