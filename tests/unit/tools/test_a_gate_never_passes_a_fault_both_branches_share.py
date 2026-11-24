"""A fault SHARED by both branches (an empty text, a finish other than stop/length) is NOT COMPARED and makes the
gate RED; the tool defaults no eos, dtype or token count; its hardware copy is read-only."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_cell_same_request as G  # noqa: E402


def _ans(text, finish="stop"):
    """A branch's answer as the run records it."""
    return {"text": text, "finish": finish, "tokens": 3, "seconds": 1.0}


def test_both_empty_is_red():
    """Both branches answering an empty text with finish error is NOT COMPARED, not SAME."""
    ok, msg = G.compare("req1", {"compiled": _ans("", "error"), "triton": _ans("", "error")})
    assert not ok and "NOT COMPARED" in msg and "compiled (req1)" in msg and "triton (req1)" in msg


def test_an_empty_text_with_a_good_finish_is_red():
    """An empty text is refused even when its finish is stop."""
    ok, msg = G.compare("req1", {"compiled": _ans(""), "triton": _ans("")})
    assert not ok and "NOT COMPARED" in msg


def test_one_empty_names_the_branch():
    """One empty answer is RED and names that branch, the request, its text and finish."""
    ok, msg = G.compare("req2", {"compiled": _ans("Paris"), "triton": _ans("", "stop")})
    assert not ok and "triton (req2)" in msg and "compiled (req2)" not in msg and "text ''" in msg and "finish 'stop'" in msg


def test_a_finish_error_is_red():
    """A non-empty text with finish error is NOT COMPARED."""
    ok, msg = G.compare("req1", {"compiled": _ans("x", "error"), "triton": _ans("x", "error")})
    assert not ok and "finish 'error'" in msg


def test_valid_equal_texts_are_same():
    """Equal real answers finishing stop or length compare SAME."""
    ok, msg = G.compare("req1", {"compiled": _ans("Paris", "stop"), "triton": _ans("Paris", "length")})
    assert ok and "SAME" in msg


def test_different_texts_differ():
    """Two real, different answers DIFFER."""
    ok, msg = G.compare("req1", {"compiled": _ans("a"), "triton": _ans("b")})
    assert not ok and "DIFFER" in msg


def test_one_branch_only_names_it():
    """A single branch cannot be compared: the line names the branch."""
    ok, msg = G.compare("req1", {"triton": _ans("Paris")})
    assert not ok and "NOT COMPARED" in msg and "triton" in msg


def test_a_container_stating_no_value_is_refused_by_name(tmp_path):
    """No eos, dtype or max_tokens in the container: refused naming the key, never defaulted."""
    for key in ("eos_token_id", "dtype", "max_tokens"):
        with pytest.raises(SystemExit, match=key):
            G.container_value({"dtype": None}, key, tmp_path)
    assert G.container_value({"eos_token_id": [1, 2]}, "eos_token_id", tmp_path) == [1, 2]


def test_the_hardware_copy_is_read_only(tmp_path):
    """The copy's files are 0444 and a write to the copy leaves the original untouched."""
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "p.json").write_text("{}")
    G.read_only_copy(src, tmp_path / "dst")
    f = tmp_path / "dst" / "sub" / "p.json"
    assert (f.stat().st_mode & 0o222) == 0
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            f.write_text("x")
    assert (src / "sub" / "p.json").read_text() == "{}"
