"""A conv2d_forward_kernel certification is retired (`tools/conv_unit_routed.py`) only when no call can form its key on
the profile: every width a one-row key's W bucket holds is walked through the library's own `conv2d_launches`, not the
bucket's top alone.

What would this file do if the code were wrong? A check of the top width only retires a 1-D conv key whose bucket's
narrow members fall under the matrix unit's row floor and launch the FMA kernel (qa 1244-1: in_w 65..69, M 59..63 < 64)
-> the first test RED; a key whose width is no ladder top kept as formed -> the second RED; a 2-D key walked over
widths it does not bucket -> the third RED.
"""
import sys
from pathlib import Path

import nbxtensor
import pytest
from nbxtensor.ops import _configs

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import conv_unit_routed as T  # noqa: E402

VOLTA = Path(nbxtensor.__file__).parent / "backend" / "devices" / "nvidia" / "volta.yml"


@pytest.fixture
def unit():
    saved = dict(_configs._ACTIVE_PROFILE)
    yield T.bind_profile(VOLTA)
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(saved)


def _k(in_c, in_w, out_c, out_w, pw=1):
    return (1, in_c, 1, in_w, out_c, 1, out_w, 1, 7, 1, 1, 0, 0, 1, pw, 1, True, "fp16", "fp16", "fp16")


def test_a_one_row_key_whose_bucket_holds_a_width_under_the_unit_floor_is_not_retired(unit):
    key = _k(768, 80, 768, 74)                     # W bucket 65..80: the top (M 74) is the unit's, 65..69 (M 59..63) not
    formed, unit_all, ms = T.walk(key, unit)
    assert formed == [69, 68, 67, 66, 65] and unit_all is False and ms == (59, 74)
    assert T.verdict("never_reached", formed) == "failed_reachable"


def test_a_one_row_key_whose_width_is_no_ladder_top_is_formed_by_no_width(unit):
    key = (1, 1, 1, 3206, 32, 1, 3200, 1, 7, 1, 1, 0, 0, 1, 1, 1, True, "fp16", "fp16", "fp16")
    assert T.bucket_widths(key) == [] and T.walk(key, unit)[0] == []
    assert T.verdict("never_reached", []) == "retired_unit_routed"


def test_a_two_dimensional_key_keeps_its_one_width(unit):
    key = (1, 1, 3000, 128, 256, 1500, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, True, "fp16", "fp16", "fp16")
    assert T.bucket_widths(key) == [128]
    formed, unit_all, _ = T.walk(key, unit)
    assert formed == [] and unit_all is True     # M 96 000: the unit takes it, no key


def test_the_tool_writes_one_row_per_ledger_line(unit, tmp_path):
    led = tmp_path / "ledger.tsv"
    never = "RuntimeError: x: the wrapper never reached the autotuner"
    led.write_text("file\tkey\tmemory_class\toutcome\tseconds\treason\tdate\n"
                   f"nvidia/volta/conv2d_forward_kernel.fp16.json\t{_k(768, 80, 768, 74)}\t16g\tfailed\t1\t{never}\td\n"
                   f"nvidia/volta/conv2d_forward_kernel.fp16.json\t{_k(768, 80, 768, 74)}\t32g\tfailed\t1\t{never}\td\n"
                   f"nvidia/volta/other_kernel.fp16.json\t(1,)\t16g\tfailed\t1\t{never}\td\n")
    out = list(T.rows(led, unit))
    assert [(r[0], r[1], r[5], r[12]) for r in out] == [
        ("nvidia/volta/conv2d_forward_kernel.fp16.json", "16g", "failed_reachable", "65..69"),
        ("nvidia/volta/conv2d_forward_kernel.fp16.json", "32g", "failed_reachable", "65..69")]


def test_an_exception_row_makes_the_tool_exit_non_zero(unit, tmp_path, capsys):
    """1244-c1: EXCEPTION "must be 0" is a gate, not a line to read. A ledger row no verdict fits (here re-proven, yet no
    width forms its key) is printed EXCEPTION and the tool exits EXIT_EXCEPTION (3); a ledger with none exits 0."""
    never = "RuntimeError: x: the wrapper never reached the autotuner"
    apart = (1, 1, 1, 3206, 32, 1, 3200, 1, 7, 1, 1, 0, 0, 1, 1, 1, True, "fp16", "fp16", "fp16")
    head = "file\tkey\tmemory_class\toutcome\tseconds\treason\tdate\n"
    clean, bad = tmp_path / "clean.tsv", tmp_path / "bad.tsv"
    clean.write_text(head + f"nvidia/volta/conv2d_forward_kernel.fp16.json\t{apart}\t16g\tfailed\t1\t{never}\td\n")
    bad.write_text(head + f"nvidia/volta/conv2d_forward_kernel.fp16.json\t{apart}\t16g\tfailed\t1\t{never}\td\n"
                   f"nvidia/volta/conv2d_forward_kernel.fp16.json\t{apart}\t32g\treproven\t1\t\td\n")
    assert T.main([str(VOLTA), str(clean)]) == 0
    capsys.readouterr()
    assert T.main([str(VOLTA), str(bad)]) == 3
    out, err = capsys.readouterr()
    assert out.splitlines()[-1].split("\t")[5] == "EXCEPTION"
    assert "# exceptions 1" in err and "1 EXCEPTION row(s)" in err and T.EXIT_EXCEPTION == 3
