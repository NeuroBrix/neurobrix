"""The certified directory keeps only what the census table names (the supervisor, 2026-10-07 17:37).

The table is the single reference: a certified entry whose (kernel, key) is in no row of the
profile's census table is a certificate for a shape no catalogue container forms. The tool
retires exactly those, file by file, with a reversible record and a dated paragraph that lists
the keys; an entry the table names is kept even when its key text appears for another kernel.

What these do on the tool before 2026-10-07: `--from-log` is required and there is no
`--table`, so every run exits 2 (argparse) and `retire_outside` does not import.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_census_retire_outside_the_table.py
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/census_retire_unreachable.py"
sys.path.insert(0, str(REPO / "tools"))

IN = "(64, 32, 16, True, False, 'fp16', 'fp16', 'fp16')"


def E(n):
    """An entry as the certifier files it: a config and a proof that says it was built."""
    return {"config": n, "proof": {"built": True}}


OUT = "(65, 33, 17, True, False, 'fp16', 'fp16', 'fp16')"


def _setup(tmp_path):
    d = tmp_path / "apple" / "apple_m4_pro"; d.mkdir(parents=True)
    (d / "matmul_kernel.fp16.json").write_text(json.dumps(
        {"format": "nbx-autotune-certified/2", "entries": {IN: E(1), OUT: E(2)}}))
    (d / "baddbmm_kernel.fp16.json").write_text(json.dumps(
        {"format": "nbx-autotune-certified/2", "entries": {OUT: E(3)}}))
    table = tmp_path / "18g.jsonl"
    table.write_text("\n".join(json.dumps(r) for r in (
        {"model": "m", "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "key": IN},
        {"model": "m", "kernel": "neurobrix.kernels.ops.baddbmm_op.baddbmm_kernel", "key": OUT})) + "\n")
    return d, table


def test_it_retires_what_the_table_does_not_name_and_keeps_the_same_text_on_another_kernel(tmp_path):
    from census_retire_unreachable import retire_outside, table_keys
    d, table = _setup(tmp_path)
    keys = table_keys(table)
    kept, retired = retire_outside("matmul_kernel", {IN: E(1), OUT: E(2)}, keys)
    assert set(kept) == {IN} and set(retired) == {OUT}
    kept, retired = retire_outside("baddbmm_kernel", {OUT: E(3)}, keys)
    assert set(kept) == {OUT} and not retired, "a key text names a shape only WITH its kernel"


def test_the_real_run_rewrites_the_files_records_and_lists_the_keys(tmp_path):
    d, table = _setup(tmp_path)
    doc, rec = tmp_path / "retirements.md", tmp_path / "records"
    r = subprocess.run([sys.executable, str(TOOL), "--certified-dir", str(d), "--table", str(table),
                        "--record-dir", str(rec), "--record-doc", str(doc)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((d / "matmul_kernel.fp16.json").read_text())
    assert m["entries"] == {IN: E(1)} and m["format"] == "nbx-autotune-certified/2"
    assert json.loads((d / "baddbmm_kernel.fp16.json").read_text())["entries"] == {OUT: E(3)}
    record = json.loads(next(rec.glob("certified_retired_*.json")).read_text())
    assert record["entries"] == {"matmul_kernel.fp16.json": {OUT: E(2)}}, "reversible, per file"
    text = doc.read_text()
    assert "1 certified entries retired" in text and "3 → 2" in text and OUT in text


def test_it_refuses_an_empty_table_and_writes_nothing(tmp_path):
    d, table = _setup(tmp_path)
    table.write_text("")
    before = (d / "matmul_kernel.fp16.json").read_text()
    r = subprocess.run([sys.executable, str(TOOL), "--certified-dir", str(d), "--table", str(table),
                        "--record-dir", str(tmp_path / "r"), "--record-doc", str(tmp_path / "doc.md")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "REFUSED" in r.stderr
    assert (d / "matmul_kernel.fp16.json").read_text() == before
    assert not (tmp_path / "doc.md").exists()


def test_a_held_kernel_is_left_whole_and_named(tmp_path):
    """A kernel whose table rows are known wrong (2026-10-07: the baddbmm dtype placement) is held:
    its files are not touched, and the doc says it was held, so a later run retires it once fixed."""
    d, table = _setup(tmp_path)
    doc, rec = tmp_path / "retirements.md", tmp_path / "records"
    (d / "baddbmm_kernel.fp16.json").write_text(json.dumps(
        {"format": "nbx-autotune-certified/2", "entries": {IN: E(4), OUT: E(3)}}))
    before = (d / "baddbmm_kernel.fp16.json").read_text()
    r = subprocess.run([sys.executable, str(TOOL), "--certified-dir", str(d), "--table", str(table),
                        "--record-dir", str(rec), "--record-doc", str(doc), "--hold-kernels", "baddbmm_kernel"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (d / "baddbmm_kernel.fp16.json").read_text() == before
    assert json.loads((d / "matmul_kernel.fp16.json").read_text())["entries"] == {IN: E(1)}
    assert "held: `baddbmm_kernel`" in doc.read_text()
