"""The retirement tool retires exactly what the certifier named — seen saying yes and no.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_census_retire_unreachable.py
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/census_retire_unreachable.py"
sys.path.insert(0, str(REPO / "tools"))
from census_retire_unreachable import retire, unreachable_from_logs  # noqa: E402

LOG_LINE = ("[certify] matmul_kernel fp16 M=67 N=65 K=33 IEEE_PRECISION=False PROMOTE_B=False fp16,fp16,fp16: "
            "UNREACHABLE — the wrapper computed key (67, 65, 33, True, True, 'fp16', 'fp16', 'fp32') for inputs "
            "synthesized from (67, 65, 33, False, False, 'fp16', 'fp16', 'fp16'): the census and the kernel disagree")
OLD = "neurobrix.kernels.ops.matmul.matmul_kernel::(67, 65, 33, False, False, 'fp16', 'fp16', 'fp16')"
NEW = "neurobrix.kernels.ops.matmul.matmul_kernel::(67, 65, 33, True, True, 'fp16', 'fp16', 'fp32')"
OTHER = "neurobrix.kernels.ops.baddbmm.baddbmm_kernel::(67, 65, 33, False, False, 'fp16', 'fp16', 'fp16')"


def test_it_retires_the_named_key_and_keeps_the_rest_including_the_same_key_text_on_another_kernel():
    named = unreachable_from_logs([_log(LOG_LINE)])
    assert named == {("matmul_kernel", "(67, 65, 33, False, False, 'fp16', 'fp16', 'fp16')"):
                     "(67, 65, 33, True, True, 'fp16', 'fp16', 'fp32')"}
    kept, retired = retire({OLD: {"a": 1}, NEW: {"b": 2}, OTHER: {"c": 3}}, named)
    assert set(retired) == {OLD}
    assert set(kept) == {NEW, OTHER}, "a key text names a shape only WITH its kernel"


def test_it_says_no_when_the_log_names_nothing(tmp_path):
    census = tmp_path / "c.json"; census.write_text(json.dumps({OLD: {"a": 1}}))
    r = subprocess.run([sys.executable, str(TOOL), "--from-log", str(_log("[certify] nothing here")),
                        "--census", str(census)], capture_output=True, text=True)
    assert r.returncode == 1 and "REFUSED" in r.stderr
    assert json.loads(census.read_text()) == {OLD: {"a": 1}}, "a refusal writes nothing"
    assert not list(tmp_path.glob("census_retired_*.json"))


def test_the_real_run_writes_backup_record_and_doc_and_is_reversible(tmp_path):
    census = tmp_path / "c.json"; census.write_text(json.dumps({OLD: {"a": 1}, NEW: {"b": 2}}))
    doc = tmp_path / "retirements.md"
    r = subprocess.run([sys.executable, str(TOOL), "--from-log", str(_log(LOG_LINE)), "--census", str(census),
                        "--record-dir", str(tmp_path), "--record-doc", str(doc)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(census.read_text()) == {NEW: {"b": 2}}
    rec = json.loads(next(tmp_path.glob("census_retired_*.json")).read_text())
    assert rec["entries"] == {OLD: {"a": 1}}, "reversible: the retired entry travels with its config"
    assert next(tmp_path.glob("c.json.bak.*")).exists()
    assert "1 keys retired" in doc.read_text() and "`matmul_kernel` | 1" in doc.read_text()


def _log(text: str) -> Path:
    import tempfile
    p = Path(tempfile.mkdtemp()) / "certify.log"; p.write_text(text + "\n"); return p
