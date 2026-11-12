"""tools/package_files_check.py — the release gate that compares a built wheel and sdist with the tracked package files.

What this test does if the tool is wrong: a wheel or sdist missing a tracked file passes — red; a complete pair fails —
red; an empty input reads as a pass — red. Card-less, on synthetic archives built here.

Run: python -m pytest tests/unit/tools/test_package_files_check_names_what_a_build_left_out.py
"""
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("pfc", ROOT / "tools" / "package_files_check.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _pair(tmp_path, files):
    w, s = tmp_path / "n-1-py3-none-any.whl", tmp_path / "n-1.tar.gz"
    with zipfile.ZipFile(w, "w") as z:
        for f in files:
            z.writestr(f[len("src/"):], "x")
    with tarfile.open(s, "w:gz") as t:
        for f in files:
            info = tarfile.TarInfo("n-1/" + f); data = b"x"; info.size = len(data); t.addfile(info, io.BytesIO(data))
    return w, s


def test_a_complete_pair_passes_and_a_left_out_file_is_named(tmp_path, monkeypatch, capsys):
    files = ["src/neurobrix/a.py", "src/neurobrix/config/census/nvidia/volta/16g.jsonl"]
    monkeypatch.setattr(C, "tracked", lambda tree: files)
    w, s = _pair(tmp_path, files)
    assert C.main([str(w), str(s)]) == 0
    w2, s2 = _pair(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, files[:1])
    assert C.main([str(w2), str(s2)]) == 1
    assert "MISSING from the wheel: src/neurobrix/config/census/nvidia/volta/16g.jsonl" in capsys.readouterr().out


def test_an_empty_input_is_refused(tmp_path):
    (tmp_path / "e.whl").write_bytes(b""); (tmp_path / "e.tar.gz").write_bytes(b"")
    assert C.main([str(tmp_path / "e.whl"), str(tmp_path / "e.tar.gz")]) == 2
