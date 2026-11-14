"""The publish workflow uploads only the files the release gate tested (tools/release_files_check.py): a file whose
bytes changed, a missing one or an extra one stops it before the upload step; an empty manifest or directory is refused.

What each test would do if the check were wrong: a check that compared names only passes the tampered wheel (the
second test fails); one that ignored extra files would let an untested file through (the third); one that treated an
empty manifest as nothing to check would pass anything (the fourth)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import release_files_check as R  # noqa: E402


def _release(tmp_path, files):
    dist = tmp_path / "dist"
    dist.mkdir()
    lines = []
    for name, data in files.items():
        (dist / name).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    manifest = tmp_path / "v0.0.0.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest, dist


FILES = {"neurobrix-0.0.0-py3-none-any.whl": b"wheel bytes", "neurobrix-0.0.0.tar.gz": b"sdist bytes"}


def test_the_tested_files_pass(tmp_path):
    manifest, dist = _release(tmp_path, FILES)
    assert R.main(["--manifest", str(manifest), "--dist", str(dist)]) == 0


def test_a_tampered_file_stops_the_publication(tmp_path, capsys):
    manifest, dist = _release(tmp_path, FILES)
    (dist / "neurobrix-0.0.0-py3-none-any.whl").write_bytes(b"rebuilt wheel bytes")
    assert R.main(["--manifest", str(manifest), "--dist", str(dist)]) == 1
    assert "neurobrix-0.0.0-py3-none-any.whl: sha256" in capsys.readouterr().out


def test_a_missing_or_extra_file_stops_it(tmp_path, capsys):
    manifest, dist = _release(tmp_path, FILES)
    (dist / "neurobrix-0.0.0.tar.gz").unlink()
    (dist / "untested.whl").write_bytes(b"x")
    assert R.main(["--manifest", str(manifest), "--dist", str(dist)]) == 1
    out = capsys.readouterr().out
    assert "absent from" in out and "untested.whl: in" in out


def test_an_empty_manifest_or_dist_is_refused(tmp_path):
    manifest, dist = _release(tmp_path, FILES)
    empty = tmp_path / "empty.sha256"
    empty.write_text("")
    assert R.main(["--manifest", str(empty), "--dist", str(dist)]) == 2
    (tmp_path / "none").mkdir()
    assert R.main(["--manifest", str(manifest), "--dist", str(tmp_path / "none")]) == 2
    bad = tmp_path / "bad.sha256"
    bad.write_text("not a hash  file.whl\n")
    assert R.main(["--manifest", str(bad), "--dist", str(dist)]) == 2
