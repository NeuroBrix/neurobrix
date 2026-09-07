"""A FAIL gate releases its staged build: the cache holds the installed copy and the backup the
old one, and the build step's deferral ("uploads must drain first") would otherwise wait for an
upload a failed gate never queues — 2026-09-07 20:16, two dead staged builds (34 GB) stalled
phase B. A gate that may still pass keeps its staging."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402


def _model(tmp):
    args = types.SimpleNamespace(out=str(tmp / "out"), backup=str(tmp / "backup"), models_root=str(tmp / "builds"),
                                 tmp=str(tmp / "tmp"), gpu=None, src=None, extra=[], timeout=10, trace_timeout=10,
                                 restore_mbps=10.0, upload_mbps=10.0)
    (tmp / "out" / "m").mkdir(parents=True)
    return R.Model("m", args)


def test_fail_releases_the_staged_build_and_says_so(tmp_path):
    m = _model(tmp_path)
    nbx = tmp_path / "builds" / "m" / "model.nbx"; nbx.parent.mkdir(parents=True); nbx.write_bytes(b"x" * 10)
    m.mark("build", True, nbx=str(nbx))
    assert m.release_staging("gate FAIL: test") is True
    assert not nbx.exists()
    st = json.loads(m.state_path.read_text())["steps"]["build"]
    assert st["staged_removed"]["why"] == "gate FAIL: test" and st["nbx"] == str(nbx)


def test_nothing_staged_is_a_no_op(tmp_path):
    m = _model(tmp_path)
    assert m.release_staging("gate FAIL: test") is False
    m.mark("build", True, nbx=str(tmp_path / "gone.nbx"))
    assert m.release_staging("gate FAIL: test") is False
