"""The dispatcher runs a candidate only on a COMPLETE snapshot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_dispatch as D  # noqa: E402


def _roots(monkeypatch, tmp_path):
    export = tmp_path / "export"; export.mkdir()
    monkeypatch.setattr(D, "SNAP_LOGS", tmp_path / "snap")
    (tmp_path / "snap").mkdir()
    real = D.has_snapshot.__globals__["Path"]
    monkeypatch.setattr(D, "Path", real)
    import retrace_dispatch
    src = retrace_dispatch.has_snapshot
    def patched(name):
        # the same rule over the test export only
        for cand in (D.ALIAS.get(name, name), name):
            p = export / cand
            if not (p.is_dir() and any(p.iterdir())):
                continue
            if any(p.rglob("*.incomplete")):
                return False
            if (D.SNAP_LOGS / f"{cand}.log").exists() and not (p / ".snapshot_complete").exists():
                return False
            return True
        return False
    return export, patched


def test_a_partial_directory_is_not_a_snapshot(tmp_path, monkeypatch):
    export, has = _roots(monkeypatch, tmp_path)
    d = export / "chatterbox"; (d / ".cache/huggingface/download").mkdir(parents=True)
    (d / "config.json").write_text("{}"); (d / ".cache/huggingface/download/w.safetensors.incomplete").write_bytes(b"x")
    assert has("chatterbox") is False


def test_a_repository_the_tool_touched_needs_the_marker(tmp_path, monkeypatch):
    export, has = _roots(monkeypatch, tmp_path)
    d = export / "granite"; d.mkdir(); (d / ".gitattributes").write_text("")
    (D.SNAP_LOGS / "granite.log").write_text("downloading")
    assert has("granite") is False
    (d / ".snapshot_complete").write_text("done")
    assert has("granite") is True


def test_an_old_complete_snapshot_the_tool_never_touched_is_present(tmp_path, monkeypatch):
    export, has = _roots(monkeypatch, tmp_path)
    d = export / "Kokoro-82M"; d.mkdir(); (d / "config.json").write_text("{}")
    assert has("Kokoro-82M") is True
