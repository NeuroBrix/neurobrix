"""The flight recorder's repository gate: `run` refuses to launch while `git fsck --full` is
not clean, the verdict is measured once per boot, and `fsck` re-checks by hand after the repair.
The injected corruption is the real one of 2026-09-03 00:12 — a loose object left empty by a
power cut — on a throwaway repository, so the gate is seen failing on git's own verdict."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import flightrec as F  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A one-commit repository whose records live beside it; the recorder points at it."""
    r = tmp_path / "repo"; r.mkdir()
    subprocess.run(["git", "init", "-q", r], check=True)
    (r / "a.txt").write_text("alpha\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", r, "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", r, "commit", "-q", "-m", "one"], check=True, env={**env, "PATH": "/usr/bin:/bin"})
    monkeypatch.setattr(F, "REPO", r)
    monkeypatch.setattr(F, "REC_DIR", r / ".flightrec")
    monkeypatch.setattr(F, "FSCK_FILE", r / ".flightrec" / "fsck.json")
    return r


def _run_args(cmd):
    return types.SimpleNamespace(label="t", note=None, gpu=None, log=None, resume_cmd=None, cmd=cmd)


def _empty_one_loose_object(r: Path) -> Path:
    """Empties the loose blob of a.txt — the object the repair rebuilds from the working tree."""
    sha = subprocess.run(["git", "-C", r, "rev-parse", "HEAD:a.txt"], check=True,
                         capture_output=True, text=True).stdout.strip()
    blob = r / ".git" / "objects" / sha[:2] / sha[2:]
    blob.chmod(0o644); blob.write_bytes(b"")
    return blob


def test_clean_repository_lets_run_launch(repo):
    assert F.cmd_run(_run_args(["true"])) == 0
    assert F.fsck_verdict()["clean"] is True
    assert [p for p in F.REC_DIR.glob("*.json") if p.name != "fsck.json"], "the job's record was written"


def test_empty_loose_object_refuses_run_and_writes_no_record(repo, capsys):
    _empty_one_loose_object(repo)
    assert F.cmd_run(_run_args(["true"])) == F.FSCK_REFUSED
    err = capsys.readouterr().err
    assert "REPOSITORY CORRUPT: RESUME REFUSED" in err and "is empty" in err
    assert not [p for p in F.REC_DIR.glob("*.json") if p.name != "fsck.json"], "no job record on a refusal"


def test_verdict_is_measured_once_per_boot_and_again_on_a_new_boot(repo, monkeypatch):
    calls = []
    real = F.run_git_fsck
    monkeypatch.setattr(F, "run_git_fsck", lambda: (calls.append(1), real())[1])
    F.fsck_verdict(); F.fsck_verdict()
    assert len(calls) == 1, "the same boot re-reads the recorded verdict"
    monkeypatch.setattr(F, "current_boot_id", lambda: "another-boot")
    F.fsck_verdict()
    assert len(calls) == 2, "a new boot is a new measurement"


def test_fsck_command_reflects_the_repair(repo):
    bad = _empty_one_loose_object(repo)
    assert F.cmd_fsck(None) == F.FSCK_REFUSED
    quarantine = repo.parent / "quarantine"; quarantine.mkdir()
    bad.rename(quarantine / bad.name)                       # the repair: the bad file out of .git/objects
    assert F.cmd_fsck(None) == F.FSCK_REFUSED, "a missing reachable object is still not clean"
    # the object is rebuilt from the working tree (the blob's content is a.txt's)
    subprocess.run(["git", "-C", repo, "hash-object", "-w", "a.txt"], check=True, capture_output=True)
    assert F.cmd_fsck(None) == 0
    assert F.cmd_run(_run_args(["true"])) == 0


def test_hook_prints_the_corruption_first(repo, capsys):
    _empty_one_loose_object(repo)
    F.cmd_check(types.SimpleNamespace(hook=True))
    out = capsys.readouterr().out
    assert out.startswith("=" * 64 + "\nFLIGHT RECORDER — REPOSITORY CORRUPT")
