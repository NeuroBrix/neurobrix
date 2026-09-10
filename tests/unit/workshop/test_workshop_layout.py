"""The workshop root holds three kinds of thing, and worktrees live in one place.

Written 2026-09-10 after an audit found, on a rig whose root filesystem was 90 %
full: 55 git worktrees scattered across the home directory, 59 `nbx_*`
directories, 44 loose files, and 68 GB of unlabelled artefacts. The campaign
running that morning was measuring an engine thirteen commits behind its own
remote, from a directory whose name said nothing about it — and nothing on the
filesystem could have told anyone.

The discipline is written in `docs/reference/workshop-layout.md`. This is the
gate that keeps it true.

It tests the NATURE of each thing rather than a list of names: a repository is a
directory with a `.git`, a virtualenv is one with `pyvenv.cfg` or `bin/activate`,
a mount is a mount. A name list would go stale the first time something is
renamed, and would leak machine-specific names into a public repository.

Anything the machine's owner keeps at the root that is none of those — another
project, a package manager's directory — is declared once in the untracked file
`nbx/.root-exceptions`, one name per line. Declared, so it is a decision rather
than a drift.

Inert where the discipline is not installed: no `nbx/` root, no test.

Run: PYTHONPATH=src python -m pytest tests/unit/workshop/test_workshop_layout.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO.parent
WORKSHOP = ROOT / "nbx"

WORKSHOP_DIRS = ("worktrees", "builds", "stage", "logs", "stats", "campaigns", "tmp")


def _installed() -> bool:
    return WORKSHOP.is_dir()


def _exceptions() -> set:
    f = WORKSHOP / ".root-exceptions"
    if not f.is_file():
        return set()
    return {ln.strip() for ln in f.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}


def _is_repo(p: Path) -> bool:
    return (p / ".git").exists()


def _is_venv(p: Path) -> bool:
    """A virtualenv, or a directory that holds them.

    `ml/venv/`, `venvs/<name>/` and `bench_venvs/<name>/` are all the same
    nature — a place virtualenvs live — and the root allows that nature, not
    three particular names."""
    if (p / "pyvenv.cfg").is_file() or (p / "bin" / "activate").is_file():
        return True
    try:
        children = [c for c in p.iterdir() if c.is_dir()]
    except OSError:
        return False
    return bool(children) and all(
        (c / "pyvenv.cfg").is_file() or (c / "bin" / "activate").is_file()
        for c in children)


def _is_mount(p: Path) -> bool:
    try:
        return os.path.ismount(p)
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _installed(), reason=f"no workshop root at {WORKSHOP} — discipline not installed here")


def test_no_loose_file_at_the_workshop_root():
    """A file at the root belongs to nobody and is removed by nobody. Dotfiles
    are configuration and stay."""
    loose = sorted(p.name for p in ROOT.iterdir()
                   if p.is_file() and not p.name.startswith("."))
    assert not loose, (
        f"{len(loose)} loose file(s) at {ROOT} — they belong under nbx/logs/ or "
        f"the campaign directory that produced them: {loose[:12]}")


def test_the_root_holds_only_repositories_venvs_mounts_and_the_workshop():
    """Everything else has a home under nbx/."""
    allowed = _exceptions() | {"nbx"}
    strays = []
    for p in sorted(ROOT.iterdir()):
        if p.name.startswith(".") or not p.is_dir():
            continue
        if p.name in allowed or _is_repo(p) or _is_venv(p) or _is_mount(p):
            continue
        strays.append(p.name)
    assert not strays, (
        f"{strays} at {ROOT} is neither a repository, a virtualenv, a mount, nor "
        f"the workshop root. Move it under nbx/, or declare it in "
        f"{WORKSHOP / '.root-exceptions'} if it is not this project's business")


def test_every_worktree_lives_under_the_workshop_worktrees_directory():
    """A worktree anywhere else is a measurement nobody can attribute."""
    out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, f"git worktree list failed: {out.stderr}"
    paths = [Path(ln.split(" ", 1)[1]) for ln in out.stdout.splitlines()
             if ln.startswith("worktree ")]
    wt_root = WORKSHOP / "worktrees"
    strays = [str(p) for p in paths
              if p != REPO and wt_root not in p.resolve().parents]
    assert not strays, (
        f"worktree(s) outside {wt_root}: {strays}. Prove the commit is reachable "
        f"from a remote, then `git worktree remove` it, or re-create it under "
        f"{wt_root}")


@pytest.mark.parametrize("name", WORKSHOP_DIRS)
def test_each_workshop_directory_says_what_it_is_for(name):
    """A directory whose removal condition is not written down is a directory
    nobody dares empty — which is how 68 GB accumulated."""
    d = WORKSHOP / name
    assert d.is_dir(), f"{d} is missing from the workshop root"
    readme = d / "README.md"
    assert readme.is_file(), (
        f"{readme} is missing — every workshop directory states what it holds "
        f"and the condition under which its content may be removed")
    text = readme.read_text()
    assert "Removal condition" in text or "removal condition" in text, (
        f"{readme} does not state a removal condition")
