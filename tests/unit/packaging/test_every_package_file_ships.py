"""Every file the engine tracks under src/neurobrix ships in the package: no exclude pattern of the sdist (the wheel is
built from it, as CI builds it) matches a tracked package file, and a census file the package should carry is never
read as an empty table.

0.6.10 shipped without its census tables: the sdist's exclude list held "**/*.jsonl", every census table, chunk record
and refused register was left out, and `census_table.read*` answered an empty table for a missing file — Prism planned
as if no model had a certified prefill chunk, in silence (found the day it was published, 2026-11-12).

What this test does if the code is wrong: an exclude pattern that matches a tracked package file — red (naming the
files); a missing census table read as [] / {} — red; a certified hardware whose census files are all absent planned
as "no record" — red; a memory class without a census (a fact, not a broken install) refused — red. Card-less.

Run: python -m pytest tests/unit/packaging/test_every_package_file_ships.py
"""
import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _sdist_excludes():
    """The sdist target's exclude list, read from pyproject.toml (Python 3.10 has no tomllib: the section's quoted
    strings between `exclude = [` and `]`)."""
    text = (ROOT / "pyproject.toml").read_text()
    section = text[text.index("[tool.hatch.build.targets.sdist]"):]
    block = section[section.index("exclude = ["):]
    block = block[:block.index("]")]
    excludes = re.findall(r'"([^"]+)"', block)
    assert excludes, "no sdist exclude list read: the parser no longer matches pyproject.toml"
    return excludes


def _tracked_package_files():
    out = subprocess.run(["git", "ls-files", "src/neurobrix"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.splitlines() if p]


def _matches(path, pattern):
    p = pattern[2:] if pattern.startswith("**/") else pattern
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, p) or \
        any(fnmatch.fnmatch(part, p) for part in Path(path).parts)


def test_no_exclude_pattern_drops_a_tracked_package_file():
    dropped = sorted(f for f in _tracked_package_files() for pat in _sdist_excludes() if _matches(f, pat))
    assert not dropped, f"the sdist (and the wheel built from it) would leave out {len(dropped)} tracked file(s): " \
                        f"{dropped[:10]}"


def test_a_missing_census_file_is_refused_by_name(tmp_path):
    from neurobrix.kernels import census_table as T
    table = tmp_path / "nvidia" / "volta" / "16g.jsonl"
    for read in (T.read, T.read_chunks, T.read_inputs):
        with pytest.raises(FileNotFoundError, match="census file missing"):
            read(table)


def test_a_certified_hardware_without_its_census_is_a_broken_install(tmp_path, monkeypatch):
    from neurobrix.core.prism import prefill_chunk as PC
    from neurobrix.kernels import census_table as T
    census = tmp_path / "config" / "census"
    (tmp_path / "config" / "autotune" / "nvidia" / "volta").mkdir(parents=True)     # a certified hardware
    monkeypatch.setattr(T, "ROOT", census)
    with pytest.raises(FileNotFoundError, match="broken install"):
        PC.chunk_record(tmp_path, "nvidia", "volta", 16, "triton")
    (census / "nvidia" / "volta").mkdir(parents=True)
    (census / "nvidia" / "volta" / "32g.jsonl").write_text("")
    assert PC.chunk_record(tmp_path, "nvidia", "volta", 16, "triton") is None       # a class without census: a fact
    assert PC.chunk_record(tmp_path, "nvidia", "ampere", 16, "triton") is None      # uncertified hardware: a fact
