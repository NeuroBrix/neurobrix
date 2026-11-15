"""A processor's code tree is the standard library only: tools/build_processor.py refuses, by file, line and name and
before any pinned byte is fetched, an import of anything else (a third-party package, the engine itself), a dynamic
import, or a file that is not Python source; a relative import of its own modules passes.

Red without the change: there is no tools/build_processor.py; with the import check removed, the build of a reader
that imports numpy passes."""
from __future__ import annotations

import pytest

from tests.unit.processors import toy
from tests.unit.processors.test_a_processor_artifact_is_built_the_same_on_any_host import tool

DEAD = "http://127.0.0.1:9"


@pytest.mark.parametrize("line, said", [
    ("import numpy", r"toy_letters/reader/reader.py:2 imports numpy"),
    ("import os, torch.nn", r"toy_letters/reader/reader.py:2 imports torch.nn"),
    ("from neurobrix.processors import messages", r"toy_letters/reader/reader.py:2 imports neurobrix.processors"),
    ("import importlib\nM = importlib.import_module('numpy')",
     r"toy_letters/reader/reader.py:3 imports by import_module\(\) \(a dynamic import\)"),
    ("M = __import__('numpy')", r"toy_letters/reader/reader.py:2 imports by __import__\(\)"),
])
def test_an_import_beyond_the_standard_library_is_refused_by_name(tmp_path, line, said):
    bp = tool()
    toy.write_source(tmp_path / "src", DEAD, reader="\n" + line + "\n" + toy.READER)
    with pytest.raises(bp.Refused, match="the code tree is the standard library only: .*" + said):
        bp.build(toy.NAME, tmp_path / "out", tmp_path / "src")
    assert not (tmp_path / "out").exists()


def test_a_file_that_is_not_python_source_is_refused_by_name(tmp_path):
    bp = tool()
    sub = toy.write_source(tmp_path / "src", DEAD)
    (sub / "reader" / "table.bin").write_bytes(b"\0")
    with pytest.raises(bp.Refused, match=r"toy_letters/reader/table.bin is not Python source"):
        bp.build(toy.NAME, tmp_path / "out", tmp_path / "src")


def test_the_standard_library_and_a_relative_import_pass(tmp_path):
    bp = tool()
    sub = toy.write_source(tmp_path / "src", DEAD, reader="import re, unicodedata\nfrom collections import abc\n"
                                                         "from . import table\n" + toy.READER)
    (sub / "reader" / "table.py").write_text("ROWS = 26\n")
    members = bp.code_members(sub, bp.read_source(sub))
    assert sorted(members) == ["code/LICENSE", "code/__init__.py", "code/reader.py", "code/table.py"]
