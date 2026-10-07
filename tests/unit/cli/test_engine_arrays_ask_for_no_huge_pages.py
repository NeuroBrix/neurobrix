"""Unit test — the engine's process does not ask the kernel for huge pages behind its numpy arrays.

numpy reads NUMPY_MADVISE_HUGEPAGE once, at import, and otherwise madvises every large array MADV_HUGEPAGE. With
THP defrag "madvise" on a host fragmented by the engine's shared weight pools, each 2 MB fault then compacts memory
synchronously: on 2026-10-07 a clip over a decoded video sat 20+ minutes in system time while its card idled. The
CLI's pre-import startup therefore sets the variable to 0 before anything imports numpy, and keeps a user's value.

  PYTHONPATH=src python3 -m pytest tests/unit/cli/test_engine_arrays_ask_for_no_huge_pages.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[3] / "src")
PROBE = "import neurobrix.cli, numpy; print(numpy._core.multiarray._get_madvise_hugepage())"


def _advice(**env) -> str:
    e = {k: v for k, v in os.environ.items() if k != "NUMPY_MADVISE_HUGEPAGE"}
    e.update(PYTHONPATH=SRC, **env)
    return subprocess.run([sys.executable, "-c", PROBE], env=e, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_the_engine_process_asks_for_no_huge_pages():
    assert _advice() == "False"


@pytest.mark.parametrize("value,expected", [("1", "True"), ("0", "False")])
def test_a_user_value_is_kept(value, expected):
    assert _advice(NUMPY_MADVISE_HUGEPAGE=value) == expected
