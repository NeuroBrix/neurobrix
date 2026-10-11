"""`neurobrix run --help` describes what a flag does, not the project's
internal wagers.

Until 2026-10-12 the `--triton` help ended with "Should match or beat
--compiled on target shapes (project bet: custom kernels > cuDNN)": an
unmeasured claim addressed to the developers, printed to every user.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_run_help_speaks_to_the_user.py
"""
from __future__ import annotations

import re
import subprocess
import sys

import pytest

# Wording that belongs to the project's working notes, never to a user's help.
INTERNAL = ("project bet", "should match or beat", "custom kernels >")


def _run_help() -> str:
    out = subprocess.run([sys.executable, "-m", "neurobrix", "run", "--help"],
                         capture_output=True, text=True, check=True).stdout
    return re.sub(r"\s+", " ", out).lower()


def test_the_help_is_read():
    assert "--triton" in _run_help()


@pytest.mark.parametrize("phrase", INTERNAL)
def test_no_internal_wording(phrase):
    assert phrase not in _run_help(), f"`neurobrix run --help` says {phrase!r}"


def test_the_triton_flag_says_what_it_runs():
    text = _run_help()
    assert "no cudnn or cublas" in text and "triton" in text
