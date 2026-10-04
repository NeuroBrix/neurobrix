"""A placement refusal reaches the matrix row whole, not as its last line of advice.

The Mac's 15:36 (2026-10-04): the harness row for MiniCPM-o-4_5 read "4. A smaller model", so the log could not tell
a plan defect from a busy host. The row now carries every decline and the figures the planner read."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("regression_matrix", ROOT / "tools" / "regression_matrix.py")
RM = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("regression_matrix", RM)
_spec.loader.exec_module(RM)

LOG = """[Execute] Running pipeline...
[ERROR] Pipeline failed: This model cannot run on this machine.

Every strategy was tried, down to streaming one component at a time from disk:
  single_gpu, single_gpu_lifecycle, lazy_sequential, layer_streaming - ALL FAILED
  layer_streaming declined: no room for a single segment: the usable 5652 MB of the rung is filled by what stays resident beside the streamed ['llm.model'] - whole components 5502 MB

Components:
  llm.model: 15000MB (W=14000, A=1000)

Total GPU available: 7075MB

What would make it run:
  1. More host RAM - the streaming path needs 12045MB for that one component
  2. A GPU with more memory
  3. A smaller input (resolution, batch, context)
  4. A smaller model
"""


def test_the_row_carries_the_declines_and_the_free_figure(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text(LOG)
    err = RM.first_error(log)
    assert err.startswith("This model cannot run on this machine.")
    assert "no room for a single segment" in err and "whole components 5502 MB" in err
    assert "Total GPU available: 7075MB" in err
    assert "A smaller model" not in err


def test_pytest_prefixed_refusal_is_read_the_same(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text("\n".join("E       " + l for l in LOG.splitlines()))
    err = RM.first_error(log)
    assert "no room for a single segment" in err and "Total GPU available: 7075MB" in err
    plain = tmp_path / "plain.log"
    plain.write_text(LOG)
    assert err == RM.first_error(plain)       # the pytest margin, blank lines included, leaves no trace


def test_a_kill_or_timeout_still_names_itself(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text(LOG + "\nTIMEOUT after 900 s\n")
    assert RM.first_error(log).startswith("TIMEOUT after")


def test_an_ordinary_error_is_unchanged(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text("Traceback (most recent call last):\nValueError: bad shape\n")
    assert RM.first_error(log) == "ValueError: bad shape"
