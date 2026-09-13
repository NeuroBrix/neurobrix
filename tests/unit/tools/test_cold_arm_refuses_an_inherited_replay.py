"""A cold arm must not inherit a replay directory (seen: a re-run replayed the
previous run's sweep as its own control, 2026-09-13).

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_cold_arm_refuses_an_inherited_replay.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from precision_zoo_campaign import refuse_reused_replay  # noqa: E402


def test_a_fresh_or_empty_directory_passes(tmp_path):
    refuse_reused_replay(tmp_path / "B_replay_r0", allow=False)      # absent
    (tmp_path / "B_replay_r1").mkdir()
    refuse_reused_replay(tmp_path / "B_replay_r1", allow=False)      # empty


def test_an_inherited_replay_artifact_is_refused_and_named(tmp_path):
    rd = tmp_path / "B_replay_r0"; rd.mkdir(); (rd / "autotune_configs_cuda-70.json").write_text("{}")
    with pytest.raises(SystemExit, match="already holds a replay artifact"):
        refuse_reused_replay(rd, allow=False)
    refuse_reused_replay(rd, allow=True)                              # the deliberate opening
