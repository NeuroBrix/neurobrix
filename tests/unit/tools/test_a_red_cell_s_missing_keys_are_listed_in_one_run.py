"""`tools/cell_missing_keys.py`: a red cell's command replayed past its first miss, every key it formed classified.

The certified-only cell stops at its first missing key; three gates in a row each named one more (2026-10-23/24). The
tool's contract: the replay is the row's own command without `--certified-only`, its output moved into the diagnosis
directory (never the gate's artefact); a row that is not a certified-only run is refused (there is nothing to diagnose);
every recorded key is classified against the tree's committed census table — a key the table lacks is ABSENT, one it
holds is served or uncertified, never absent.

What these tests would do if the code were wrong: a replay that kept `--certified-only` would stop at the first miss
again — the first case fails on it; an output left where the row wrote it would overwrite a gate artefact — the first
case fails on the path; a classifier that read keys loosely would call the table's own row absent — the third fails.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import cell_missing_keys as M  # noqa: E402


def test_the_replay_is_the_row_s_command_past_its_first_miss(tmp_path):
    row = {"command": "/py -m neurobrix run --model X --prompt 'a b' --certified-only --output /gate/X/triton.mp4"}
    cmd = M.diagnostic_command(row, tmp_path)
    assert "--certified-only" not in cmd and cmd[cmd.index("--prompt") + 1] == "a b"
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "diagnostic_triton.mp4")


def test_a_row_that_is_no_certified_only_run_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not a certified-only run"):
        M.diagnostic_command({"command": "/py -m neurobrix run --model X"}, tmp_path)


def test_a_key_the_table_holds_is_never_called_absent():
    table = REPO / "src/neurobrix/config/census/nvidia/volta/32g.jsonl"
    row = json.loads(table.read_text().splitlines()[0])
    held = f"{row['kernel']}::{row['key']}"
    made_up = f"{row['kernel']}::(3, 5, 7, 'fp32', 'fp32')"
    got = M.classify([held, made_up], REPO, 32, "nvidia", "volta")
    assert got["absent"] == [made_up], got
    assert held in got["uncertified"] or got["served"] == 1, got
