"""The compiled engine's conv3d workspace on the M4 Pro is a MEASURED law with its source (the Dell 2026-11-05 00:20, decision 3.20):
the fraction of the input, the MPS heap floor, and the measurement that gave them. What this test does if a value loses its source or
the floor is dropped: RED — a number in a vendor file without its measurement is a hand-written value."""
from pathlib import Path

import yaml

APPLE = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors" / "apple" / "apple_m4_pro.yml"


def test_the_m4_pro_states_its_conv3d_workspace_law_and_where_it_was_measured():
    mem = (yaml.safe_load(APPLE.read_text()) or {}).get("memory") or {}
    assert float(mem["compiled_conv3d_workspace_input_fraction"]) >= 1.0
    assert int(mem["compiled_conv3d_workspace_floor_bytes"]) > 0
    src = str(mem.get("compiled_conv3d_workspace_source") or "")
    assert "measured" in src and "law.jsonl" in src, f"the conv3d workspace has no measurement named: {src!r}"
