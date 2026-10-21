"""A video gate cell runs the family's gate values (config/families/video.yml `gate:`), the container's own over them:
no video container states gate values, so every video cell ran its judging request (40-50 steps, hours a cell) where a
gate proves rc 0, zero miss and the agreement (Hocine 2026-10-21 02:47). Proven on one cell: Wan2.1-I2V-14B native at
2 steps, 777 s on a 16 GB V100 against 4.4 h at 50 (2026-10-21). What this test would do without the family's values:
the judging steps kept -> red."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import trace_request as T  # noqa: E402


def test_the_family_gate_steps_apply_without_a_container_value():
    assert T.model_gate("video", {"extracted_values": {"_global": {}}}) == {"steps": 2}


def test_the_container_s_gate_value_wins():
    topo = {"extracted_values": {"_global": {"gate": {"steps": 6, "source": "s"}}}}
    assert T.model_gate("video", topo) == {"steps": 6}
