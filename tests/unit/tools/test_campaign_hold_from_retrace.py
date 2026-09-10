"""A campaign holds off every model a retrace still holds: listed in a phase file, or carrying a
state whose gate is not a PASS. CogVideoX-2b's proof row straddled phase B's install at 14:34 on
2026-09-07 — the held rule the levers' scheduler applied was missing from the campaign."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def _state(out, m, gate):
    (out / m).mkdir(parents=True, exist_ok=True)
    (out / m / "state.json").write_text(json.dumps({"steps": {"gate": gate} if gate else {}}))


def test_held_are_the_listed_and_the_ungated(tmp_path):
    (tmp_path / "phase_b_models.txt").write_text("Wan-VACE,PixArt-XL\n")
    _state(tmp_path, "Kokoro", {"ok": True, "verdict": "PASS"})                       # gated: free
    _state(tmp_path, "PixArt-XL", {"ok": True, "verdict": "PASS (the hub's object …)"})  # listed but PASS: free
    _state(tmp_path, "CogVideoX-2b", {"ok": False, "verdict": "FAIL"})                # held
    _state(tmp_path, "Voxtral", None)                                                  # traced, no gate yet: held
    assert C.held_by_retrace(tmp_path) == {"Wan-VACE", "CogVideoX-2b", "Voxtral"}
