"""Two gate schedulers running at once (two gates) never place cells on the same card: a hold is CREATED
EXCLUSIVELY before a cell starts, so the second scheduler finds the card held. Reading holds and the device alone,
two gates placed Qwen3-Omni and DeepSeek-Coder on card 2 one second apart (2026-10-20 08:01) and the second cell died
out of memory. What this test would do if the hold were a plain write: both take it -> red."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_sched as S  # noqa: E402


def test_the_second_holder_of_a_card_is_refused_and_its_owner_releases_it(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "a", SimpleNamespace(holds=str(tmp_path)), raising=False)
    assert S.take_hold(2) is True
    assert S.take_hold(2) is False                      # the other scheduler sees it held
    (tmp_path / "hold_c3").write_text("999999\n")       # another holder's hold
    S.drop_hold(3)
    assert (tmp_path / "hold_c3").exists()              # never released by someone else
    S.drop_hold(2)
    assert not (tmp_path / "hold_c2").exists()
    assert S.take_hold(2) is True
