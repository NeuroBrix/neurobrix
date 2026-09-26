"""A matrix cell that passes its wall-clock budget is a red TIMEOUT row that names its last stage.

2026-09-27: three cells held their cards 40 minutes with the GPU idle (a VAE decoding on the CPU,
a fused MoE on the op-by-op path, a per-op dump) and their rows would have said only "TIMEOUT
after 3600s". The budget is now 15 minutes and the row carries where the cell was. On a row
without the stage this fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


def test_the_stage_is_the_last_progress_line_and_the_last_line(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("$ python -m neurobrix run\n[progress] M step 4/4 · 3.9min elapsed\n"
                   "[progress] M complete · 3.9min total\n   [Compiled] 'vae': loading 436 weights\n\n"
                   "TIMEOUT after 900s\n")
    st = R.last_stage(log)
    assert st["progress"] == "[progress] M complete · 3.9min total"
    assert st["last_line"] == "[Compiled] 'vae': loading 436 weights"


def test_the_budget_is_fifteen_minutes():
    assert 'r.add_argument("--timeout", type=int, default=900)' in Path(R.__file__).read_text()
