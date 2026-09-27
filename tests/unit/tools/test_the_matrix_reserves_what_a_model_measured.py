"""The matrix reserves for a model the host peak it MEASURED (plus a margin), the static estimate only
for a model it never measured, and every row records the cell's host peak.

2026-09-27 14:20 (the supervisor's reading): three cards at 0 % while the ledger held card 0 behind
188 GB of reservations — 1.7x of each model's weights — and the four running cells used 10.3, 1.1, 0.7
and 0.5 GB of RSS. Before this branch no row carries a peak and every reservation is the estimate:
these fail.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

MB = 1 << 20


def test_the_peak_of_a_child_is_measured():
    with R.PeakRSS(interval=0.1) as peak:
        subprocess.run([sys.executable, "-c",
                        "import time; b = bytearray(300 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); time.sleep(1.0)"],
                       check=True)
    assert peak.peak >= 280 * MB, peak.peak


def test_a_measured_model_reserves_its_peak_plus_the_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "container_bytes", lambda m: 100 << 30)
    assert R.host_need("m", tmp_path) == (int((100 << 30) * R.HOST_PER_WEIGHT_BYTE), "estimate")
    rows = [{"model": "m", "mode": "native", "host_peak_rss": 10 << 30},
            {"model": "m", "mode": "triton", "host_peak_rss": 12 << 30},
            {"model": "other", "mode": "native", "host_peak_rss": 90 << 30}]
    (tmp_path / "rows_card0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert R.host_need("m", tmp_path) == (int((12 << 30) * R.MEASURED_MARGIN), "measured")


def test_a_row_records_the_cells_host_peak(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "container_bytes", lambda m: 1 << 30)

    def fake_run_cell(model, mode, gpu, out, timeout, src):
        subprocess.run([sys.executable, "-c",
                        "import time; b = bytearray(200 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); time.sleep(1.5)"],
                       check=True)
        return {"model": model, "mode": mode, "rc": 0}
    monkeypatch.setattr(R, "_run_cell", fake_run_cell)
    monkeypatch.setattr(R, "PEAK_SAMPLE_S", 0.1)
    row = R.run_cell("m", "native", "0", tmp_path, 60, tmp_path)
    assert row["host_peak_rss"] >= 180 * MB and row["host_reserved_from"] == "estimate"
    assert row["host_reserved"] == int((1 << 30) * R.HOST_PER_WEIGHT_BYTE)
