"""The dated, cumulative registry the benchmark feeds as it runs (the supervisor, 2026-11-02 11:55): every row a
kernel or model run writes into its device's TSV is also appended to one registry file, with the date, the device,
the level and the engine commit it measured — the history of a cell's time stays readable across runs, a row is
never rewritten.

    <registry dir>/nbxtensor_vs_torch.<level>.tsv     level = kernel | model
"""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

HEAD = ["date", "device", "commit"]


def commit(repo: Path) -> str:
    """The engine commit the run measures (short sha; a dirty tree says so)."""
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=10", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return f"{sha}+dirty" if dirty else sha


class Registry:
    """Appends each row, as it is written, to `<dir>/nbxtensor_vs_torch.<level>.tsv`; None writes nothing."""

    def __init__(self, directory, level: str, fields: list, device: str, repo: Path):
        self.path = None if directory is None else Path(directory) / f"nbxtensor_vs_torch.{level}.tsv"
        self.fields = HEAD + list(fields)
        self.stamp = {"device": device, "commit": commit(repo)}

    def append(self, row: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, delimiter="\t", extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({"date": time.strftime("%Y-%m-%d %H:%M:%S"), **self.stamp, **row})
