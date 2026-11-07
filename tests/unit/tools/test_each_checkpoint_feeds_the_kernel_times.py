"""tools/certified_checkpoint.py feeds the cumulative kernel-time registry at every checkpoint (Hocine Benkelaya 2026-11-01: the
registry is fed at every certification, never by hand): each <vendor>/<profile> store's newly certified keys are
appended once, with their proof's best time."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("certified_checkpoint", REPO / "tools" / "certified_checkpoint.py")
CP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CP)


def test_a_checkpoint_appends_the_new_keys_times_once(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy(REPO / "tools" / "kernel_times.py", repo / "tools" / "kernel_times.py")
    store = repo / "src" / "autotune" / "apple" / "m4"
    store.mkdir(parents=True)
    (store / "k.fp32.json").write_text(json.dumps({"vendor": "apple", "profile": "m4", "kernel": "k", "dtype": "fp32",
        "entries": {"(1, 2)": {"proof": {"date": "2026-11-01T00:00:00+00:00", "best_ms": 0.5}}}}))
    reg = tmp_path / "kernel_times.tsv"
    monkeypatch.setattr(CP, "_KERNEL_TIMES", str(reg))
    for _ in range(2):                                          # a second checkpoint adds nothing already there
        CP._feed_kernel_times(str(repo), "src/autotune", lambda *a: None)
    rows = list(csv.DictReader(reg.open(), delimiter="\t"))
    assert [(r["kernel"], r["key"], r["best_ms"]) for r in rows] == [("k", "(1, 2)", "0.5")]
