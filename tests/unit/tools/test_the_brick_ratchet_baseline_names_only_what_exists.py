"""The brick ratchet's baseline names only packages the ratchet measures and files of this repository that exist.

When the library's copy left the engine, the baseline kept 111 functions, 63 private imports
and a long module under `nbxtensor/src/...`, paths of a tree that is no longer here: figures no run can measure, so no
run could ever hold them down. What this test does if a stale entry is back: a package missing from
`tools/brick_ratchet.json`, or a path that names no file, is RED by name."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASELINE = json.loads((REPO / "tools" / "brick_ratchet.baseline.json").read_text())
CONFIG = json.loads((REPO / "tools" / "brick_ratchet.json").read_text())


def _paths(entry: dict):
    """Every repository path a package's baseline names, with the list it is in."""
    for kind, sep in (("functions", "::"), ("modules", None), ("tensor_logic", None)):
        for key in entry.get(kind, {}):
            yield kind, key.split(sep, 1)[0] if sep else key
    for kind, sep in (("private", " <- "), ("direction", " -> "), ("mute", None)):
        for item in entry.get(kind, []):
            yield kind, item.split(sep, 1)[0] if sep else item


def test_the_baseline_records_only_the_packages_the_ratchet_measures():
    unmeasured = sorted(set(BASELINE) - set(CONFIG["packages"]))
    assert not unmeasured, (f"the baseline records {unmeasured}, which tools/brick_ratchet.json does not measure: run "
                            f"python tools/brick_ratchet.py --update")


def test_every_path_the_baseline_names_exists():
    missing = sorted({(name, kind, rel) for name, entry in BASELINE.items() for kind, rel in _paths(entry)
                      if not (REPO / rel).is_file()})
    assert not missing, (f"{len(missing)} baseline entries name a file that does not exist: run "
                         f"python tools/brick_ratchet.py --update: " + "; ".join(f"{n} {k} {r}" for n, k, r in missing[:20]))
