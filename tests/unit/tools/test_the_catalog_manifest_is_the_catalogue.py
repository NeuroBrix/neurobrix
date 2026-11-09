"""docs/catalog/manifest.json is the public catalogue the hub syncs from (tools/catalog_manifest.py; schema agreed with
the hub 2026-11-08, decided between agents 2026-11-09 17:18). It holds the 49 models, one entry per hub slug, each with
its container sha256 and bytes and the validation judged by sha, and no path or machine name.

What this test would do if the file were wrong: a model missing or doubled (red), a slug that is not the catalogue's
(red, where the internal catalogue exists), a path or a register machine name in the file (red), a validation row
without its output sha256 or with a branch the engine does not have (red), the tool not refusing a leak (red)."""
from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOC = json.loads((REPO / "docs/catalog/manifest.json").read_text())


def _tool():
    spec = importlib.util.spec_from_file_location("catalog_manifest", REPO / "tools/catalog_manifest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_49_models_one_entry_per_slug_with_the_schema():
    slugs = [m["slug"] for m in DOC["models"]]
    assert DOC["schema"] == 1 and len(slugs) == 49 and len(set(slugs)) == 49
    for m in DOC["models"]:
        assert set(m) == {"slug", "family", "category", "container", "neurotax", "licence", "validation"}, m["slug"]
        assert re.fullmatch(r"[0-9a-f]{64}", m["container"]["sha256"]) and m["container"]["bytes"] > 0, m["slug"]
        assert m["validation"], f"{m['slug']}: no judged run"
        for v in m["validation"]:
            assert v["branch"] in ("pytorch", "triton") and re.fullmatch(r"[0-9a-f]{64}", v["output_sha256"]), v
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", v["judged"]), v


def test_no_path_and_no_machine_name():
    text = (REPO / "docs/catalog/manifest.json").read_text()
    assert not re.search(r"/home/|/Users/|~/|\\\\|nasPath|model\.nbx", text)
    with open(REPO / "docs/reference/register-machines.tsv", newline="") as f:
        machines = [r["machine"] for r in csv.DictReader(f, delimiter="\t")]
    assert machines and not any(f'"{m}"' in text for m in machines)


def test_the_slugs_are_the_catalogue_s():
    import sys
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import atelier
        cat = Path(atelier.root("communication")) / "catalogue.tsv"
    except (ImportError, OSError, RuntimeError):        # no workshop file on this machine: nothing to compare with
        cat = REPO / "docs/internal/catalogue.tsv"
    if not cat.is_file():
        pytest.skip("the internal catalogue is not on this machine (gitignored): checked on the machine that builds the file")
    with open(cat, newline="") as f:
        slugs = {r["hub slug"].strip() for r in csv.DictReader(f, delimiter="\t") if r.get("hub slug", "").strip()}
    assert slugs == {m["slug"] for m in DOC["models"]}


def test_the_tool_refuses_a_leak(monkeypatch):
    t = _tool()
    monkeypatch.setattr(t, "_machines", lambda: ['"somebox"'])
    doc = {"schema": 1, "models": [{"slug": "a/b", "validation": [{"platform": "somebox"}]}]}
    with pytest.raises(RuntimeError, match="no path or machine name goes public"):
        t._refuse_leaks(doc, ["/x/cache", "/x/models"])
