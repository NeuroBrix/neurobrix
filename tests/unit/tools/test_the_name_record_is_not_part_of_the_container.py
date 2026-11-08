"""The name record `import` writes beside an entry's manifest (neurobrix/nbx/names.py) is the machine's, not the
container's: a hub verification, a repack and the engine's own file list never count it.

Injection (RED, then GREEN; see the commit message): hub_cache_diff walking the raw tree — the verification reports
`name.json` as a member the hub object lacks, and the repack ships it.
"""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "hub_cache_diff.py"


def _tool():
    spec = importlib.util.spec_from_file_location("hub_cache_diff_names_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(tmp_path):
    e = tmp_path / "Model_1B"
    (e / "components" / "x").mkdir(parents=True)
    (e / "manifest.json").write_text(json.dumps({"model_name": "Model_1B"}))
    (e / "components" / "x" / "graph.json").write_text("{}")
    from neurobrix.nbx.names import record_name
    record_name(e, "vendor/Model_1B")
    return e


def test_a_hub_verification_and_a_repack_ignore_the_name_record(tmp_path):
    tool = _tool()
    e = _entry(tmp_path)
    files = tool.cache_all_files(e)
    assert sorted(files) == ["components/x/graph.json", "manifest.json"], sorted(files)
    nbx = tool.repack_from_cache(e, tmp_path / "out" / "model.nbx")
    with zipfile.ZipFile(nbx) as z:
        assert sorted(z.namelist()) == ["components/x/graph.json", "manifest.json"]
