"""tools/flightrec.py writes its records in the machine's run register (`docs/internal/_atelier.md` §2 and §4:
`<registres>/runs/flightrec/`), never in the repository (`<repo>/.flightrec/`, where it wrote until 2026-10-23), and
refuses by name where atelier.json declares no register. What this test does if the code were wrong: a record directory
under the repository, or a silent default where the register is undeclared -> red."""
import importlib
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def _load(monkeypatch, tmp_path, cfg):
    f = tmp_path / "atelier.json"
    f.write_text(json.dumps(cfg))
    monkeypatch.setenv("NBX_ATELIER_JSON", str(f))
    monkeypatch.syspath_prepend(str(TOOLS))
    for m in ("flightrec", "atelier"):
        sys.modules.pop(m, None)
    return importlib.import_module("flightrec")


def test_records_go_to_the_run_register(monkeypatch, tmp_path):
    reg = tmp_path / "registres"
    fr = _load(monkeypatch, tmp_path, {"registres": str(reg), "_writable": ["registres"]})
    d = fr.rec_dir()
    assert d == (reg / "runs" / "flightrec").resolve()
    assert TOOLS.parent not in d.parents


def test_an_undeclared_register_is_refused_by_name(monkeypatch, tmp_path):
    fr = _load(monkeypatch, tmp_path, {"tmp": str(tmp_path / "t"), "_writable": ["tmp"]})
    with pytest.raises(Exception, match="registres"):
        fr.rec_dir()
