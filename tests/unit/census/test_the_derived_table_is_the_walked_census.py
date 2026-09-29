"""`derived_census.py table` writes THE census table from the derivation (no execution), and on a model
the walk and the derivation both reach it writes the walk's own rows: TinyLlama, both served Triton
modes, the 16 GB profile's top rung — the key set of the derived rows equals the committed walked rows.
The Mac's walked Apple rows proved wrong where the census door placed a component on a stand-in device
(2026-09-29 12:49); the derivation projects through the profile and matched the run.
Injection: a served mode dropped from the table's loop -> RED (that mode's rows are missing).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODEL = "TinyLlama-1.1B-Chat-v1.0"
PROFILE = "default-ff6008b7"


def _rows(path, model):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["model"] == model:
            out.setdefault(r["mode"], set()).add((r["kernel"], r["key"]))
    return out


def test_the_derived_rows_are_the_walked_rows(tmp_path):
    if not (REPO / "src/neurobrix/config/hardware" / f"{PROFILE}.yml").exists():
        pytest.fail(f"the rack's profile {PROFILE} is not in this tree (an ignored file: copy it in)")
    table = REPO / "src/neurobrix/config/census/nvidia/volta/16g.jsonl"
    walked = _rows(table, MODEL)
    assert walked, f"the committed table holds no walked rows for {MODEL}"
    scratch = tmp_path / "census" / "nvidia" / "volta"
    scratch.mkdir(parents=True)
    (scratch / "16g.jsonl").write_text(table.read_text())
    code = (f"import sys; sys.path.insert(0, {str(REPO / 'tools')!r}); sys.path.insert(0, {str(REPO / 'src')!r})\n"
            f"from neurobrix.kernels import census_table as T; from pathlib import Path\n"
            f"T.ROOT = Path({str(tmp_path / 'census')!r})\n"
            f"import derived_census as D\n"
            f"sys.exit(D.main(['table', '--models', {MODEL!r}, '--hardware', {PROFILE!r}, '--table', 'nvidia/volta',"
            f" '--rungs', '16384', '--logs', {str(tmp_path)!r}]))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    derived = _rows(scratch / "16g.jsonl", MODEL)
    assert set(derived) == {"triton", "triton-sequential"}, derived.keys()
    for mode in derived:
        assert derived[mode] == walked[mode], (mode, derived[mode] ^ walked[mode])
