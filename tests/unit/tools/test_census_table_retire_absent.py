"""`census_table.py retire-absent` retires the rows of a container gone from the cache, and only those.

A container renamed in the cache (the upscalers took their makers' names on 2026-10-05) leaves its
rows under the old name; the certifier would keep certifying keys of a graph nobody serves. The door
is seen saying yes (the absent model's rows go, recorded first) and no (a dry run writes nothing; an
empty or mis-pointed cache, which would retire everything, is refused by name).

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_census_table_retire_absent.py
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/census_table.py"
sys.path.insert(0, str(REPO / "src"))
from neurobrix.kernels import census_table as T  # noqa: E402


def _row(model, key):
    return {"model": model, "container": "c0ffee", "mode": "triton", "rungs_mb": None, "ops": ["aten.mm::1"],
            "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "key": key, "dtype": "fp16", "tool": "t"}


def _setup(tmp_path, in_cache=("kept",)):
    cache = tmp_path / "cache"
    for m in in_cache:
        (cache / m).mkdir(parents=True)
        (cache / m / "manifest.json").write_text("{}")
    table = tmp_path / "census" / "nvidia" / "volta" / "16g.jsonl"
    T.write(table, [_row("kept", "(1,)"), _row("kept", "(2,)"), _row("gone", "(3,)")])
    return cache, table


def _run(*args):
    return subprocess.run([sys.executable, str(TOOL), "retire-absent", *map(str, args)],
                          capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})


def test_it_retires_the_absent_model_and_records_its_rows_first(tmp_path):
    cache, table = _setup(tmp_path)
    r = _run(table, "--cache", cache, "--record", tmp_path / "rec", "--apply")
    assert r.returncode == 0, r.stderr
    assert {x["model"] for x in T.read(table)} == {"kept"} and len(T.read(table)) == 2
    rec = list((tmp_path / "rec").glob("nvidia_volta_16g.retired.*.jsonl"))
    assert len(rec) == 1
    assert [json.loads(l)["model"] for l in rec[0].read_text().splitlines()] == ["gone"]


def test_a_dry_run_names_it_and_writes_nothing(tmp_path):
    cache, table = _setup(tmp_path)
    before = table.read_bytes()
    r = _run(table, "--cache", cache, "--record", tmp_path / "rec")
    assert r.returncode == 0 and "gone" in r.stdout and "WOULD retire" in r.stdout
    assert table.read_bytes() == before
    assert not (tmp_path / "rec").exists()


def test_an_empty_cache_is_refused_by_name(tmp_path):
    cache, table = _setup(tmp_path, in_cache=())
    before = table.read_bytes()
    r = _run(table, "--cache", cache, "--record", tmp_path / "rec", "--apply")
    assert r.returncode != 0 and "no container" in r.stderr
    assert table.read_bytes() == before


def test_a_cache_holding_none_of_the_table_s_models_is_refused(tmp_path):
    cache, table = _setup(tmp_path, in_cache=("unrelated",))
    before = table.read_bytes()
    r = _run(table, "--cache", cache, "--record", tmp_path / "rec", "--apply")
    assert r.returncode != 0 and "mis-pointed" in r.stderr
    assert table.read_bytes() == before
