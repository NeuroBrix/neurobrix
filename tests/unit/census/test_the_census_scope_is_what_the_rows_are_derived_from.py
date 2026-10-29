"""`derived_census.py table --only-moved` derives again only the models whose rows could move: each model's input
record (`census_table.replace_inputs`) holds what its rows were derived FROM — every plan, every unit of code the
derivation executed, every file it read — and a model is kept only when all of them are this tree's
(tools/census_inputs.py).
Injections: a measured function's body edited -> moved, named; an unexecuted function edited -> kept; a module
constant edited -> moved; a read file edited -> moved; a plan digest that differs -> the model derived again,
the plan named; the room-priced field of a plan dropped -> the derivation's keys unchanged (it reads none of it).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import census_inputs as CI                     # noqa: E402

MODEL = "TinyLlama-1.1B-Chat-v1.0"
PROFILE = "default-ff6008b7"
REL = "src/neurobrix/kernels/census_table.py"


def _measured_record(tmp_path):
    from neurobrix.kernels import census_table as T
    CI.install()
    CI.reset()
    T.dtypes_of("(4, 'fp16', 'fp32')")
    m = CI.take()
    sys.settrace(None)
    units, files = CI.units_and_files(m)
    return {"units": {k: v for k, v in units.items() if k.startswith(f"repo:{REL}::")}, "files": {}}


def _tree(tmp_path):
    root = tmp_path / "tree"
    (root / REL).parent.mkdir(parents=True)
    shutil.copy(REPO / REL, root / REL)
    return root


def test_an_executed_unit_moves_and_an_unexecuted_one_does_not(tmp_path):
    rec = _measured_record(tmp_path)
    assert f"repo:{REL}::dtypes_of" in rec["units"], sorted(rec["units"])
    assert f"repo:{REL}::key_line" not in rec["units"]          # never called: not an input
    assert f"repo:{REL}::<module>" in rec["units"]               # its constants are
    root = _tree(tmp_path)
    assert CI.static_move(rec, repo=root) == ""
    src = (root / REL).read_text()
    (root / REL).write_text(src.replace('return f"{kernel_qual}::{key_repr_text}"',
                                        'return f"{kernel_qual}:::{key_repr_text}"'))
    assert CI.static_move(rec, repo=root) == ""                  # key_line edited: not what was derived from
    (root / REL).write_text(src.replace('return ",".join(x for x in tup if isinstance(x, str))',
                                        'return ";".join(x for x in tup if isinstance(x, str))'))
    assert CI.static_move(rec, repo=root) == f"code repo:{REL}::dtypes_of changed"
    (root / REL).write_text(src.replace('COLUMNS = ("model",', 'COLUMNS = ("model", "extra",'))
    assert CI.static_move(rec, repo=root) == f"code repo:{REL}::<module> changed"


LIB_REL = "nbxtensor/src/nbxtensor/shape/__init__.py"


def test_the_librarys_executed_code_is_an_input_and_moves(tmp_path):
    """The key path runs in the tensor library since 2026-10-28; its executed code is recorded like the engine's, and
    an edit to it moves the model (it did not: the record's roots were src/ and tools/ only, and the table went stale
    for TinyLlama and whisper while nothing said so)."""
    from nbxtensor.shape import numel
    CI.install()
    CI.reset()
    numel((2, 3))
    m = CI.take()
    sys.settrace(None)
    units, _files = CI.units_and_files(m)
    rec = {"units": {k: v for k, v in units.items() if k.startswith(f"repo:{LIB_REL}::")}, "files": {}}
    assert f"repo:{LIB_REL}::numel" in rec["units"], sorted(units)[:20]
    root = tmp_path / "tree"
    (root / LIB_REL).parent.mkdir(parents=True)
    shutil.copy(REPO / LIB_REL, root / LIB_REL)
    assert CI.static_move(rec, repo=root) == ""
    src = (root / LIB_REL).read_text()
    (root / LIB_REL).write_text(src.replace("        n *= int(d)\n    return n", "        n = n * int(d)\n    return n"))
    assert CI.static_move(rec, repo=root) == f"code repo:{LIB_REL}::numel changed"


def test_a_record_taken_under_other_roots_is_derived_again():
    """A record taken before the library joined the roots holds no library unit: nothing in it can move when the
    library changes, so it is derived again by name — never kept as unchanged."""
    import derived_census as DC
    rec = {"container": "c", "jobs": {}, "units": {}, "files": {}, "roots": ["src", "tools"]}
    assert DC._moved("m", None, rec, "c", []) == (
        f"recorded under the roots ['src', 'tools'], this tree records {CI.recorded_roots()}")
    del rec["roots"]
    assert DC._moved("m", None, rec, "c", []).startswith("recorded under the roots None")
    assert "nbxtensor/src" in CI.recorded_roots()


def test_a_read_file_moves(tmp_path):
    root = tmp_path / "tree"
    f = root / "src/neurobrix/config/x.yml"
    f.parent.mkdir(parents=True)
    f.write_text("a: 1\n")
    rec = {"units": {}, "files": {"repo:src/neurobrix/config/x.yml": CI._file_sha(f)}}
    assert CI.static_move(rec, repo=root) == ""
    f.write_text("a: 2\n")
    assert CI.static_move(rec, repo=root) == "file repo:src/neurobrix/config/x.yml changed"


def test_a_room_field_leaves_the_plan_digest():
    import derived_census as D
    plan = {"strategy": "single_gpu", "components": [{"name": "model"}], "host_footprint": {"resident_bytes": 1}}
    moved_room = dict(plan, host_footprint={"resident_bytes": 2})
    moved_plan = dict(plan, strategy="layer_streaming")
    assert D._plan_digest(plan) == D._plan_digest(moved_room)
    assert D._plan_digest(plan) != D._plan_digest(moved_plan)


def _cache_has(model):
    import derived_census as D
    return (D.CACHE / model / "manifest.json").exists()


@pytest.mark.skipif(not (REPO / "src/neurobrix/config/hardware" / f"{PROFILE}.yml").exists(),
                    reason="the rack's profile is an ignored file")
def test_the_derivation_reads_no_room_field():
    import derived_census as D
    from trace_request import derived_request
    if not _cache_has(MODEL):
        pytest.fail(f"{MODEL} is not in the catalogue")
    req = derived_request(MODEL)
    whole = D.derive_keys(MODEL, PROFILE, "triton", None, req)[0]
    real = D.plan_record
    try:
        D.plan_record = lambda *a, **k: {k_: v for k_, v in real(*a, **k).items() if k_ not in D.ROOM_FIELDS}
        without = D.derive_keys(MODEL, PROFILE, "triton", None, req)[0]
    finally:
        D.plan_record = real
    assert whole and whole == without


def _table(tmp_path, *extra):
    code = (f"import sys; sys.path.insert(0, {str(REPO / 'tools')!r}); sys.path.insert(0, {str(REPO / 'src')!r})\n"
            f"from neurobrix.kernels import census_table as T; from pathlib import Path\n"
            f"T.ROOT = Path({str(tmp_path / 'census')!r})\n"
            f"import derived_census as D\n"
            f"sys.exit(D.main(['table', '--models', {MODEL!r}, '--hardware', {PROFILE!r}, '--table', 'nvidia/volta',"
            f" '--modes', 'triton', '--rungs', '16384', '--logs', {str(tmp_path / 'logs')!r}] + {list(extra)!r}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


@pytest.mark.skipif(not (REPO / "src/neurobrix/config/hardware" / f"{PROFILE}.yml").exists(),
                    reason="the rack's profile is an ignored file")
def test_only_moved_keeps_what_did_not_move_and_derives_a_moved_plan(tmp_path):
    if not _cache_has(MODEL):
        pytest.fail(f"{MODEL} is not in the catalogue")
    table = tmp_path / "census" / "nvidia" / "volta" / "16g.jsonl"
    table.parent.mkdir(parents=True)
    out = _table(tmp_path)
    assert f"[derived table] {MODEL}: " in out and "row(s) written" in out, out[-2000:]
    rows = table.read_text()
    rec_path = table.with_name("16g.inputs.jsonl")
    rec = json.loads(rec_path.read_text())
    assert rec["model"] == MODEL and len(rec["jobs"]) == 1 and rec["units"] and rec["files"]
    assert not any(f.startswith("repo:src/neurobrix/config/census/") or "16g.jsonl" in f for f in rec["files"])
    out = _table(tmp_path, "--only-moved")
    assert f"[derived table] {MODEL}: unchanged" in out, out[-2000:]
    assert table.read_text() == rows
    jid = next(iter(rec["jobs"]))
    rec["jobs"][jid] = "0" * 16
    rec_path.write_text(json.dumps(rec, sort_keys=True) + "\n")
    out = _table(tmp_path, "--only-moved")
    assert f"[derived scope] {MODEL}: derived again — plan {jid.rsplit('|', 1)[0]} moved" in out, out[-2000:]
    assert "row(s) written" in out
