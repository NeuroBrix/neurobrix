"""A certification carries one proof PER CODE GENERATOR; a re-proof under a new generator never un-serves another.

The Mac, 2026-11-08 19:23: `neurobrix autotune certify --reprove-generator` for the macOS 14 Metal wheel identity
(content-3a38a05dfda2da61) REPLACED each entry's one proof, so the published macOS 15 build (content-0c61dc742284af40)
lost every key it was served — addmv_kernel 54 keys, 54 proofs replaced — although the two builds compile
byte-identical kernels (.ttir/.ttgir/.msl/.metallib identical on 235-236 kernels x 4 builds). The schema now: a
certification (an entry's primary or a class variant) keeps its `proof` and carries the proofs of the SAME
configuration under other generators in `also_proven`; a generator under which the stored configuration failed and a
sweep chose another configuration files it as an `alternates` item of that class. Serving picks the class, then the
certification any of whose proofs names the running generator.

Seen red on 77ad7f651 (one proof per certification: `file_certification` replaced it), green after.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nbxtensor.ops import autotune_certified as C

REPO = Path(__file__).resolve().parents[3]
KEY = (10, 1536, 1536)
KTEXT = C.key_repr(KEY)
X = {"kwargs": {"BLOCK_M": 64}, "num_warps": 4, "num_stages": 3}
Y = {"kwargs": {"BLOCK_M": 32}, "num_warps": 2, "num_stages": 2}
A = {"triton": "3.8.0+gitPIN", "name": "metal", "backend_hash": "content-0c61dc742284af40"}   # the published macOS 15 build
B = {"triton": "3.8.0+gitPIN", "name": "metal", "backend_hash": "content-3a38a05dfda2da61"}   # the macOS 14 wheels


def _proof(backend, date="2026-11-08T18:00:00+00:00", memory_mb=16384, deviation=1e-6):
    return {"date": date, "engine_version": "0.6.2", "backend": dict(backend), "shape": list(KEY),
            "deviation": deviation, "tolerance": 1e-4, "oracle": "fp64",
            "machine": {"hostname": "t", "device": {"memory_mb": memory_mb}}, "built": {"gpu": True}}


def _cert(config, backend, **kw):
    return {"config": copy.deepcopy(config), "proof": _proof(backend, **kw), "excluded": []}


def _label(backend):
    return C.proof_backend({"backend": backend})


def _doc(entries):
    return {"format": C.format_for(entries), "vendor": "apple", "profile": "m4", "kernel": "nbxtensor.ops.k.matmul_kernel",
            "dtype": "fp32", "entries": entries}


def _valid(entries, tmp_path):
    d = tmp_path / "apple" / "m4"; d.mkdir(parents=True, exist_ok=True)
    path = d / "matmul_kernel.fp32.json"
    doc = _doc(entries)
    path.write_text(json.dumps(doc))
    return C.validate(json.loads(path.read_text()), path)


def _served(entries, gen, cls=16):
    return C.certification_for_generator(C.entry_for_memory_gb(entries.get(KTEXT), cls), _label(gen))


# (1) ---------------------------------------------------------------------------------------------------------------

def test_a_re_proof_under_a_new_generator_keeps_the_old_one_served_and_serves_the_new(tmp_path):
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, dict(_cert(X, B, date="2026-11-08T19:30:00+00:00")))
    a, b = _served(entries, A), _served(entries, B)
    assert a is not None, "the published generator's proof was replaced: its keys stopped serving"
    assert b is not None and b["config"] == X == a["config"]
    assert b["proof"]["backend"]["backend_hash"] == B["backend_hash"] and b["proof"]["date"].startswith("2026-11-08T19:30")
    assert a["proof"]["backend"]["backend_hash"] == A["backend_hash"]
    assert C.proof_backends(entries[KTEXT]) == {_label(A), _label(B)}
    assert C.covered_memory_gbs(entries[KTEXT]) == {16}, "two proofs of one class are one class"
    assert C.entry_covers(entries, KTEXT, 16, need_generator=_label(A))
    assert C.entry_covers(entries, KTEXT, 16, need_generator=_label(B))
    assert not _valid(entries, tmp_path)


def test_the_same_holds_in_a_memory_class_variant():
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(Y, A, memory_mb=32768))
    C.file_certification(entries, KTEXT, _cert(Y, B, memory_mb=32768))
    assert _served(entries, A, 32)["config"] == Y and _served(entries, B, 32)["config"] == Y
    assert _served(entries, A, 16)["config"] == X and _served(entries, B, 16) is None, \
        "a class is served only the generators its own proofs name"


# (2) ---------------------------------------------------------------------------------------------------------------

def test_a_re_proof_under_the_same_generator_replaces_that_generators_proof_only():
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B))
    C.file_certification(entries, KTEXT, _cert(X, A, date="2026-11-09T01:00:00+00:00"))
    proofs = C.certification_proofs(entries[KTEXT])
    assert sorted(_label(p["backend"]) for p in proofs) == sorted([_label(A), _label(B)]), "one proof per generator"
    assert _served(entries, A)["proof"]["date"].startswith("2026-11-09"), "A's proof is the fresh one"
    assert _served(entries, B)["proof"]["date"].startswith("2026-11-08"), "B's proof untouched"
    assert entries[KTEXT]["proof"]["backend"]["backend_hash"] == A["backend_hash"], \
        "the head keeps its place: a reader of `proof` alone still serves A"


def test_a_sweep_of_the_only_generator_still_replaces_the_certification():
    """The single-generator behaviour is unchanged: the class's only generator re-swept to another configuration
    replaces it (no alternate is kept for a generator that has a fresher certification)."""
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(Y, A))
    assert entries[KTEXT]["config"] == Y and C.ALTERNATES not in entries[KTEXT] and C.ALSO_PROVEN not in entries[KTEXT]


# (3) ---------------------------------------------------------------------------------------------------------------

def test_a_stored_configuration_failing_under_the_new_generator_gives_it_its_own(tmp_path):
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(Y, B))          # X failed the oracle under B; the sweep chose Y
    assert _served(entries, A)["config"] == X, "A keeps its configuration"
    assert _served(entries, B)["config"] == Y, "B is served its own"
    head = entries[KTEXT]
    assert head["config"] == X and _label(head["proof"]["backend"]) == _label(A), "the published reader's view is A's"
    assert [a["config"] for a in head[C.ALTERNATES]] == [Y]
    assert C.covered_memory_gbs(head) == {16}
    assert not _valid(entries, tmp_path)
    # B re-proves its own configuration later: its alternate is replaced in place, A untouched
    C.file_certification(entries, KTEXT, _cert(Y, B, date="2026-11-09T02:00:00+00:00"))
    assert len(entries[KTEXT][C.ALTERNATES]) == 1 and _served(entries, B)["proof"]["date"].startswith("2026-11-09")
    assert _served(entries, A)["config"] == X
    # a third generator proving X joins the head, not the alternate
    Cg = dict(A, backend_hash="content-ffffffffffffffff")
    C.file_certification(entries, KTEXT, _cert(X, Cg))
    assert _served(entries, Cg) is not None and _served(entries, Cg)["config"] == X
    assert not _valid(entries, tmp_path)


def test_a_generator_moving_from_the_head_to_another_configuration_leaves_the_head_to_the_others():
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B))
    C.file_certification(entries, KTEXT, _cert(Y, B))          # B re-swept to Y
    assert _served(entries, A)["config"] == X and _served(entries, B)["config"] == Y
    assert C.ALSO_PROVEN not in entries[KTEXT], "B's proof of X left with B's move"


def test_lookup_serves_each_generator_its_own_certification(monkeypatch, tmp_path):
    """End to end through `lookup`, the launch site's call, with the running generator set to A then B then a
    third that no proof names."""
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(Y, B))

    class _Tuner:
        pass

    monkeypatch.delenv("NBX_AUTOTUNE_ANY_GENERATOR", raising=False)
    monkeypatch.setattr(C, "enabled", lambda: True)
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))   # a file the directory holds: `_load` is the stand-in
    monkeypatch.setattr(C, "output_dtype", lambda tuner, k: "fp32")
    monkeypatch.setattr(C, "_load", lambda *a, **k: entries)
    for i, (gen, want) in enumerate(((A, X), (B, Y), (dict(A, backend_hash="content-0000000000000000"), None))):
        C.reset()
        monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / f"r{i}"))
        monkeypatch.setattr(C, "running_generator", lambda gen=gen: _label(gen))
        got = C.lookup("matmul_kernel", _Tuner(), KEY, memory_gb=16)
        assert (got["config"] if got else None) == want, (gen, got)
    C.reset()


def test_a_class_never_serves_another_class_s_certification_for_the_generator(monkeypatch, tmp_path):
    """`lookup` asks the generator inside the card's class: a 32 GB variant proven under B is not served to a
    16 GB card whose class only A proved."""
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(Y, B, memory_mb=32768))

    class _Tuner:
        pass

    monkeypatch.delenv("NBX_AUTOTUNE_ANY_GENERATOR", raising=False)
    monkeypatch.setattr(C, "enabled", lambda: True)
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))   # a file the directory holds: `_load` is the stand-in
    monkeypatch.setattr(C, "output_dtype", lambda tuner, k: "fp32")
    monkeypatch.setattr(C, "_load", lambda *a, **k: entries)
    monkeypatch.setattr(C, "running_generator", lambda: _label(B))
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "r"))
    C.reset()
    assert C.lookup("matmul_kernel", _Tuner(), KEY, memory_gb=16) is None
    C.reset()


# (4) ---------------------------------------------------------------------------------------------------------------

def test_a_single_proof_file_reads_as_before(tmp_path):
    entries = {KTEXT: _cert(X, A)}
    assert not _valid(entries, tmp_path)
    assert C.certification_proofs(entries[KTEXT]) == [entries[KTEXT]["proof"]]
    assert _served(entries, A) is entries[KTEXT] and _served(entries, B) is None


def test_a_reader_of_proof_alone_reads_a_multi_proof_entry_as_the_heads_certification():
    """The published readers know `proof` only: the head's `proof` and `config` are the first generator's, and the
    additive fields are ones those readers do not ask."""
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B))
    C.file_certification(entries, KTEXT, _cert(Y, dict(B, backend_hash="content-1111111111111111")))
    head = entries[KTEXT]
    assert head["config"] == X and head["proof"]["backend"] == A
    assert set(head) == {"config", "proof", "excluded", C.ALSO_PROVEN, C.ALTERNATES}


def test_the_gate_refuses_a_further_proof_that_does_not_re_read(tmp_path):
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B))
    entries[KTEXT][C.ALSO_PROVEN][0]["deviation"] = 1.0
    assert any("also_proven[0]" in p and "exceeds" in p for p in _valid(entries, tmp_path))
    entries[KTEXT][C.ALSO_PROVEN][0]["deviation"] = 1e-6
    entries[KTEXT][C.ALSO_PROVEN][0]["backend"] = dict(A)          # two proofs of one class naming one generator
    assert any("two proofs of one memory class" in p for p in _valid(entries, tmp_path))
    entries[KTEXT][C.ALSO_PROVEN][0]["backend"] = dict(B)
    entries[KTEXT][C.ALSO_PROVEN][0]["machine"]["device"]["memory_mb"] = 32768
    assert any("also_proven[0]" in p and "32 GB" in p for p in _valid(entries, tmp_path))


def test_the_format_stamp_asks_every_proof():
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B))
    assert C.format_for(entries) == C.FORMAT
    del entries[KTEXT][C.ALSO_PROVEN][0]["built"]
    assert C.format_for(entries) == "nbx-autotune-certified/1"


# (5) ---------------------------------------------------------------------------------------------------------------

def test_the_rename_tool_renames_every_proof_of_a_multi_proof_entry(tmp_path):
    old = dict(A, backend_hash="5690465c1234567")
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, old))
    C.file_certification(entries, KTEXT, _cert(Y, dict(B, backend_hash="5690465c1234567"), memory_mb=32768))
    C.file_certification(entries, KTEXT, _cert(X, B))
    C.file_certification(entries, KTEXT, _cert(Y, dict(B, backend_hash="5690465cfeedbee")))
    d = tmp_path / "apple" / "m4"; d.mkdir(parents=True)
    f = d / "matmul_kernel.fp32.json"
    f.write_text(json.dumps(_doc(entries)))
    r = subprocess.run([sys.executable, str(REPO / "tools" / "certified_rename_generator.py"), "--dir", str(d),
                        "--from", "5690465c1234567,5690465cfeedbee", "--to", "content-0c61dc742284af40",
                        "--evidence", "test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "total 3" in r.stdout, r.stdout
    after = json.loads(f.read_text())["entries"][KTEXT]
    hashes = sorted(p["backend"]["backend_hash"] for p in C.entry_proofs(after))
    assert hashes == sorted(["content-0c61dc742284af40"] * 3 + [B["backend_hash"]]), hashes


def test_the_checkpointers_gate_commits_a_multi_proof_file_and_refuses_a_bad_further_proof(tmp_path, monkeypatch):
    # the gate runs in the checkpointed repository's directory: the engine under test is named absolutely
    monkeypatch.setenv("PYTHONPATH", f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}")
    sys.path.insert(0, str(REPO / "tools"))
    import certified_checkpoint as CP

    def sh(*cmd, cwd=None):
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout

    r = tmp_path / "repo"; r.mkdir()
    sh("git", "init", "-q", "-b", "certify-branch", cwd=r)
    sh("git", "config", "user.email", "t@t", cwd=r); sh("git", "config", "user.name", "t", cwd=r)
    (r / "x.txt").write_text("x"); sh("git", "add", ".", cwd=r); sh("git", "commit", "-q", "-m", "base", cwd=r)
    bare = tmp_path / "origin.git"; sh("git", "init", "-q", "--bare", "-b", "certify-branch", str(bare))
    sh("git", "remote", "add", "origin", str(bare), cwd=r); sh("git", "push", "-q", "origin", "certify-branch", cwd=r)
    d = r / "src/neurobrix/config/autotune/apple/m4"; d.mkdir(parents=True)
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B))
    C.file_certification(entries, KTEXT, _cert(Y, dict(B, backend_hash="content-1111111111111111")))
    good = _doc(entries)
    (d / "matmul_kernel.fp32.json").write_text(json.dumps(good))
    bad = copy.deepcopy(good); bad["dtype"] = "fp16"; bad["kernel"] = "nbxtensor.ops.k.addmv_kernel"
    bad["entries"][KTEXT][C.ALSO_PROVEN][0]["deviation"] = 1.0           # a further proof above its tolerance
    (d / "addmv_kernel.fp16.json").write_text(json.dumps(bad))
    res = CP.checkpoint(str(r), "src/neurobrix/config/autotune", ["origin"], CP.DEFAULT_GATE, [], say=lambda *a: None)
    names = [Path(f).name for f in res["committed"]]
    refused = [Path(f).name for f in res["refused"]]
    assert "matmul_kernel.fp32.json" in names, res
    assert "addmv_kernel.fp16.json" in refused and "addmv_kernel.fp16.json" not in names, res


def test_the_certifier_re_proves_from_every_named_generator_and_never_the_running_one():
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(X, B, date="2026-11-08T19:30:00+00:00"))
    head = C.entry_for_memory_gb(entries[KTEXT], 16)
    third = _label(dict(A, backend_hash="content-2222222222222222"))
    assert C.reproof_source(head, third) == f"{_label(A)}, {_label(B)}"
    assert C.reproof_source(head, _label(B)) is None, "a class proven under the running generator is not re-proven"
    assert C.reproof_source(None, third) is None
    assert C.class_proven_since(head, "2026-11-08T19:00:00") and not C.class_proven_since(head, "2026-11-08T20:00:00")


def test_the_re_proof_coverage_tool_counts_a_class_once_and_done_when_any_proof_is_the_target(tmp_path):
    sys.path.insert(0, str(REPO / "tools"))
    import reproof_coverage as RC
    entries = {}
    C.file_certification(entries, KTEXT, _cert(X, dict(A, triton="3.7.0")))
    C.file_certification(entries, KTEXT, _cert(X, A))
    C.file_certification(entries, KTEXT, _cert(Y, dict(A, triton="3.7.0"), memory_mb=32768))
    (tmp_path / "matmul_kernel.fp32.json").write_text(json.dumps(_doc(entries)))
    rec = RC.coverage(tmp_path, "3.8.0+gitPIN")
    assert rec["by_memory_class"]["16g"] == {"total": 1, "done": 1, "behind": 0, "percent_done": 100.0}
    assert rec["by_memory_class"]["32g"]["total"] == 1 and rec["by_memory_class"]["32g"]["behind"] == 1
    assert rec["total"]["partly"] == 1 and rec["total"]["entries"] == 1
