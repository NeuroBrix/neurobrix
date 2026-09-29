"""The certifier's contract (docs/reference/tool-contracts.md, "The certifier"), one test per clause.

The tools audit of 2026-09-29 found: the lock on a kernel's certified file never contended by any
test (both "two writers" tests call the writer twice in one process); a certified file that does
not parse read as empty, so the next writer wrote over every proof in it; `restamp` rewriting a
file outside the certifier's lock; `--kernels ""` certifying every kernel and an unknown kernel
name certifying 0 with exit 0; an empty census table certifying 0 with exit 0 and an unparsable
row dropped in silence; no SIGTERM handler, so a killed pass lost the proofs its bounded writer
held.

What would this file do if the code were wrong? Each test names its injection; every one was run
and seen RED before this file was committed.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.kernels import autotune_certified as C
from neurobrix.kernels import autotune_certify as CF
from neurobrix.kernels import census_table as T

REPO = Path(__file__).resolve().parents[3]


def _cert(memory_mb: int, block_m: int) -> dict:
    return {"config": {"kwargs": {"BLOCK_M": block_m}, "num_warps": 4, "num_stages": 3},
            "proof": {"machine": {"device": {"memory_mb": memory_mb}}, "deviation": 1e-7},
            "excluded": []}


def _write(path, fresh, cache=None):
    CF._write_file(path, "nvidia", "volta", "q.matmul_kernel", "fp32", fresh, cache=cache)


# --- two writers ------------------------------------------------------------------------------

def _stalled_writer(path, key, go):
    real = CF._read_file

    def slow(p):
        entries = real(p)
        time.sleep(0.6)
        return entries
    CF._read_file = slow
    go.wait()
    _write(Path(path), {key: _cert(16384, 32)})


def test_two_certifier_processes_on_one_file_keep_both_proofs(tmp_path):
    """Two processes, each stalled between its read and its write, released together.
    Injection: the flock in `_write_file` removed -> both read the empty file, the last write
    drops the other's key, RED."""
    path = tmp_path / "matmul_kernel.fp32.json"
    ctx = mp.get_context("fork")
    go = ctx.Event()
    procs = [ctx.Process(target=_stalled_writer, args=(str(path), k, go)) for k in ("k1", "k2")]
    for p in procs:
        p.start()
    go.set()
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    assert set(json.loads(path.read_text())["entries"]) == {"k1", "k2"}


def test_a_restamp_waits_for_the_certifier_s_lock(tmp_path):
    """Injection: `restamp` without the lock -> it rewrites the file while the lock is held, RED."""
    import fcntl
    path = tmp_path / "matmul_kernel.fp32.json"
    _write(path, {"k1": _cert(16384, 32)})
    doc = json.loads(path.read_text())
    doc["format"] = "stale"
    path.write_text(json.dumps(doc))
    done = threading.Event()
    with open(path.with_suffix(".json.lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t = threading.Thread(target=lambda: (C.restamp(path), done.set()))
        t.start()
        time.sleep(0.5)
        assert not done.is_set(), "restamp rewrote the file under the certifier's lock"
        assert json.loads(path.read_text())["format"] == "stale"
        fcntl.flock(lk, fcntl.LOCK_UN)
    t.join(10)
    assert done.is_set() and json.loads(path.read_text())["format"] != "stale"


# --- a file that cannot be read -----------------------------------------------------------------

def test_a_corrupt_certified_file_is_refused_never_overwritten(tmp_path):
    """Injection: `_read_file` returning {} on a parse error -> the writer replaces the file by its
    one fresh entry, RED."""
    path = tmp_path / "matmul_kernel.fp32.json"
    path.write_text('{"entries": {"k0": ')                     # a file cut mid-write
    before = path.read_bytes()
    with pytest.raises(CF.UnreadableCertifiedFile, match="never overwritten"):
        _write(path, {"k1": _cert(16384, 32)})
    assert path.read_bytes() == before
    assert CF._read_file(tmp_path / "absent.json") == {}


# --- inputs that name nothing -------------------------------------------------------------------

SHAPES = {"neurobrix.kernels.ops.matmul.matmul_kernel": [(1,)],
          "neurobrix.kernels.ops.baddbmm_op.baddbmm_kernel": [(2,)]}


def test_the_kernel_list_is_refused_when_it_names_nothing_or_an_unknown_kernel():
    """Injection: `select_kernels` treating an empty list as None, or filtering an unknown name to
    nothing -> every kernel, or none with exit 0, RED."""
    assert CF.select_kernels(SHAPES, None) == SHAPES
    assert list(CF.select_kernels(SHAPES, ["matmul_kernel"])) == ["neurobrix.kernels.ops.matmul.matmul_kernel"]
    with pytest.raises(RuntimeError, match="names no kernel"):
        CF.select_kernels(SHAPES, [])
    with pytest.raises(RuntimeError, match="matmul_kernal"):
        CF.select_kernels(SHAPES, ["matmul_kernal"])


def test_the_command_line_refuses_an_empty_kernel_list(monkeypatch, capsys):
    """Injection: the CLI's check removed -> `--kernels ""` reaches certify as every kernel, RED
    (certify is replaced by a stub that fails the test if reached)."""
    from neurobrix.cli.commands import autotune as A
    monkeypatch.setattr(CF, "certify", lambda *a, **k: pytest.fail("certify was reached"))
    args = SimpleNamespace(action="certify", profile="volta", vendor="nvidia", kernels=" , ",
                           allow_uncheckpointed=True, census=None, out=None, limit=None, only_missing=True)
    assert A.cmd_autotune(args) == 2
    assert "names no kernel" in capsys.readouterr().out


def _row(key):
    return {"model": "M", "container": "c", "mode": "triton", "rungs_mb": [4096], "ops": ["aten.mm::0"],
            "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "key": key, "dtype": "fp32", "tool": "t"}


def test_an_empty_census_table_or_an_unreadable_key_is_refused(tmp_path, monkeypatch):
    """Injection: the empty-table or the unparsed-key refusal removed -> 0 certified with exit 0,
    or a key dropped in silence, RED."""
    table = tmp_path / "16g.jsonl"
    monkeypatch.setattr(T, "table_path", lambda *a, **k: table)
    table.write_text("")
    with pytest.raises(RuntimeError, match="holds no row"):
        CF.census("nvidia", "volta", 16)
    T.write(table, [_row("(64, 64, 64, 'fp32')"), _row("not a key")])
    with pytest.raises(RuntimeError, match="do not parse"):
        CF.census("nvidia", "volta", 16)
    T.write(table, [_row("(64, 64, 64, 'fp32')")])
    assert list(CF.census("nvidia", "volta", 16)) == ["neurobrix.kernels.ops.matmul.matmul_kernel"]


# --- every exit leaves its proofs ---------------------------------------------------------------

def test_sigterm_ends_a_pass_through_its_finally(tmp_path):
    """A child process holds a pending proof inside `sigterm_ends_through_finally`, is sent
    SIGTERM, and its `finally` must write the proof. Injection: the handler not installed ->
    SIGTERM kills the child before the finally, no file, RED."""
    out = tmp_path / "flushed.json"
    ready = tmp_path / "ready"
    code = textwrap.dedent(f"""
        import json, time, pathlib
        from neurobrix.kernels import autotune_certify as CF
        with CF.sigterm_ends_through_finally():
            try:
                pathlib.Path({str(ready)!r}).write_text("1")
                time.sleep(30)
            finally:
                pathlib.Path({str(out)!r}).write_text(json.dumps({{"k": "pending proof"}}))
    """)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="")
    p = subprocess.Popen([sys.executable, "-c", code], env=env)
    for _ in range(200):
        if ready.exists():
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    p.wait(20)
    assert json.loads(out.read_text()) == {"k": "pending proof"}
    assert p.returncode == 128 + signal.SIGTERM


# --- idempotence --------------------------------------------------------------------------------

def test_writing_the_same_proof_twice_leaves_the_same_bytes(tmp_path):
    """Injection: entries written unsorted -> a second write reorders them, RED."""
    path = tmp_path / "matmul_kernel.fp32.json"
    _write(path, {"kb": _cert(16384, 32), "ka": _cert(16384, 64)})
    once = path.read_bytes()
    _write(path, {"ka": _cert(16384, 64)})
    assert path.read_bytes() == once
