"""A confirmation run is certified-only: a key the certified directory does not serve is an ERROR,
never a runtime sweep, and the local replay cache is neither read nor written.

The owner's method (2026-09-28 20:06): the census and the certifier do the autotune without running
models; every matrix, gate and verification run is a CONFIRMATION, served entirely from the
certified directory. Before this mode a miss always swept at runtime and the gates looked served:
queue-13's gate swept in 28 of 130 cells (401 misses), and on the Mac 39 % of a "served" gate's
key uses came from the replay cache — earlier sweeps, not certifications.

What each test would do if the code were wrong:
* the refusal removed from `_configs.run_with_notice` — the injected missing key sweeps and the run
  succeeds: `test_a_missing_key_fails_the_confirmation_run_naming_it` fails (seen red, 2026-09-28);
* the replay cache still seeded under the flag — the pre-seeded config serves the key and the run
  succeeds: `test_the_replay_cache_is_neither_read_nor_written` fails;
* `census_row` answering the wrong way round, or reading another class's table — the table tests fail.
The control (the same injected miss WITHOUT the flag sweeps and succeeds) proves the refusal is what
fails the first test, not the injection.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

SCRIPT = r'''
from neurobrix.kernels.nbx_tensor import NBXTensor
from neurobrix.kernels import wrappers as W
a = NBXTensor.empty((19, 2048), "float16", 0)
b = NBXTensor.empty((2048, 2048), "float16", 0)
out = W.mm(a, b)
print("OUT", tuple(out.shape))
'''


def _card():
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    return DeviceAllocator.device_count() > 0


needs_a_card = pytest.mark.skipif(not _card(), reason="needs a device to run the kernel")


def _run(tmp_path, **env_extra):
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(SRC), "NEUROBRIX_REPLAY_CACHE": str(tmp_path / "replay"),
                # the injected missing key: serving switched off, every key is a miss
                "NBX_AUTOTUNE_CERTIFIED": "off"})
    env.pop("NBX_AUTOTUNE_CERTIFIED_ONLY", None)
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=900)


@needs_a_card
def test_a_missing_key_fails_the_confirmation_run_naming_it(tmp_path):
    r = _run(tmp_path, NBX_AUTOTUNE_CERTIFIED_ONLY="1")
    assert r.returncode != 0, "the confirmation run swept a missing key instead of failing"
    assert "KeyNotCertified" in r.stderr and "CERTIFIED-ONLY: no certified setting for" in r.stderr, r.stderr[-1500:]
    assert "census table" in r.stderr, r.stderr[-1500:]
    assert not any((tmp_path / "replay").glob("*.json")), "a confirmation run wrote the replay cache"


@needs_a_card
def test_the_same_miss_without_the_flag_sweeps(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "OUT (19, 2048)" in r.stdout and "sweeping at runtime" in r.stdout


@needs_a_card
def test_the_replay_cache_is_neither_read_nor_written(tmp_path):
    seeded = _run(tmp_path)                       # a sweep lands in this replay cache
    assert seeded.returncode == 0 and any((tmp_path / "replay").glob("*.json")), seeded.stderr[-800:]
    before = {p.name: p.read_bytes() for p in (tmp_path / "replay").glob("*.json")}
    r = _run(tmp_path, NBX_AUTOTUNE_CERTIFIED_ONLY="1")
    assert r.returncode != 0, "the replay cache's earlier sweep served a confirmation run"
    after = {p.name: p.read_bytes() for p in (tmp_path / "replay").glob("*.json")}
    assert after == before


def _key():
    return ("neurobrix.kernels.ops.matmul.matmul_kernel", (19, 2048, 2048, True, True, "fp16", "fp16", "fp16"))


def _table(tmp_path, monkeypatch, rows):
    """A real census table (the P1 format) for nvidia/volta 16 GB under a tmp root, and that profile bound."""
    from neurobrix.kernels import census_table as T
    from neurobrix.kernels import autotune_certified as C
    monkeypatch.setattr(T, "ROOT", tmp_path / "census")
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))
    T.write(T.table_path("nvidia", "volta", 16), rows)
    return C


def test_a_key_absent_from_the_census_table_is_named_a_census_defect(tmp_path, monkeypatch):
    C = _table(tmp_path, monkeypatch, [])
    assert "CENSUS defect" in C.census_row(*_key(), 16)


def test_a_key_in_the_census_table_is_named_a_certification_gap(tmp_path, monkeypatch):
    from neurobrix.kernels.autotune_certified import key_repr
    qual, key = _key()
    row = {"model": "TinyLlama-1.1B-Chat-v1.0", "container": "3b8bbc487525959b", "mode": "triton", "rungs_mb": [16384],
           "ops": [None], "kernel": qual, "key": key_repr(key), "dtype": "fp16,fp16,fp16", "tool": "test"}
    C = _table(tmp_path, monkeypatch, [row])
    said = C.census_row(qual, key, 16)
    assert "CERTIFICATION gap" in said and "TinyLlama-1.1B-Chat-v1.0" in said
    other = C.census_row(qual, key, 32)            # the 32 GB class reads ANOTHER table, absent here
    assert "32g.jsonl" in other and "CERTIFICATION gap" not in other, other


def test_a_card_of_unknown_memory_names_no_table(monkeypatch):
    from neurobrix.kernels import autotune_certified as C
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))
    assert "memory class is unknown" in C.census_row(*_key(), None)


def test_the_run_command_takes_the_flag():
    from neurobrix.cli import create_parser
    args = create_parser().parse_args(["run", "--model", "m", "--certified-only"])
    assert args.certified_only is True
