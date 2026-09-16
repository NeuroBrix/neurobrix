"""Two card classes are re-proven in parallel into two trees, and one class's
proofs are carried back without touching the other's.

Two certifiers may not write one file (2026-09-16: two cards shared a kernel
file and the second died on a rename). Each class writes its own tree; this
merge puts a class's proof into the destination's slot for that class and
leaves everything else alone.

Injection: the variant slot written unconditionally (ignoring which class the
destination's primary proof carries) → the 16 GB primary was overwritten by a
32 GB proof and the first test caught it."""
import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "merge_certified_class.py"


def _entry(cls, ver, config="A"):
    return {"config": {"BLOCK_M": config},
            "proof": {"backend": {"triton": ver, "name": "cuda"},
                      "machine": {"device": {"memory_mb": cls * 1024}}},
            "excluded": []}


def _file(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "nbx-autotune-certified/1", "vendor": "nvidia",
                                "profile": "volta", "kernel": "k", "dtype": "fp32",
                                "entries": entries}))


def _run(src, dst, cls, apply=True):
    cmd = [sys.executable, str(TOOL), "--from", str(src), "--into", str(dst), "--class", str(cls)]
    if apply:
        cmd.append("--apply")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_the_class_lands_in_its_own_slot_and_the_other_class_is_untouched(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"key1": _entry(32, "3.8.0", "NEW32")})
    _file(dst / "k.fp32.json", {"key1": _entry(16, "3.8.0", "OLD16")})
    rec = _run(src, dst, 32)
    assert rec["files"][0]["moved"] == 1
    out = json.loads((dst / "k.fp32.json").read_text())["entries"]["key1"]
    assert out["config"]["BLOCK_M"] == "OLD16", "the 16 GB primary was overwritten"
    assert out["variants"]["32g"]["config"]["BLOCK_M"] == "NEW32"


def test_a_destination_whose_primary_is_that_class_takes_the_new_proof(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"key1": _entry(32, "3.8.0", "NEW")})
    _file(dst / "k.fp32.json", {"key1": _entry(32, "3.6.0", "OLD")})
    _run(src, dst, 32)
    out = json.loads((dst / "k.fp32.json").read_text())["entries"]["key1"]
    assert out["config"]["BLOCK_M"] == "NEW"
    assert out["proof"]["backend"]["triton"] == "3.8.0"


def test_a_key_only_in_the_source_is_copied_and_one_without_that_class_is_skipped(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"new": _entry(32, "3.8.0"), "other": _entry(16, "3.8.0")})
    _file(dst / "k.fp32.json", {"kept": _entry(16, "3.6.0")})
    rec = _run(src, dst, 32)
    out = json.loads((dst / "k.fp32.json").read_text())["entries"]
    assert set(out) == {"kept", "new"} and rec["files"][0]["source_without_that_class"] == 1


def test_without_apply_nothing_is_written(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _file(src / "k.fp32.json", {"key1": _entry(32, "3.8.0", "NEW")})
    _file(dst / "k.fp32.json", {"key1": _entry(16, "3.6.0", "OLD")})
    before = (dst / "k.fp32.json").read_text()
    _run(src, dst, 32, apply=False)
    assert (dst / "k.fp32.json").read_text() == before
