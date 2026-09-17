"""How far a re-proof has got is one answer PER MEMORY CLASS, never one answer overall.

A setting is proven for one code generator and serves only the memory class it was proven on.
Raising Triton re-proves the directory, the pass takes a night across four cards, and nothing
said how far it had got: `autotune check` is the format gate and passes a 3.6.0 proof and a
3.8.0 proof alike (measured 2026-09-16 on a directory 63 % through: 12 files, 9 866 shapes,
0 refused), and `autotune status` does report "proven under" but resolves the hardware profile
first, so it answers nothing on a machine with no GPU and costs a CUDA context on a busy rig.

Two readings this file pins, because the first version of the tool got both wrong:

* an entry whose PRIMARY is re-proven and whose 32 GB variant is not is **partly** done. Counting
  the union called it done and hid half the work; counting it behind would hide the half that is
  finished. Measured on the live directory, that distinction moved the headline from "63 % done"
  to "0.8 % fully done, 6 075 partly" — and the second is what is true, because the 32 GB half of
  a two-tree pass lives in a side tree until it is merged.
* therefore the number worth reading is PER CLASS: 16g at 72.6 % in the main tree while 32g sits
  at 0 % there and 33.8 % in its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import reproof_coverage as C  # noqa: E402


def _proof(ver, mb=None):
    p = {"backend": {"triton": ver, "name": "cuda"}}
    if mb is not None:
        p["machine"] = {"device": {"memory_mb": mb}}
    return p


def _dir(tmp_path, entries):
    d = tmp_path / "nvidia" / "volta"
    d.mkdir(parents=True)
    (d / "k.fp32.json").write_text(json.dumps(
        {"format": "nbx-autotune-certified/1", "kernel": "k", "dtype": "fp32", "entries": entries}))
    return tmp_path


def test_an_entry_re_proven_on_every_class_it_carries_is_done(tmp_path):
    root = _dir(tmp_path, {"a": {"proof": _proof("3.8.0", 16384),
                                 "variants": {"32g": {"proof": _proof("3.8.0", 32768)}}}})
    t = C.coverage(root, "3.8.0")["total"]
    assert (t["done"], t["partly"], t["behind"]) == (1, 0, 0)


def test_one_class_re_proven_and_the_other_not_is_partly_and_says_so(tmp_path):
    """The live shape: the 16 GB half re-proven in the main tree, the 32 GB half still in its own."""
    root = _dir(tmp_path, {"a": {"proof": _proof("3.8.0", 16384),
                                 "variants": {"32g": {"proof": _proof("3.6.0", 32768)}}}})
    rec = C.coverage(root, "3.8.0")
    t = rec["total"]
    assert (t["done"], t["partly"], t["behind"]) == (0, 1, 0), "this may not read as done"
    assert rec["by_memory_class"]["16g"]["percent_done"] == 100.0
    assert rec["by_memory_class"]["32g"]["percent_done"] == 0.0


def test_nothing_re_proven_is_behind_and_no_proof_at_all_is_unproven(tmp_path):
    root = _dir(tmp_path, {"a": {"proof": _proof("3.6.0", 16384)},
                           "b": {"config": {"BLOCK_M": 32}}})
    t = C.coverage(root, "3.8.0")["total"]
    assert (t["done"], t["partly"], t["behind"], t["unproven"]) == (0, 0, 1, 1)


def test_a_proof_naming_no_memory_size_is_its_own_class_and_not_silently_counted(tmp_path):
    """The legacy unknown-card proofs — 1 168 of them in the live directory — serve no card until
    re-proven. Folding them into a real class would overstate that class's coverage."""
    root = _dir(tmp_path, {"a": {"proof": _proof("3.8.0")}})
    rec = C.coverage(root, "3.8.0")
    assert "?" in rec["by_memory_class"], rec["by_memory_class"]
    assert rec["by_memory_class"]["?"]["total"] == 1
    assert "16g" not in rec["by_memory_class"]


def test_the_generators_seen_are_reported_so_a_mixed_directory_is_visible(tmp_path):
    root = _dir(tmp_path, {"a": {"proof": _proof("3.8.0", 16384),
                                 "variants": {"32g": {"proof": _proof("3.6.0", 32768)}}}})
    assert C.coverage(root, "3.8.0")["generators_seen"] == {"3.6.0": 1, "3.8.0": 1}
