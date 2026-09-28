"""Two certifiers write one kernel's certified file at once — one per memory class of a profile,
pinned to two cards. Each writes after every key it proves; neither may erase the other's proofs.

The first repair (2026-09-21) merged by KEY and was tested only on DISJOINT keys, so it passed
while a key BOTH classes hold — one entry, a primary and its class variants — was overwritten
whole by whichever writer's copy was older (2026-09-28/29: the 16 GB GEMM pass erased the 32 GB
card's variant of eleven shared baddbmm keys for two hours, and the 32 GB card re-certified them
on every pass). The writer now places only what it just proved into the file as it stands.

What each test would do if the write were wrong: a writer that writes back its own earlier copy
of the file drops the other class's variant of the shared key (the shared-key test fails: seen
red on the key-level merge); a writer that dropped disjoint keys fails the second test.
"""
from __future__ import annotations

import json

from neurobrix.kernels import autotune_certified as C
from neurobrix.kernels import autotune_certify as CF


def _cert(memory_mb: int, block_m: int) -> dict:
    return {"config": {"kwargs": {"BLOCK_M": block_m}, "num_warps": 4, "num_stages": 3},
            "proof": {"machine": {"device": {"memory_mb": memory_mb}}, "deviation": 1e-7},
            "excluded": []}


def _write(path, fresh, cache):
    CF._write_file(path, "nvidia", "volta", "q.baddbmm_kernel", "fp32", fresh, cache=cache)


def test_a_key_both_classes_hold_keeps_both_classes(tmp_path):
    path = tmp_path / "baddbmm_kernel.fp32.json"
    a, b = {}, {}                                   # each writer's view, as the certify loop keeps it
    _write(path, {"k": _cert(16384, 32)}, a)        # the 16 GB card proves k
    stale_a = json.loads(json.dumps(a))             # the 16 GB pass goes on with this view
    _write(path, {"k": _cert(32768, 64)}, b)        # the 32 GB card proves k: its variant
    a = stale_a
    _write(path, {"k2": _cert(16384, 16)}, a)       # the 16 GB card's NEXT key, from its older view
    on_disk = json.loads(path.read_text())["entries"]
    assert C.entry_for_memory_class(on_disk["k"], 32) is not None, "the 32 GB variant was erased"
    assert C.entry_for_memory_class(on_disk["k"], 32)["config"]["kwargs"] == {"BLOCK_M": 64}
    assert C.entry_for_memory_class(on_disk["k"], 16)["config"]["kwargs"] == {"BLOCK_M": 32}
    assert set(on_disk) == {"k", "k2"}
    assert a == on_disk, "the writer's view is refreshed to what it wrote"
    assert not list(tmp_path.glob("*.tmp"))


def test_interleaved_writers_leave_every_entry_in_the_file(tmp_path):
    path = tmp_path / "matmul_kernel.fp32.json"
    a, b = {}, {}
    _write(path, {"k1": _cert(16384, 32)}, a)
    _write(path, {"k2": _cert(32768, 32)}, b)       # b never saw a
    _write(path, {"k3": _cert(16384, 64)}, a)
    _write(path, {"k4": _cert(32768, 64)}, b)
    on_disk = json.loads(path.read_text())["entries"]
    assert set(on_disk) == {"k1", "k2", "k3", "k4"}, sorted(on_disk)
    assert not list(tmp_path.glob("*.tmp"))
