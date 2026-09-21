"""Two certifiers write one kernel's certified file at once — one per memory class of a profile,
pinned to two cards (the 16 GB and 32 GB matrix rounds, 2026-09-21). Each rewrote the whole
file from its own entries after every key, dropping the other's proofs, and their fixed
`.json.tmp` collided (rc 1 on `os.replace`). The write is a locked merge through a temp file
only this process names.

Shapes: two writers, two disjoint entries each, interleaved; the file must end with all four.
"""
from __future__ import annotations

import json

from neurobrix.kernels import autotune_certify as CF


def test_interleaved_writers_leave_every_entry_in_the_file(tmp_path):
    path = tmp_path / "matmul_kernel.fp32.json"
    a = {"k1": {"proof": "class 16 GB"}}
    b = {"k2": {"proof": "class 32 GB"}}
    CF._write_file(path, "nvidia", "volta", "q.matmul_kernel", "fp32", a)
    CF._write_file(path, "nvidia", "volta", "q.matmul_kernel", "fp32", b)      # b never saw a
    a["k3"] = {"proof": "class 16 GB, later"}
    CF._write_file(path, "nvidia", "volta", "q.matmul_kernel", "fp32", a)      # a's next write carries the union
    b["k4"] = {"proof": "class 32 GB, later"}
    CF._write_file(path, "nvidia", "volta", "q.matmul_kernel", "fp32", b)
    on_disk = json.loads(path.read_text())["entries"]
    assert set(on_disk) == {"k1", "k2", "k3", "k4"}, sorted(on_disk)
    assert not list(tmp_path.glob("*.tmp"))
