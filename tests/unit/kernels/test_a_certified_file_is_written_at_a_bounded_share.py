"""A certified kernel file is rewritten at a BOUNDED SHARE of the certifier's time, and nothing proven
is ever left unwritten.

The file is 13-21 MB of JSON; rewriting it after every key took 36.7 % of a 16 GB conv pass (py-spy,
2026-09-29) while the card sat idle. `_BoundedWriter` writes a file when the time since its last
write reaches that write's duration over the share, and `flush_all` (in `certify`'s finally) writes
the rest.

What each test would do if the writer were wrong: a writer that wrote after every key makes 5 writes
for 5 keys where the bound allows 2 (seen red with `add` flushing unconditionally); a writer that lost
the tail leaves keys out of the file (the second test fails).
"""
from __future__ import annotations

import json

from neurobrix.kernels import autotune_certify as CF


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _cert(memory_mb: int, n: int) -> dict:
    return {"config": {"kwargs": {"BLOCK_M": n}, "num_warps": 4, "num_stages": 3},
            "proof": {"machine": {"device": {"memory_mb": memory_mb}}, "deviation": 1e-7}, "excluded": []}


def test_writes_are_bounded_by_their_own_cost(tmp_path):
    clock, writes = _Clock(), []

    def fake_write(path, vendor, profile, qual, dtype, fresh, cache=None):
        writes.append(sorted(fresh))
        clock.t += 1.0                      # a write costs one second

    w = CF._BoundedWriter("nvidia", "volta", share=0.10, write=fake_write, clock=clock)
    p = tmp_path / "conv.json"
    for i, t in enumerate([0.0, 2.0, 4.0, 6.0, 12.0]):   # keys certified at these times
        clock.t = max(clock.t, t)
        w.add(p, "q.conv", "fp16", {}, f"k{i}", _cert(16384, i))
    w.flush_all()
    # k0 at once (first write, 1 s); k1-k3 within 10 s of its end: held; k4 at 12 s >= 1 + 10: written
    assert writes == [["k0"], ["k1", "k2", "k3", "k4"]], writes


def test_nothing_proven_is_left_out_of_the_file(tmp_path):
    w = CF._BoundedWriter("nvidia", "volta", share=1e-9)          # never due after the first write
    p = tmp_path / "matmul_kernel.fp32.json"
    cache: dict = {}
    for i in range(6):
        w.add(p, "q.matmul_kernel", "fp32", cache, f"k{i}", _cert(16384, i))
    w.flush_all()
    on_disk = json.loads(p.read_text())["entries"]
    assert set(on_disk) == {f"k{i}" for i in range(6)}
    assert not list(tmp_path.glob("*.tmp"))
