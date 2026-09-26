"""The regression matrix reserves host memory per cell; two cells over half the budget never overlap.

2026-09-26: three concurrent 30B-class cells held 180 GB of a 251 GB host (1.35-1.65x their
weights), memory pressure reached 33 % "full" and the gate running beside them timed out. The
matrix now reserves 1.7x a container's weights in a flock'd ledger against 55 % of the host.
On a runner that never refused a reservation this fails.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


def _child(out, need, q):
    q.put(R.reserve_host(Path(out), need))
    time.sleep(1.5)
    R.release_host(Path(out))


def test_two_cells_over_half_the_budget_never_overlap(tmp_path):
    need = int(R._host_bytes() * R.HOST_SHARE) // 2 + 1
    q = mp.Queue()
    procs = [mp.Process(target=_child, args=(str(tmp_path), need, q)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert sorted([q.get(), q.get()]) == [False, True]
    assert json.loads((tmp_path / "host_ledger.json").read_text()) == {}
