"""The census table's write lock leaves no file in the package (2026-11-08: every table write left an empty
`<table>.jsonl.lock` beside the table, in src/neurobrix/config/census/, and the packaging test refused the package).
The lock is an flock on the table's directory (`kernels/census_table.locked`).

What this test does if the code is wrong: a sidecar lock file appears beside the table after a write — red (seen red on
the sidecar lock this replaced); two writers on one table do not exclude each other — red."""
import multiprocessing as mp
import time

from neurobrix.kernels import census_table as T


def test_a_write_leaves_only_the_table(tmp_path):
    path = T.table_path("v", "p", 16, root=tmp_path)
    row = {"model": "m", "container": "c", "mode": "triton", "rungs_mb": None, "ops": ["op::0"],
           "kernel": "k.kernel", "key": "(1, 'fp16')", "dtype": "fp16", "tool": "t"}
    T.replace_model(path, "m", [row])
    assert sorted(p.name for p in path.parent.iterdir()) == ["16g.jsonl"]


def _hold(path, held, done):
    with T.locked(path):
        held.set()
        time.sleep(0.5)
        done.set()          # still holding: a second writer that gets in before this point is not excluded


def test_two_writers_exclude_each_other(tmp_path):
    path = T.table_path("v", "p", 16, root=tmp_path)
    ctx = mp.get_context("fork")
    held, done = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold, args=(path, held, done))
    p.start()
    assert held.wait(10)
    with T.locked(path):
        assert done.is_set(), "a second writer took the lock while the first held it"
    p.join(10)
