"""Two first runs of one model at the same moment download its processor ONCE: the second waits for the first's
lock, then finds the verified file and says nothing.

Red without the change: there is no `neurobrix processor` command (argparse exits 2)."""
import os
import subprocess
import sys

from tests.unit.processors import toy
from tests.unit.processors.conftest import REPO, env_of


def test_two_runs_at_once_ask_the_host_once_and_one_of_them_speaks(places):
    data = toy.artifact(padding=400_000)
    toy.install_model(places["NEUROBRIX_CACHE"], "toy", [toy.declaration(data)])
    hub = toy.Host({toy.FILE: data}, rate=200_000)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "nbxtensor" / "src")]),
               **env_of(places, hub.url, toy.dead_url()))
    argv = [sys.executable, "-m", "neurobrix", "processor", "fetch", toy.hub_name("toy")]
    try:
        runs = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                for _ in range(2)]
        outs = [r.communicate(timeout=120) for r in runs]
    finally:
        hub.close()
    assert [r.returncode for r in runs] == [0, 0], outs
    assert hub.gets == {toy.FILE: 1}
    assert sorted(len(err.splitlines()) for _, err in outs) == [0, 2], outs
    assert (places["NEUROBRIX_CACHE"] / "processors" / toy.FILE).read_bytes() == data
