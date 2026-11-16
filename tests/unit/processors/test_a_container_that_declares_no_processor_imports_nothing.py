"""A container without the `processors` key costs nothing: in a fresh interpreter, the run door's module loaded and
a container without the key opened, the processors' call returns None, its door (store.py: the network, the hash,
the lock) is never imported and the HTTP GET (`requests.get`, replaced by a counter) is never called. Both doors
that load a model (the run command and the serving daemon's load) make the call.

Red without the change: `neurobrix.processors` does not exist (the child fails on the import), and neither door
calls it."""
import ast
import os
import subprocess
import sys
from pathlib import Path

from tests.unit.processors import toy

REPO = Path(__file__).resolve().parents[3]

CHILD = r"""
import sys
from pathlib import Path
import requests
import neurobrix.cli.commands.run
from neurobrix.nbx import NBXContainer
from neurobrix.processors import open_declared
container = NBXContainer.load(sys.argv[1])
asked = []
requests.get = lambda *a, **k: asked.append(a)
got = open_declared(container.get_manifest(), Path(sys.argv[1]), [c.name for c in container.get_neural_components()])
assert got is None, got
assert "neurobrix.processors.store" not in sys.modules, "the processors' door was imported"
assert not asked, "the download was asked"
print("NOTHING")
"""


def test_no_declaration_imports_no_door_and_asks_no_download(places):
    entry = toy.install_model(places["NEUROBRIX_CACHE"], "plain")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "nbxtensor" / "src")]))
    r = subprocess.run([sys.executable, "-c", CHILD, str(entry)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "NOTHING", r.stderr[-3000:]


def _calls(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_the_run_door_and_the_daemon_load_both_open_the_declared_processors():
    for door in ("src/neurobrix/cli/commands/run.py", "src/neurobrix/serving/engine.py"):
        assert "open_declared" in _calls(REPO / door), f"{door} does not open the declared processors"
