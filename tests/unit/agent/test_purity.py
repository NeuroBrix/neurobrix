"""The mechanical purity gate (maintainer decision, P-AGENTIC).

`neurobrix.agent` is PURE STDLIB: importing it in a fresh process must
pull in NO torch, NO third-party module, and NO engine subpackage. The
check measures the import DELTA in a clean subprocess (clean-room style,
proven on the audio family) — same closure rank as the anti-reg battery.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[3] / "src")

_PROBE = """
import json, sys
before = set(sys.modules)
import neurobrix.agent  # the import under test
added = set(sys.modules) - before
stdlib = set(getattr(sys, "stdlib_module_names", ()))
third_party = sorted(
    {m.split(".")[0] for m in added}
    - stdlib
    - {"neurobrix", "__main__", "nbxtensor"}
)
# the engine installs its tensor library's host services at import (neurobrix/_nbxtensor_host.py): the package and
# `nbxtensor.host` (stdlib only) are loaded, and nothing else of the library
library = sorted(m for m in added if m == "nbxtensor" or m.startswith("nbxtensor."))
# and the library's settings an application sets at start, in modules of the standard library only (the launch
# observer, the autotune's store and miss explainer) — never the autotune, a kernel or a backend
_STDLIB_ONLY = ("nbxtensor", "nbxtensor.host", "nbxtensor.ops", "nbxtensor.ops.launch_observer",
                "nbxtensor.ops.autotune_settings")
third_party += [m for m in library if m not in _STDLIB_ONLY]
engine_leaks = sorted(
    m for m in added
    if m.startswith("neurobrix.")
    and not m.startswith("neurobrix.agent")
    and m != "neurobrix._nbxtensor_host"   # the package's own init: the tensor library's host services (stdlib only)
)
print(json.dumps({"third_party": third_party, "engine_leaks": engine_leaks}))
"""


def test_agent_package_is_stdlib_pure():
    env = dict(os.environ, PYTHONPATH=_SRC)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["third_party"] == [], (
        f"neurobrix.agent pulled third-party modules: {report['third_party']}"
    )
    assert report["engine_leaks"] == [], (
        f"neurobrix.agent pulled engine subpackages: {report['engine_leaks']}"
    )
