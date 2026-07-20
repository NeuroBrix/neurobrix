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
    - {"neurobrix", "__main__"}
)
engine_leaks = sorted(
    m for m in added
    if m.startswith("neurobrix.")
    and not m.startswith("neurobrix.agent")
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
