"""The scope probe calls every changed function with opaque arguments; a function whose body EXITS (a CLI command:
`sys.exit(1)` when no daemon runs) is a result like a raise — uncaught, SystemExit ended the whole probe and every lot
whose diff reached the CLI had its scope refused (the Mac's pointer slots, 2026-10-20 19:43 and 19:46).
What this test would do if the probe caught Exception only: SystemExit escapes -> red."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_scope_probe as P  # noqa: E402


def _exits(_a):
    sys.exit(1)


def test_an_exiting_function_is_recorded_as_raised():
    rack = SimpleNamespace(root="/", lines=None)
    out = P._call_const(rack, _exits, 1, False, "none")
    assert out["raised"].startswith("SystemExit")
