"""The tools under test are scripts: run as `python tools/<tool>.py`, their own directory is first on `sys.path`, and
they import their neighbours by name (`import atelier`, the workshop's roots). A test loads a tool by its file
(`importlib.util.spec_from_file_location`), which does not — so the tools' directory is put where the script's run puts
it. (Since the workshop's roots became `tools/atelier.py`, 2026-10-22, every tool test that loaded such a tool failed
with `No module named 'atelier'`: about 60 cells of the CPU suite, red on base.)"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[3] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
