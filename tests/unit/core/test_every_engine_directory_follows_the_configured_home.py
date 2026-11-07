"""Every directory the engine keeps under its home follows the configured home (`NEUROBRIX_HOME`): the Mac's new-user
run (2026-11-07, a fresh NEUROBRIX_HOME) found the replay cache, the license acceptances and the hardware profiles
written to ~/.neurobrix — module constants built as `Path.home() / ".neurobrix" / ...`. In a child process, so the
constants are computed at import with the variable set, as a user's process does. No card.

What this test would do if the code were wrong: any of the directories below outside the configured home (red)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.unit.child_env import child_env

REPO = Path(__file__).resolve().parents[3]

_DRIVER = r"""
import json, os
from neurobrix.cli.commands import registry
from neurobrix.triton import replay
from neurobrix.core.prism import loader
from neurobrix.core.dtype import calibration
from neurobrix.serving import protocol
from neurobrix.core.runtime import render_checkpoint
from nbxtensor.ops import autotune_cache
print(json.dumps({"acceptances": str(registry._ACCEPTANCES_FILE), "replay_sizes": replay._SIZE_CACHE_DIR,
                  "hardware": str(loader.GENERATED_DIR), "calibration": str(calibration.STORE_ROOT),
                  "daemon": str(protocol.DAEMON_DIR), "checkpoints": str(render_checkpoint._DEFAULT_DIR),
                  "library_replay_cache": autotune_cache._dir()}))
"""


def test_every_directory_is_under_the_configured_home(tmp_path):
    home = tmp_path / "engine_home"
    home.mkdir()
    env = child_env(PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO / 'nbxtensor' / 'src'}{os.pathsep}{REPO}",
                    PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", NEUROBRIX_HOME=str(home))
    env.pop("NEUROBRIX_REPLAY_CACHE", None)
    out = subprocess.run([sys.executable, "-c", _DRIVER], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    dirs = json.loads(out.stdout.strip().splitlines()[-1])
    outside = {k: v for k, v in dirs.items() if not str(v).startswith(str(home))}
    assert not outside, outside
