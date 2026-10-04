"""The gate harness looks for its containers where the ENGINE looks for them.

The engine resolves its cache through one door, `neurobrix.core.paths.cache_dir()`: `NEUROBRIX_CACHE`, then
`~/.neurobrix/paths.json`, then the default. `tools/regression_matrix.py` had the default written as a literal
(`~/.neurobrix/ca` + `che`, the word split so a search would not see it), so it refused any container the engine
would have found elsewhere while the cells it launches DO read the door. On the Mac, 2026-10-04 08:55: a 57 GiB
container that does not fit the local disk was to run in place from the NAS mount (`NEUROBRIX_CACHE` set to the
mount for that cell), and the harness answered in 5 s "REFUSED: --models: Qwen3-Coder-30B-A3B-Instruct — no
container of that name in the cache /Users/hocine/.neurobrix/cache".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _harness(cache: Path, code: str) -> str:
    env = dict(os.environ, NEUROBRIX_CACHE=str(cache))
    r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(REPO / 'tools')!r}); "
                                              f"import regression_matrix as R; {code}"],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip().splitlines()[-1]


def test_the_harness_cache_is_the_engines_door(tmp_path):
    assert _harness(tmp_path, "print(R.CACHE)") == str(tmp_path)


def test_a_container_under_the_configured_cache_is_a_cell_of_the_matrix(tmp_path):
    (tmp_path / "a-model").mkdir()
    (tmp_path / "a-model" / "manifest.json").write_text(json.dumps({"name": "a-model"}))
    cells = _harness(tmp_path, "print(sorted(m for m, _ in R.full_matrix()))")
    assert cells == "['a-model', 'a-model', 'a-model']", cells       # one per mode, from the configured cache
