"""The Mac's chains (tools/mac/: the matrix chain, its precheck, the watchers, the progress line, the benchmark regen)
take every path from atelier.json (`docs/internal/_atelier.md` §4) — no /Users/..., $HOME/..., ~/..., /tmp/... or
/System/... written in their code — and a writer refuses by name an output outside the writable roots. What this test
does if the code were wrong: a hard-coded path in any tools/mac file, or an inventory written outside the roots -> red."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
HARD = re.compile(r"(/Users/|\$HOME/|~/|(?<![\w$])/tmp/|/System/Volumes|/private/)")


def _code_lines(p: Path):
    for i, l in enumerate(p.read_text().splitlines(), 1):
        s = l.strip()
        if s.startswith("#") or s.startswith('"""') or not s:
            continue
        yield i, l


@pytest.mark.parametrize("p", sorted(p for p in (TOOLS / "mac").glob("*") if p.is_file()), ids=lambda p: p.name)
def test_no_hard_coded_path(p):
    bad = [f"{p.name}:{i}: {l.strip()[:100]}" for i, l in _code_lines(p) if HARD.search(l)]
    assert not bad, bad


def test_the_inventory_refuses_an_output_outside_the_roots(tmp_path):
    cfg = tmp_path / "atelier.json"
    cfg.write_text(json.dumps({"campaigns": str(tmp_path / "c"), "registres": str(tmp_path / "r"),
                               "tmp": str(tmp_path / "t"), "_writable": ["tmp"]}))
    (tmp_path / "r" / "runs").mkdir(parents=True)
    (tmp_path / "c").mkdir()
    env = {"NBX_ATELIER_JSON": str(cfg), "PATH": "/usr/bin:/bin"}
    ok = subprocess.run([sys.executable, str(TOOLS / "mac" / "artefact_inventory.py"), str(tmp_path / "t" / "inv.tsv")],
                        env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr[-400:]
    bad = subprocess.run([sys.executable, str(TOOLS / "mac" / "artefact_inventory.py"), str(tmp_path / "elsewhere.tsv")],
                         env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "OutsideTheAtelier" in bad.stderr
