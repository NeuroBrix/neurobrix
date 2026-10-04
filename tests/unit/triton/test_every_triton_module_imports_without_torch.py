"""Every module of the Triton package imports, with torch blocked — none is dead on arrival.

`triton/executor.py` imported a `load_safetensors` that `triton/weight_loader.py` no longer had: the
module could not be imported at all, nothing imported it, and nothing noticed. It was a third weight
loader reading shard files with no index check — the blindness behind the Allegro None of
2026-10-04 — kept alive only by not being imported. It is removed (with `triton/constants.py`, which
only it used), and the package is walked here so the next module that cannot import is named.

The R33 orchestrator gate lists the SHARED modules by name; this walks the whole Triton package, so a
new module joins the proof without anyone adding it. Torch is blocked by the same tool
(`tools/r33_import_without_torch.py`), so a module that pulls torch at import fails here too.
Seen failing on the tree before the removal: `FAIL neurobrix.triton.executor <- executor.py:15:
ImportError: cannot import name 'load_safetensors'` (54/55).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
TOOL = REPO / "tools" / "r33_import_without_torch.py"


def _modules(root: Path, package: str) -> list:
    out = []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root.parent).with_suffix("")
        parts = list(rel.parts)
        if "__pycache__" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out.append(".".join(["neurobrix"] + parts) if package == "neurobrix" else ".".join(parts))
    return out


def _run(names, extra_path=None):
    env = {"PYTHONPATH": str(SRC) + (f":{extra_path}" if extra_path else ""), "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home()), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, str(TOOL), *names], capture_output=True, text=True, env=env)


def test_every_triton_module_imports_without_torch():
    names = _modules(SRC / "neurobrix" / "triton", "neurobrix")
    assert "neurobrix.triton.weight_loader" in names and len(names) > 20, names
    proc = _run(names)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    assert f"{len(names)}/{len(names)} imported without torch" in proc.stdout, proc.stdout


def test_the_walk_is_seen_failing_on_a_module_that_cannot_import(tmp_path):
    pkg = tmp_path / "r33walk"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ok.py").write_text("X = 1\n")
    (pkg / "dead.py").write_text(textwrap.dedent("""
        from .ok import load_safetensors   # a name that does not exist
    """))
    names = _modules(pkg, "r33walk")
    assert sorted(names) == ["r33walk", "r33walk.dead", "r33walk.ok"], names
    proc = _run(names, extra_path=tmp_path)
    assert proc.returncode == 1, proc.stdout
    assert "FAIL r33walk.dead" in proc.stdout, proc.stdout
