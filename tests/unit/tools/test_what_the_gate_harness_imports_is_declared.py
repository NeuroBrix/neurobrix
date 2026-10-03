"""Every third-party module the gate harness imports is declared in pyproject.toml.

The Mac, 2026-10-03: tools/regression_matrix.py imports psutil (a cell's process tree, the runner's
children) and pyproject declared it nowhere — the served venv lacked it and the harness's own test
failed on ModuleNotFoundError there. It is declared in the `dev` extra (the harness is dev tooling);
the runtime engine reads the host through core.host_memory and imports no psutil.

What would this file do if the code were wrong? psutil removed from the `dev` extra -> the first
test, RED; an engine module importing psutil again -> the second, RED.
"""
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "tools" / "regression_matrix.py"


def _declared():
    try:
        import tomllib as T
    except ImportError:                       # Python 3.10
        import tomli as T
    project = T.loads((REPO / "pyproject.toml").read_text())["project"]
    specs = project["dependencies"] + sum(project.get("optional-dependencies", {}).values(), [])
    return {re.split(r"[<>=\[ ;]", s)[0].lower().replace("-", "_") for s in specs}


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            yield from (a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            yield n.module.split(".")[0]


def test_every_third_party_module_of_the_harness_is_declared():
    tools = {p.stem for p in (REPO / "tools").glob("*.py")}
    third = {m for m in _imports(HARNESS)
             if m not in sys.stdlib_module_names and m not in tools and m != "neurobrix"}
    alias = {"PIL": "pillow", "yaml": "pyyaml"}
    missing = sorted(m for m in third if alias.get(m, m).lower() not in _declared())
    assert not missing, f"imported by the gate harness, declared nowhere: {missing}"


def test_the_runtime_engine_imports_no_psutil():
    hits = [str(p.relative_to(REPO)) for p in (REPO / "src" / "neurobrix").rglob("*.py")
            if "triton_kernels_ref" not in p.parts and "autotune_certify.py" != p.name
            and "psutil" in set(_imports(p))]
    assert not hits, hits
