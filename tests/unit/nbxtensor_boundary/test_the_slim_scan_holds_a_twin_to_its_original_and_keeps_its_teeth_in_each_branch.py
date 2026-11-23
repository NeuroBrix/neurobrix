"""`tools/engine_slim_scan.py` reads the engine's two branches (RED_LINES.md lines 16-22): a twin is held to its
original, and the scan keeps its teeth inside each side. Proven on a repository written here, with a doubled brick
`prism` (`pytorch/prism/` and `triton/prism/`), one case per rule.

What would this file do if the scan were wrong?
  * a twin counted as a duplicate of its original, or listed beside its original's entry: RED (twin rule);
  * a real duplicate inside triton/ (two triton functions, one of them in a file that twins nothing), inside pytorch/,
    or in the rest of the engine no longer reported: RED (teeth);
  * a definition kept alive only by its twin naming it, or by the other side of its brick: RED (a dead one hidden);
  * a member named only through a branch-door call called dead, or the door resolved to one side only or to the wrong
    module: RED;
  * a member nobody names reported alive: RED;
  * a twin file grown past its original's code lines not shown in the twin-aware line count: RED;
  * the raw scan (`twin_aware=False`) no longer the scan of before the doubling: RED."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("engine_slim_scan", REPO / "tools" / "engine_slim_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _body(op: str) -> str:
    """A function body long enough for the duplicate rule; `op` makes bodies differ."""
    return ("    total = 0\n"
            "    for i, v in enumerate(values):\n"
            "        if v > limit:\n"
            f"            total += v {op} i\n"
            "        else:\n"
            "            total -= v\n"
            "    return total, len(values) - total\n")


SOLVER = (
    "def plan(values, limit):\n" + _body("*") + "\n"                 # the twin pair: one definition per branch
    "def door_only(x):\n    return x + 1\n\n"                         # named only through the door
    "def nobody():\n    return 1\n\n"                                 # named by nobody
    "def recursive_only(n):\n    return recursive_only(n - 1) if n else 0\n\n"   # names itself only
    "def used_by_triton_only(x):\n    return x * 3\n\n"               # named by the triton side only
    "def py_dup_a(values, limit):\n" + _body("+") + "\n"              # a real duplicate inside pytorch/
)
TRITON_EXTRA = "def tri_dup_b(values, limit):\n" + _body("-") + "\n"   # a twin's own definition its original lacks


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    _write(r, "site/nbxtensor/__init__.py", "")
    _write(r, "src/neurobrix/__init__.py", "")
    _write(r, "src/neurobrix/branch_door.py", "def names(brick, sub, *members):\n    return members\n")
    for side in ("pytorch", "triton"):
        _write(r, f"src/neurobrix/{side}/__init__.py", "")
        _write(r, f"src/neurobrix/{side}/prism/__init__.py", "from .solver import plan\n")
    _write(r, "src/neurobrix/pytorch/prism/solver.py", SOLVER)
    _write(r, "src/neurobrix/pytorch/prism/other.py", "def py_dup_b(values, limit):\n" + _body("+"))
    _write(r, "src/neurobrix/triton/prism/solver.py", SOLVER + TRITON_EXTRA + "\n\ndef grown():\n    a = 1\n    return a\n")
    _write(r, "src/neurobrix/triton/prism/extra.py", "def tri_dup_a(values, limit):\n" + _body("-"))
    _write(r, "src/neurobrix/triton/prism/user.py",
           "from neurobrix.triton.prism.solver import used_by_triton_only\nused_by_triton_only(1)\n")
    _write(r, "src/neurobrix/core.py",
           "def common_a(values, limit):\n" + _body("//") + "\n"
           "def common_b(values, limit):\n" + _body("//"))
    _write(r, "src/neurobrix/cli.py",
           "from neurobrix import branch_door as _branch_door\n\n"
           "def go():\n"
           "    plan = _branch_door.names('prism', '', 'plan')[0]\n"
           "    return _branch_door.names('prism', 'solver', 'door_only')[0](plan)\n")
    _write(r, "tools/use.py",
           "from neurobrix.cli import go\nfrom neurobrix.core import common_a, common_b\n"
           "go(), common_a, common_b, py_dup_a, py_dup_b, tri_dup_a, tri_dup_b, grown\n")
    return r


PY, TR = "src/neurobrix/pytorch/prism/solver.py", "src/neurobrix/triton/prism/solver.py"


def _scan(r: Path, twin_aware: bool = True):
    return {(k, p, s) for k, p, s, _ in S.scan(r, r / "site" / "nbxtensor", twin_aware=twin_aware)}


def test_a_twin_is_not_a_duplicate_of_its_original_and_is_never_listed_twice(tmp_path):
    r = _repo(tmp_path)
    got = _scan(r)
    assert ("duplicate", PY, "plan") not in got and ("duplicate", TR, "plan") not in got, sorted(got)
    # a finding the original carries covers its twin: one entry, on the original's path
    assert ("dead", PY, "nobody") in got and ("dead", TR, "nobody") not in got, sorted(got)
    assert not any(p == TR and s in ("py_dup_a", "recursive_only") for _, p, s in got), sorted(got)


def test_a_real_duplicate_inside_each_branch_and_in_the_rest_of_the_engine_is_still_reported(tmp_path):
    got = _scan(_repo(tmp_path))
    expected = {
        ("duplicate", "src/neurobrix/triton/prism/extra.py", "tri_dup_a"),     # inside triton/, a file twinning nothing
        ("duplicate", TR, "tri_dup_b"),                                         # inside triton/, a twin's own function
        ("duplicate", PY, "py_dup_a"),                                          # inside pytorch/
        ("duplicate", "src/neurobrix/pytorch/prism/other.py", "py_dup_b"),
        ("duplicate", "src/neurobrix/core.py", "common_a"),                     # the rest of the engine
        ("duplicate", "src/neurobrix/core.py", "common_b"),
    }
    assert not expected - got, sorted(expected - got)


def test_a_member_named_only_through_the_door_is_alive_and_one_nobody_names_is_dead(tmp_path):
    got = _scan(_repo(tmp_path))
    assert not any(s == "door_only" for k, _, s in got if k == "dead"), sorted(got)
    assert ("dead", PY, "nobody") in got
    # a definition its twin names (itself only) or the other side names is not used by that
    assert ("dead", PY, "recursive_only") in got
    assert ("dead", PY, "used_by_triton_only") in got and ("dead", TR, "used_by_triton_only") not in got, sorted(got)


def test_the_door_names_both_sides_and_follows_a_package_member_to_its_module(tmp_path):
    r = _repo(tmp_path)
    refs = S.door_references(ast.parse((r / "src/neurobrix/cli.py").read_text()))
    assert refs == {("neurobrix.pytorch.prism", "plan"), ("neurobrix.triton.prism", "plan"),
                    ("neurobrix.pytorch.prism.solver", "door_only"), ("neurobrix.triton.prism.solver", "door_only")}
    computed = ast.parse("from neurobrix import branch_door as _branch_door\nn = 'x'\n"
                         "_branch_door.names('prism', 'solver', n)\n_branch_door.module('prism', 'memory_budget')\n")
    assert S.door_references(computed) == {("neurobrix.pytorch.prism.memory_budget", ""),
                                           ("neurobrix.triton.prism.memory_budget", "")}
    trees = {p: ast.parse(p.read_text()) for p in S._py_files(r / "src")}
    reexp = S._package_reexports(trees, r, r / "site" / "nbxtensor")
    assert reexp[("neurobrix.triton.prism", "plan")] == ("neurobrix.triton.prism.solver", "plan")


def test_the_raw_scan_counts_twins_separately_and_hides_what_a_twin_names(tmp_path):
    r = _repo(tmp_path)
    raw, aware = _scan(r, twin_aware=False), _scan(r)
    assert ("duplicate", PY, "plan") in raw and ("duplicate", TR, "plan") in raw
    assert ("dead", TR, "nobody") in raw
    assert ("dead", PY, "recursive_only") not in raw          # its twin's self-call read as a use: the raw scan's blind spot
    assert len(raw) > len(aware)


def test_a_twin_grown_past_its_original_shows_in_the_twin_aware_line_count(tmp_path):
    r = _repo(tmp_path)
    aware, grew = S.engine_lines_twin_aware(r)
    assert [g[0] for g in grew] == [TR], grew
    _, base, n = grew[0]
    assert n > base
    twin_files = [p for p in S._py_files(r / "src/neurobrix/triton/prism")
                  if (r / "src/neurobrix/pytorch/prism" / p.relative_to(r / "src/neurobrix/triton/prism")).is_file()]
    spec = importlib.util.spec_from_file_location("brick_ratchet", REPO / "tools" / "brick_ratchet.py")
    bricks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bricks)
    twin_lines = sum(len(bricks.code_lines(p.read_text())) for p in twin_files)
    assert aware == S.engine_lines(r) - twin_lines + (n - base)
