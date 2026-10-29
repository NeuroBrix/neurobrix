"""`tools/engine_slim_scan.py` finds each of its four kinds, and nothing that is in use (Hocine 2026-10-29 22:39: the
engine imports nbxtensor and keeps only its own job). Proven on a repository written here, one case per rule.

What would this file do if the scan were wrong? A rule that stopped matching drops its finding (RED); a definition
called from another file reported dead, or a function that computes something reported a wrapper (RED)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("engine_slim_scan", REPO / "tools" / "engine_slim_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_BODY = (
    "    total = 0\n"
    "    for i, v in enumerate(values):\n"
    "        if v > limit:\n"
    "            total += v * i\n"
    "        else:\n"
    "            total -= v\n"
    "    return total, len(values) - total\n")

A = (
    "from nbxtensor.shape import numel\n"
    "alias_numel = numel\n\n"
    "def used(x):\n    return helper(x)\n\n"
    "def scaled(x):\n    return helper(x * 2, 4)\n\n"
    "def helper(values, limit=3):\n" + _BODY + "\n"
    "def dead_one():\n    return 1\n\n"
    "def recursive(n):\n    return recursive(n - 1) if n else 0\n\n"
    "def _itemsize(d):\n    return {'float16': 2, 'float32': 4, 'bfloat16': 2, 'int8': 1}[d]\n\n"
    "class Base:\n    def run(self):\n        return 0\n\n"
    "class Renamed(Base):\n    pass\n\n"
    "def compute(a, flag=False):\n    return a * 2 if not flag else a\n\n"
    "def caller():\n    return used(1) + scaled(1) + compute(2) + _itemsize('float16')\n")
B = (
    "from neurobrix.a import numel, caller, Renamed\n\n"
    "def engine_twin(values, limit):\n" + _BODY + "\n"
    "def entry():\n    return caller(), numel((1,)), Renamed().run()\n")


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    _write(r, "nbxtensor/src/nbxtensor/__init__.py", "")
    _write(r, "nbxtensor/src/nbxtensor/shape/__init__.py",
           "def numel(shape):\n    n = 1\n    for d in shape:\n        n *= d\n    return n\n")
    _write(r, "nbxtensor/src/nbxtensor/dtype/__init__.py", "def itemsize(name):\n    return _SIZES[name]\n")
    _write(r, "nbxtensor/src/nbxtensor/ops/__init__.py", "")
    _write(r, "nbxtensor/src/nbxtensor/ops/kernels.py", "def library_twin(values, limit):\n" + _BODY)
    _write(r, "src/neurobrix/__init__.py", "")
    _write(r, "src/neurobrix/a.py", A)
    _write(r, "src/neurobrix/b.py", B)
    _write(r, "tools/use.py", "from neurobrix.b import entry, engine_twin\nentry()\nengine_twin([1], 0)\n")
    return r


def test_each_kind_is_found_and_nothing_in_use_is(tmp_path):
    got = {(k, p, s) for k, p, s, _ in S.scan(_repo(tmp_path))}
    a, b = "src/neurobrix/a.py", "src/neurobrix/b.py"
    expected = {
        ("dead", a, "dead_one"), ("dead", a, "recursive"),            # a self-call is no use
        ("wrapper", a, "used"),                                        # hands x to helper, nothing else
        ("wrapper", a, "compute(flag)"),                               # every call site leaves the default
        ("wrapper", a, "alias_numel"),                                 # an alias of a library name
        ("wrapper", a, "Renamed"),                                     # renames its base
        ("wrapper", b, "from neurobrix.a import numel"),               # a library name through an engine module
        ("duplicate", b, "engine_twin"),                               # the library's body, renamed
        ("fact", a, "_itemsize"),                                      # a library fact by its name
    }
    missing = expected - got
    assert not missing, sorted(missing)
    assert any(k == "fact" and p == a and s.startswith("dtype size table") for k, p, s in got), sorted(got)
    in_use = {"helper", "caller", "entry", "engine_twin", "used", "compute", "_itemsize", "Base", "Base.run"}
    wrongly_dead = {s for k, p, s in got if k == "dead" and s in in_use}
    assert not wrongly_dead, wrongly_dead
    # one call that computes its arguments is no wrapper
    assert not {s for k, p, s in got if k == "wrapper" and s in ("helper", "caller", "entry", "scaled")}, sorted(got)


def test_the_line_count_is_the_engines_python_lines(tmp_path):
    r = _repo(tmp_path)
    expect = sum(len(p.read_text().splitlines()) for p in (r / "src" / "neurobrix").rglob("*.py"))
    assert S.engine_lines(r) == expect > 0


def test_a_file_it_cannot_read_is_refused_by_name(tmp_path):
    import pytest
    r = _repo(tmp_path)
    _write(r, "src/neurobrix/broken.py", "def f(:\n")
    with pytest.raises(S.ScanRefused, match="broken.py"):
        S.scan(r)
