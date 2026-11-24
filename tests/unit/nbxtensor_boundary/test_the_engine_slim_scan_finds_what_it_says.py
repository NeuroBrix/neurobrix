"""`tools/engine_slim_scan.py` finds each of its four kinds, and nothing that is in use (Hocine Benkelaya 2026-10-29 22:39: the
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
    _write(r, "site/nbxtensor/__init__.py", "")
    _write(r, "site/nbxtensor/shape/__init__.py",
           "def numel(shape):\n    n = 1\n    for d in shape:\n        n *= d\n    return n\n")
    _write(r, "site/nbxtensor/dtype/__init__.py", "def itemsize(name):\n    return _SIZES[name]\n")
    _write(r, "site/nbxtensor/ops/__init__.py", "")
    _write(r, "site/nbxtensor/ops/kernels.py", "def library_twin(values, limit):\n" + _BODY)
    _write(r, "src/neurobrix/__init__.py", "")
    _write(r, "src/neurobrix/a.py", A)
    _write(r, "src/neurobrix/b.py", B)
    _write(r, "tools/use.py", "from neurobrix.b import entry, engine_twin\nentry()\nengine_twin([1], 0)\n")
    return r


def test_each_kind_is_found_and_nothing_in_use_is(tmp_path):
    r = _repo(tmp_path)
    got = {(k, p, s) for k, p, s, _ in S.scan(r, r / "site" / "nbxtensor")}
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


def test_the_line_count_is_the_engines_code_lines(tmp_path):
    """Code lines — no docstring, comment-only or blank line (the brick ratchet's count): a docstring or a reason
    written in a comment never reads as growth. Red when every line of text is counted."""
    r = _repo(tmp_path)
    before = S.engine_lines(r)
    text_lines = sum(len(p.read_text().splitlines()) for p in (r / "src" / "neurobrix").rglob("*.py"))
    assert 0 < before <= text_lines
    _write(r, "src/neurobrix/counted.py", '"""A module\nof two docstring lines."""\n\n# a reason\nX = 1  # kept\n\n\n'
                                           'def f():\n    """Says."""\n    return (\n        X)\n')
    assert S.engine_lines(r) == before + 4          # X = 1, def f, return (, X)


def test_a_file_it_cannot_read_is_refused_by_name(tmp_path):
    import pytest
    r = _repo(tmp_path)
    _write(r, "src/neurobrix/broken.py", "def f(:\n")
    with pytest.raises(S.ScanRefused, match="broken.py"):
        S.scan(r)


# ─── CALLER PENDING (ruling 24/11 11:24): a marker in the code, never a line written by hand ───

_MARK = "# CALLER PENDING: 1374 {date}\n"
_DAY = S.datetime.date(2026, 11, 24)


def _pending_repo(tmp_path: Path, marked: str, called: bool = False, date: str = "2026-11-24") -> Path:
    r = tmp_path / "repo"
    _write(r, "site/nbxtensor/__init__.py", "")
    _write(r, "src/neurobrix/__init__.py", "")
    _write(r, "src/neurobrix/c.py",
           "def plain():\n    return 1\n\n" + marked.format(date=date) + "def later(x):\n    return x + 1\n")
    _write(r, "tools/use.py", "from neurobrix.c import later\nlater(1)\n" if called else "x = 1\n")
    return r


def _kinds(r: Path, today=_DAY):
    return {(k, s) for k, _, s, _ in S.scan(r, r / "site" / "nbxtensor", today=today)}


def test_a_marked_definition_with_no_caller_is_pending_not_dead(tmp_path):
    r = _pending_repo(tmp_path, _MARK)
    got = _kinds(r)
    assert ("dead", "plain") in got and not {f for f in got if f[1] == "later"}, got
    assert S.pending(r, r / "site" / "nbxtensor", _DAY) == {
        ("src/neurobrix/c.py", "later"): ("1374", "2026-11-24")}


def test_without_the_marker_it_is_dead_as_before(tmp_path):
    r = _pending_repo(tmp_path, "# an ordinary comment\n")
    assert ("dead", "later") in _kinds(r)
    assert S.pending(r, r / "site" / "nbxtensor", _DAY) == {}


def test_the_release_gate_is_red_while_an_entry_stands(tmp_path, capsys):
    r = _pending_repo(tmp_path, _MARK)
    assert S.release_gate(r, r / "site" / "nbxtensor", _DAY) == 1
    assert "later" in capsys.readouterr().out
    r2 = _pending_repo(tmp_path / "two", "")
    assert S.release_gate(r2, r2 / "site" / "nbxtensor", _DAY) == 0


def test_a_marker_left_behind_by_its_caller_is_a_finding(tmp_path):
    r = _pending_repo(tmp_path, _MARK, called=True)
    assert ("pending", "later") in _kinds(r)
    assert S.pending(r, r / "site" / "nbxtensor", _DAY) == {}


def test_a_marker_past_its_days_is_a_finding(tmp_path):
    r = _pending_repo(tmp_path, _MARK, date="2026-11-01")
    assert ("pending", "later") not in _kinds(r, today=S.datetime.date(2026, 11, 15))      # day 14: still pending
    assert ("pending", "later") in _kinds(r, today=S.datetime.date(2026, 11, 16))          # day 15: stale
    assert S.pending(r, r / "site" / "nbxtensor", S.datetime.date(2026, 11, 16)) == {}


def test_the_list_section_is_written_from_the_markers(tmp_path):
    r = _pending_repo(tmp_path, _MARK)
    lst = tmp_path / "list.tsv"
    lst.write_text("# findings: 9\n")
    lib = tmp_path / "installed" / "nbxtensor"                 # the list is written against a library outside the tree
    _write(lib.parent, "nbxtensor/__init__.py", "__version__ = '0.0'\n")
    S.write_list(r, lib, lst, today=_DAY)
    assert S.read_pending(lst) == {("src/neurobrix/c.py", "later"): ("1374", "2026-11-24")}
    assert all(k != "pending" for k, _, _ in S.read_list(lst)[1])          # an entry is no finding


def test_a_hand_in_that_is_not_a_card_id_is_refused(tmp_path):
    for bad in ("# CALLER PENDING: soon {date}\n", "# CALLER PENDING: 0000 {date}\n", "# CALLER PENDING: 1374\n"):
        r = _pending_repo(tmp_path / str(abs(hash(bad))), bad)
        try:
            S.scan(r, r / "site" / "nbxtensor", today=_DAY)
        except S.ScanRefused as e:
            assert "CALLER PENDING" in str(e) and "c.py" in str(e), e
        else:
            raise AssertionError(f"{bad!r} was read as a marker")


def test_a_changed_date_is_a_finding(tmp_path):
    r = _pending_repo(tmp_path, _MARK, date="2026-11-20")
    first = {("src/neurobrix/c.py", "later"): ("1374", "2026-11-10")}          # the list recorded it on 11-10
    got = {(k, s) for k, _, s, _ in S.scan(r, r / "site" / "nbxtensor", today=_DAY, recorded=first)}
    assert ("pending", "later") in got
    assert S.pending(r, r / "site" / "nbxtensor", _DAY, recorded=first) == {}
    same = {("src/neurobrix/c.py", "later"): ("1374", "2026-11-20")}
    assert S.pending(r, r / "site" / "nbxtensor", _DAY, recorded=same) == {("src/neurobrix/c.py", "later"): ("1374", "2026-11-20")}


# ─── HELD rows: an attribute call of another class is not a call of a module function ───

def _held_repo(tmp_path: Path, other_call: str) -> Path:
    r = tmp_path / "repo"
    _write(r, "site/nbxtensor/__init__.py", "")
    _write(r, "src/neurobrix/__init__.py", "")
    _write(r, "src/neurobrix/stamps.py", "def mark(name, sync=None):\n    return (name, sync() if sync else None)\n")
    _write(r, "src/neurobrix/user.py", "from neurobrix import stamps\n\ndef go():\n    return stamps.mark('x')\n")
    _write(r, "tools/judge.py", "class Judge:\n    def mark(self, arm, ok):\n        return arm, ok\n\n"
                                "def run(j):\n    " + other_call + "\n")
    _write(r, "tools/use.py", "from neurobrix.user import go\nfrom judge import run\ngo(), run(None)\n")
    return r


def test_an_attribute_call_of_another_class_does_not_keep_a_module_function_alive(tmp_path):
    row = ("wrapper", "src/neurobrix/stamps.py", "mark(sync)")
    r = _held_repo(tmp_path, "return j.mark('gate', False)")               # another class's mark, two positionals
    lib = r / "site" / "nbxtensor"
    assert row not in {f[:3] for f in S.scan(r, lib, today=_DAY)}              # the by-name reading is fooled
    assert S.held(r, lib, {row}) == {row}                                      # the listed row is held
    r2 = _held_repo(tmp_path / "two", "from neurobrix import stamps\n    return stamps.mark('gate', lambda: 0)")
    assert S.held(r2, r2 / "site" / "nbxtensor", {row}) == set()              # the very function passes it: cleared
