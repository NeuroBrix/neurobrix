"""The scope tool's call sites can be built FIRST, before any receiver typing (2026-10-26: gate tha4's scope died on
"sites() re-entered while building" — a commit touching a module global made `gate_constants` ask `Program.sites()`
before anything had built the attribute slots; the first call on an attribute receiver, `self.a.go()`, typed it,
`_ensure_slots` asked `sites()` again). What this test would do if the code were wrong: RuntimeError — RED. And the
answer may only be WIDER while the sites are built, never narrower: the call reaches the method it names."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import gate_scope_code as C          # noqa: E402

SRC = '''
_FLAG = False


class A:
    def go(self):
        return 1


class B:
    def __init__(self):
        self.a = A()

    def run(self):
        return self.a.go()


def set_flag():
    global _FLAG
    _FLAG = True
'''


def test_the_sites_are_built_first_and_reach_the_named_method(tmp_path):
    pkg = tmp_path / "src" / "neurobrix"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "m.py").write_text(SRC)
    index = C.SourceIndex(tmp_path / "src")
    units = C.unit_nodes(index)
    P = C.Program(index, units, set(units))
    sites = P.sites()                                    # first, before any slot typing
    go = [u for u in units if u.endswith("A.go")]
    assert go, sorted(units)
    callers = {s[0] for s in sites.get(go[0], ())}
    assert any(c.endswith("B.run") for c in callers), (go, callers)
