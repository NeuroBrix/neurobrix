"""No test in this suite would SKIP where it should FAIL.

The rule was in the vacuous-guard register for days, and then committed in the
file written to enforce it: a module-level

    try:
        from x import a, b
        HAS = True
    except Exception:
        HAS = False

turns the absence of `b` -- the function under test, not yet written -- into
the same skip as the absence of the machine. Five red tests became five skips,
and a skip is invisible in a count. Nobody audits three hundred of them.

The parry is SCOPE, not removal. A test that needs Metal on a machine without
Metal must skip, and `import x` inside a catch-all fails only when `x` is
unimportable, which is exactly the environment fact a skip is for. What must
never be inside it is `from x import a`, because `a` is what the test is
about.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_no_skip_hides_a_red.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from skips_that_hide_a_red import _swallowed_names, _catches_everything  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests"


_GUARDS_A_MODULE = """
try:
    import torch
except Exception:
    torch = None
"""

_GUARDS_NAMES = """
try:
    from neurobrix.kernels.helpers import the_function_under_test
    HAS = True
except Exception:
    HAS = False
"""

_GUARDS_A_SPECIFIC_ERROR = """
try:
    from neurobrix.kernels.helpers import a
except ImportError:
    a = None
"""


def _first_try(src):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.Try))


def test_guarding_a_module_import_is_not_flagged():
    assert _swallowed_names(_first_try(_GUARDS_A_MODULE)) == [], (
        "`import torch` inside a catch-all is the legitimate form: it fails "
        "only when torch is unimportable, which is what a skip is for")


def test_guarding_a_name_import_is_flagged():
    swallowed = _swallowed_names(_first_try(_GUARDS_NAMES))
    assert swallowed == ["neurobrix.kernels.helpers.the_function_under_test"], (
        "the name under test must be reported: its absence reading as a skip "
        "is the whole defect")


def test_a_narrow_except_is_not_a_catch_all():
    """Both directions on the handler, not only on the imports.

    `except ImportError` around a name import is a different statement from
    `except Exception`: it still swallows a missing name, but it does not
    swallow a broken module, and the scan's question is the catch-all.
    """
    node = _first_try(_GUARDS_A_SPECIFIC_ERROR)
    assert not any(_catches_everything(h) for h in node.handlers)
    node2 = _first_try(_GUARDS_NAMES)
    assert any(_catches_everything(h) for h in node2.handlers)


def test_no_file_in_this_suite_carries_the_pattern():
    """The suite-wide verdict, which the two tests above make readable.

    Without them this assertion is satisfiable by a scanner that finds
    nothing, which is what a scanner reporting zero over 193 files must be
    shown not to be.
    """
    offenders = []
    for path in sorted(TESTS.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Try):
                continue
            if not any(_catches_everything(h) for h in node.handlers):
                continue
            swallowed = _swallowed_names(node)
            if swallowed:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                 f"swallows {swallowed[:3]}")
    assert not offenders, (
        "these files would skip where they should fail:\n  "
        + "\n  ".join(offenders)
        + "\n\nGuard the package import; import the names inside the tests.")
