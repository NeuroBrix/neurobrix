"""The regression harness removes the outputs it created, and only those.

`_run_out_path` / `_upscale_out_path` write cell outputs to /tmp/regression_*
and unlink BEFORE each run, never after, so every cell that ran left its file
behind. 2026-09-10 ended with a stray `regression_run_Kokoro-82M_native.wav`
removed by hand. A campaign that does not clean is not finished.

Three properties, each of which could rot silently:

  * what the session created is removed;
  * what was already there is NOT — /tmp is shared, and a harness that
    deletes by pattern deletes other people's files;
  * on a failure the outputs STAY, because instructing a red cell means
    looking at what it produced, and tidying that away is tidying away the
    evidence.

Tested through an inline pytest session (`pytester`) rather than by running a
model cell: the property belongs to the fixture, and a test that needed a
12 GB artefact to check a cleanup would never run.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_regression_harness_cleans_up.py -v
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)

_CONFTEST = (Path(__file__).resolve().parents[3]
             / "tests" / "regression" / "conftest.py")

# The cleanup half of the real conftest, written out rather than spliced from
# it: splicing produced a file that would not parse, and a test whose fixture
# is assembled by string surgery tests the surgery. `test_the_conftest_says_
# the_same_thing` below is what keeps this copy and the real one in step.
_HARNESS = """
import pytest
from pathlib import Path

_FAILED_CELLS = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.when == 'call' and report.failed:
        _FAILED_CELLS.append(report.nodeid)


@pytest.fixture(scope='session', autouse=True)
def _harness_cleans_its_outputs():
    scratch = Path('/tmp')
    before = set(scratch.glob('regression_*'))
    yield
    made = sorted(set(scratch.glob('regression_*')) - before)
    if not made:
        return
    if _FAILED_CELLS:
        for path in made:
            print(f'  {path}')
        return
    for path in made:
        try:
            path.unlink()
        except OSError:
            pass
"""


@pytest.fixture
def stranger():
    """A /tmp/regression_* file this session did NOT create."""
    path = Path("/tmp") / f"regression_run_stranger_{uuid.uuid4().hex}.txt"
    path.write_text("not mine")
    yield path
    path.unlink(missing_ok=True)


def _run(pytester, body: str):
    pytester.makeconftest(_HARNESS)
    pytester.makepyfile(test_cell=body)
    return pytester.runpytest_subprocess()


def test_the_conftest_says_the_same_thing():
    """The inline harness above is a copy. These are the lines that carry each
    of the three properties, asserted in the REAL conftest so the copy cannot
    drift into testing something the harness does not do."""
    text = _CONFTEST.read_text()
    for line in (
        "def _harness_cleans_its_outputs():",
        "before = set(scratch.glob(\"regression_*\"))",          # only what it made
        "made = sorted(set(scratch.glob(\"regression_*\")) - before)",
        "if _FAILED_CELLS:",                                      # failures keep theirs
        "def pytest_runtest_makereport(item, call):",
        "path.unlink()",
    ):
        assert line in text, f"the regression conftest no longer carries: {line}"


def test_a_passing_session_removes_what_it_made(pytester, stranger):
    made = Path("/tmp") / f"regression_run_made_{uuid.uuid4().hex}.txt"
    result = _run(pytester, f'''
from pathlib import Path
def test_writes():
    Path({str(made)!r}).write_text("output")
''')
    result.assert_outcomes(passed=1)
    assert not made.exists(), "the harness kept an output of a passing cell"
    assert stranger.exists(), "the harness deleted a file it did not create"


def test_a_failing_session_keeps_what_it_made(pytester):
    made = Path("/tmp") / f"regression_run_kept_{uuid.uuid4().hex}.txt"
    try:
        result = _run(pytester, f'''
from pathlib import Path
def test_writes_then_fails():
    Path({str(made)!r}).write_text("output")
    assert False, "the cell is red"
''')
        result.assert_outcomes(failed=1)
        assert made.exists(), (
            "the output of a FAILED cell was deleted — that is the artefact "
            "the failure has to be instructed from")
        result.stdout.fnmatch_lines([f"*{made.name}*"])
    finally:
        made.unlink(missing_ok=True)
