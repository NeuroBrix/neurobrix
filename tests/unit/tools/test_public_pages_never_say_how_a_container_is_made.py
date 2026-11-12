"""The public vocabulary check (tools/public_vocabulary.py): red on the private tool's name and verb, in any page."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import public_vocabulary as pv  # noqa: E402


def test_the_banned_words_are_found_whatever_their_form():
    T, V = pv.TOOL, pv.VERB
    text = f"{T.title()} built it.\nThe model was re{V}d once.\nthe {V.title()}r's rule\nre-{V} owed, {V[:-1]}ing, {V}s\nfine line\n"
    assert [n for n, _ in pv.occurrences("p.md", text, [])] == [1, 2, 3, 4, 4, 4]


def test_a_word_that_only_contains_one_is_not_one():
    T, V = pv.TOOL, pv.VERB
    text = f"a {V}back, a {T}d key, the sub{V[:-1]}ted mean, con{V[:-1]}ted, ex{V[:-1]}tion, {V.title()}y"
    assert pv.occurrences("p.md", text, []) == []


def test_an_allowed_text_covers_only_its_own_page_and_its_own_words():
    V = pv.VERB
    pairs = [("p.md", "tools/container_request.py"), ("*", f'"{V}":')]
    assert pv.occurrences("p.md", "see `tools/container_request.py`", pairs) == []
    assert pv.occurrences("q.md", f"see `tools/{V}_request.py`", pairs) == [(1, V)]
    assert pv.occurrences("q.md", f'{{"{V}": 9}}', pairs) == []
    assert pv.occurrences("p.md", f"tools/container_request.py reads the {V}", pairs) == [(1, V)]


def test_a_name_with_underscores_hides_nothing():
    """A name that holds a banned word shows it to a reader: each real name stays in the allow-list with its reason, so
    the list says what a rename would remove."""
    V = pv.VERB
    text = f"{V}_value, NBX_MALLOC_{V.upper()}, tools/re{V}_zoo.py"
    assert [w for _, w in pv.occurrences("p.md", text, [])] == [V, V.upper(), "re" + V]

def test_an_allowed_line_without_its_reason_is_refused(tmp_path):
    bad = tmp_path / "allow.tsv"
    bad.write_text("p.md\ttools/container_request.py\n")
    with pytest.raises(ValueError, match="path<TAB>exact text<TAB>reason"):
        pv.allowed(bad)


def test_every_allowed_text_is_still_on_its_page():
    stale = [(p, t) for p, t in pv.allowed() if p != "*" and t not in (REPO / p).read_text(encoding="utf-8")]
    assert stale == [], "the allow-list names texts their page no longer holds: remove them"


def test_no_tracked_page_says_how_a_container_is_made():
    found = pv.scan()
    assert found == {}, "\n".join(f"{p}:{n}: {w}" for p, hits in found.items() for n, w in hits[:5])


def test_a_generated_page_cannot_bring_the_words_back():
    """catalogue-state.md is printed by tools/catalogue_state_report.py, and some of its cells are read from campaign
    records that carry their writer's words: what the tool prints goes through its own `public_wording`. What this
    test does if the filter is gone: the artefact's words reach the page and the check below finds them."""
    import io
    import catalogue_state_report as report
    T, V = pv.TOOL, pv.VERB
    artefact = (f"four symbol-chain breaks unadjudicated ({T.title()}); RE-{V.upper()}D on the fifth attempt, then "
                f"re{V}d; the {V}r binds this axis; {V}d at 64x64, a second {V} costs 15 s; {V[:-1]}ing "
                f"`trace_value`=4096 with tools/container_request.py")
    page = io.StringIO()
    report._PublicPage(page).write(artefact + "\n")
    printed = page.getvalue()
    real_names = [("*", "trace_value"), ("*", "tools/container_request.py")]
    assert pv.occurrences("docs/reference/catalogue-state.md", artefact, real_names) != []
    assert pv.occurrences("docs/reference/catalogue-state.md", printed, real_names) == [], printed
    assert "(the container builder)" in printed and "REBUILT on the fifth attempt, then rebuilt" in printed
    assert "`trace_value`=4096" in printed and "tools/container_request.py" in printed       # real names untouched
    assert report.public_wording(printed) == printed                                       # and it is stable
