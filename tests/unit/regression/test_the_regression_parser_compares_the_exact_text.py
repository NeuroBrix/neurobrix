"""The regression battery compares a text run's exact text: a space the engine writes at its start or end is seen.

Before: `_parse_llm_text` stripped stdout, so a reply that began with the template's space (and the golden captured
from it) could never fail the comparison. Now only the one newline `print` ends the text with is dropped.

What this test does if the code were wrong: a strip (red) returns 'Hello' for ' Hello\\n'.
"""
from tests.regression.test_all_models import _parse_llm_text


def test_the_parser_keeps_the_space_the_text_starts_and_ends_with():
    assert _parse_llm_text(" Hello! How can I \n") == " Hello! How can I "
    assert _parse_llm_text("  Hello\n") == "  Hello"


def test_the_parser_drops_only_the_one_newline_print_adds():
    assert _parse_llm_text("Hello\n") == "Hello"
    assert _parse_llm_text("Hello\n\n") == "Hello\n"
    assert _parse_llm_text("Hello") == "Hello"


def test_a_run_that_printed_nothing_but_blanks_is_no_text():
    assert _parse_llm_text("") is None and _parse_llm_text(" \n") is None
