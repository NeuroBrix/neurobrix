import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "..", "..", "..", "src", "neurobrix", "processors", "espeak_ng", "reader")


def load_reader():
    """The reader is loaded from its source folder as one package, under a name of its own, so that the test needs no
    installed package."""
    name = "espeak_ng_reader_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(FOLDER, "__init__.py"),
                                                      submodule_search_locations=[FOLDER])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


reader = load_reader()
data_files = sys.modules[reader.__name__ + ".data_files"]

LIST = """// a comment line
zonka      zoNk@
vaneb       $2      // stress on the second syllable
(zel mot)    zel||m'ot@l   // a group of words
?5     wumba     wumb@    $only
?!5    plimb    plIm:
glorp    glorb   $text
$textmode
snerf   snurf
$phonememode
vint    vIn:t   $atend
"""


def test_each_line_gives_word_phonemes_flags_with_condition():
    e = data_files.parse_word_list(LIST)
    assert [(x.words, x.phonemes, x.text, x.flags, x.condition) for x in e] == [
        (("zonka",), "zoNk@", None, (), None),
        (("vaneb",), None, None, ("2",), None),
        (("zel", "mot"), "zel||m'ot@l", None, (), None),
        (("wumba",), "wumb@", None, ("only",), (5, True)),
        (("plimb",), "plIm:", None, (), (5, False)),
        (("glorp",), None, "glorb", ("text",), None),
        (("snerf",), None, "snurf", (), None),
        (("vint",), "vIn:t", None, ("atend",), None),
    ]
    assert e[0].line == 2


def test_a_condition_with_no_entry_is_refused_with_its_line():
    with pytest.raises(data_files.DataFileError, match="line 2"):
        data_files.parse_word_list("gilomb  gIlomb\n?5\n", "list")
