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


def test_a_voice_file_gives_each_attribute_the_values_of_its_lines():
    v = data_files.parse_voice("name Quay (North)\nlanguage ab-cd 2\nlanguage ab 3\n\nphonemes zz // c\n"
                               "dictrules 5 8\nreplace 03 o  e\nreplace 03 o2 e\n")
    assert v["language"] == [["ab-cd", "2"], ["ab", "3"]]
    assert v["phonemes"] == [["zz"]]
    assert v["dictrules"] == [["5", "8"]]
    assert v["replace"] == [["03", "o", "e"], ["03", "o2", "e"]]
