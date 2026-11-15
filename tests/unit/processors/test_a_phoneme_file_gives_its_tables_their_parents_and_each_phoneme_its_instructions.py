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
english = sys.modules[reader.__name__ + ".english"]

PHONEMES = """phoneme !1
  stress
  stress_type 4
endphoneme
phoneme !2
  stress
  stress_type 2
endphoneme
phonemetable qa base
phoneme #w virtual
endphoneme
phoneme Qa
  vwl starttype #w endtype #w
  ipa U+0275
  length 90
  FMT(vowel/ab)
endphoneme
phoneme Qb
  vwl starttype #w endtype #w
  ipa U+0289
endphoneme
phoneme Qs   // the short neutral vowel
  vwl unstressed starttype #y endtype #y
  ipa U+0259
  IF thisPh(isWordEnd) AND thisPh(isNotStressed) THEN
    FMT(vowel/cd)
  ENDIF
endphoneme
phoneme Qt
  vwl unstressed
  ipa U+0268
endphoneme
phoneme Rh
  rhotic liquid
  ipa U+027E
endphoneme
phoneme Mn
  nasal
  ipa U+006D
endphoneme
phoneme Ll
  lat liquid
  ipa U+006C
endphoneme
phoneme Vv
  frc vcd den
  ipa U+0076
endphoneme
phoneme Pp
  stp vls alv
  ipa U+0070
  Vowelin f1=2 f2=1100 -60 80
endphoneme
phoneme Dz
  afr vcd pla
endphoneme
phoneme Zs
  frc vls pal
endphoneme
include more
phonemetable qb qa
phoneme Qs
  vwl unstressed starttype #y endtype #y
  ipa U+0259
  IF thisPh(isNotStressed) AND nextPh(isRhotic) THEN
    ChangePhoneme(Qt)
  ELIF nextPh(Pp) THEN
    ipa(NULL)
  ELSE
    IfNextVowelAppend(Rh)
  ENDIF
endphoneme
phonemetable qz base
include no_such_file
"""
MORE = "phoneme Z2\n  import_phoneme qa/Pp\nendphoneme\n"


def read(name):
    assert name == "more", "a table that is not asked for must never get its files read: %s" % name
    return MORE


def test_tables_parents_statements_with_instructions():
    s = data_files.parse_phoneme_source(PHONEMES, read, tables=("qb",))
    assert s.parents == {"base": None, "qa": "base", "qb": "qa"}
    us, qa = s.table("qb"), s.table("qa")
    assert set(qa) == {"!1", "!2", "#w", "Qa", "Qb", "Qs", "Qt", "Rh", "Mn", "Ll", "Vv", "Pp", "Dz", "Zs", "Z2"}
    assert qa["Qa"].values["ipa"] == "\u0275" and qa["Qa"].words == ["vwl"] and qa["#w"].words == ["virtual"]
    assert qa["!1"].values["stress_type"] == "4" and qa["Pp"].words == ["stp", "vls", "alv"]
    assert qa["Qs"].table == "qa" and us["Qs"].table == "qb" and us["Qa"] is qa["Qa"]
    assert ("do", "import_phoneme", "qa/Pp") in qa["Z2"].program
    node = us["Qs"].program[0]
    assert node[0] == "if"
    assert node[1][0] == (("AND", [(False, "thisPh", "isNotStressed"), (False, "nextPh", "isRhotic")]),
                          [("do", "ChangePhoneme", "Qt")])
    assert node[1][1] == (("AND", [(False, "nextPh", "Pp")]), [("do", "ipa", "NULL")])
    assert node[2] == [("do", "IfNextVowelAppend", "Rh")]


def test_a_definition_that_is_not_ended_is_refused_by_name():
    with pytest.raises(data_files.DataFileError, match="'Qa' is not ended"):
        data_files.parse_phoneme_source("phoneme Qa\n  vwl\nphoneme Qb\nendphoneme\n", read)


def test_a_table_the_file_does_not_define_is_refused_by_name():
    with pytest.raises(data_files.DataFileError, match="no table 'qq'"):
        data_files.parse_phoneme_source(PHONEMES, read, tables=("qq",))
