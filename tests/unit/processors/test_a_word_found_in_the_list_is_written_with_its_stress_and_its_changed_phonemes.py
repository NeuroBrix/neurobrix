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

# the phoneme that gives the primary stress to the vowel before it carries the name the reader's rule fixes
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
phoneme =
  stress
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
"""
LIST = """melo   MnQbLlQs
pomer  PpQsMnQbRh
lopor  LlQaPpQsRh
vomole  VvQaMnQaQsPp!1Qa
bivam  PpQa=MnQb
?5 zhomp  DzQaMnPp
shomp   ZsQaMnPp   $verb
op     $abbrev
_o     Qa
_p     PpQs
lep    $2
molem  MnQaLlQbMn
temo   QbLlQs
temo   MnQaLlQs
momel  MnQbMn  LlQsLl
"""


def write_data(root, phonemes=PHONEMES, east="language en-gb 7\nphonemes qa\nreplace 2 Qb Qa\n"):
    """A toy data folder under the layout the reader names (its word list, phoneme folder and voice folder)."""
    for rel, text in ((os.path.join(english.PHONEME_FOLDER, "phonemes"), phonemes),
                      (os.path.join(english.PHONEME_FOLDER, "more"), "phoneme Z2\n  import_phoneme qa/Pp\nendphoneme\n"),
                      (english.WORD_LIST, LIST),
                      (os.path.join(english.VOICE_FOLDER, "toy", "west"),
                       "language en-us 7\nphonemes qb\ndictrules 5\n"),
                      (os.path.join(english.VOICE_FOLDER, "toy", "east"), east)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return str(root)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return write_data(tmp_path_factory.mktemp("data"))


def test_a_listed_word_is_written_per_variant(data):
    us, gb = reader.Reader(data, "us"), reader.Reader(data, "gb")
    assert us.units("melo") == "m\u02c8\u0289l\u0259 "
    assert gb.units("Melo") == "m\u02c8\u0275l\u0259 "                # the voice replaces a vowel the stress has not reached
    assert us.units("pomer") == "p\u0259m\u02c8\u0289\u027e "
    assert us.units("lopor") == "l\u02c8\u0275p\u0268\u027e "       # the variant's table changes the schwa
    assert gb.units("lopor") == "l\u02c8\u0275p\u0259\u027e "
    assert us.units("vomole") == "v\u02cc\u0275m\u0275p\u02c8\u0275 "   # a marked primary, a secondary
    assert us.units("zhomp") == "\u00f0^z\u02c8\u0275mp "          # the tie, a name written by its characters
    assert us.units("op") == "\u02cc\u0275p\u0259 "                # letters, the first lowered to secondary


def test_a_mark_of_the_phoneme_file_gives_the_primary_stress_to_the_vowel_before_it(data):
    assert reader.Reader(data, "us").units("bivam") == "p\u02c8\u0275m\u0289 "


def test_the_replace_lines_of_a_voice_change_the_phonemes_outside_a_stressed_syllable(data):
    # `replace 2 Qb Qa`: the vowel is replaced in the variant whose voice says so, before the stress is placed
    assert reader.Reader(data, "gb").units("bivam") == "p\u02c8\u0275m\u0275 "


def test_what_this_lot_does_not_read_is_refused_by_name_never_guessed(data):
    us, gb = reader.Reader(data, "us"), reader.Reader(data, "gb")
    for r, text, why in ((gb, "zhomp", "not in the word list"), (us, "shomp", "not in the word list"),
                         (us, "lep", "no phoneme string"), (us, "melo melo", "one word"), (us, "quvex", "not in")):
        with pytest.raises(reader.Unreadable, match=why):
            r.units(text)
    with pytest.raises(ValueError, match="it reads: gb, us"):
        reader.Reader(data, "au")


def test_absent_data_is_a_named_error(tmp_path):
    with pytest.raises(reader.DataFileError, match="declares the language en-us"):
        reader.Reader(str(tmp_path), "us")


def test_an_entry_with_no_stress_mark_takes_its_primary_stress_on_its_first_full_vowel(data):
    # two full vowels, no mark: the first takes the primary, the second (next to it) stays unstressed
    assert reader.Reader(data, "us").units("molem") == "m\u02c8\u0275l\u0289m "


def test_a_word_listed_twice_reads_the_entry_lowest_in_the_list(data):
    # the list is searched from its last line up: the second line of the word wins
    assert reader.Reader(data, "us").units("temo") == "m\u02c8\u0275l\u0259 "


def test_a_list_line_with_fields_beyond_the_format_is_refused_by_name_when_a_word_reaches_it(data):
    entry = [e for e in data_files.parse_word_list(LIST) if e.words == ("momel",)][0]
    assert entry.excess == ("LlQsLl",)
    with pytest.raises(reader.Unreadable, match="fields beyond the format"):
        reader.Reader(data, "us").units("momel")


def test_a_voice_that_replaces_a_phoneme_its_table_does_not_define_is_a_named_error(tmp_path):
    root = write_data(tmp_path, east="language en-gb 7\nphonemes qa\nreplace 0 Qa zz\n")
    with pytest.raises(reader.DataFileError, match="replaces the phoneme 'zz', which the table qa does not define"):
        reader.Reader(root, "gb")


def test_a_procedure_that_calls_itself_with_changes_that_come_back_are_refused_by_name(tmp_path, monkeypatch):
    # a bound on the work the reader may do: with its come-back check removed the test ends in a red, not a hang
    table_class = sys.modules[reader.__name__ + ".phonemes"].PhonemeTable
    run, calls = table_class._run, []

    def bounded(self, *args, **kwargs):
        calls.append(1)
        assert len(calls) < 1000, "the changes of a phoneme never end"
        return run(self, *args, **kwargs)

    monkeypatch.setattr(table_class, "_run", bounded)
    phonemes = PHONEMES.replace("""phoneme Mn
  nasal
""", """procedure loop
  CALL loop
endprocedure
phoneme Mn
  nasal
  CALL loop
""").replace("""phoneme Zs
  frc vls pal
""", """phoneme Zs
  frc vls pal
  ChangePhoneme(Dz)
""").replace("""phoneme Dz
  afr vcd pla
""", """phoneme Dz
  afr vcd pla
  ChangePhoneme(Zs)
""")
    us = reader.Reader(write_data(tmp_path, phonemes=phonemes), "us")
    with pytest.raises(reader.DataFileError, match="the procedure 'loop' calls itself"):
        us.units("melo")
    with pytest.raises(reader.Unreadable, match="come back to it"):
        us.units("zhomp")


def test_a_stress_phoneme_with_no_stress_type_is_a_named_error(tmp_path):
    phonemes = PHONEMES.replace("  stress_type 4\n", "")
    with pytest.raises(reader.DataFileError, match="has no stress_type"):
        reader.Reader(write_data(tmp_path, phonemes=phonemes), "us").units("vomole")


def test_a_voice_line_with_no_value_or_a_dictrules_value_that_is_no_number_is_a_named_error(tmp_path):
    with pytest.raises(reader.DataFileError, match="`language` line with no value"):
        reader.Reader(write_data(tmp_path, east="language\nphonemes qa\n"), "gb")
    root = write_data(tmp_path / "b", east="language en-gb 7\nphonemes qa\ndictrules x7\n")
    with pytest.raises(reader.DataFileError, match="`dictrules` value which is no number: 'x7'"):
        reader.Reader(root, "gb")
