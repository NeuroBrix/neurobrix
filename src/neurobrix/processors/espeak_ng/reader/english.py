"""The constants of the English reading that are in no data file: the layout of the pinned data, the rules this
lot observed, and the limits of the lot. Each observed rule is stated in one plain line with its observation id
(OBS-nn); the observations themselves (the inputs, the outputs of the vendor binary 1.52.0, the list entries) are
kept with the lot's proof, outside the package and outside this repository.
"""
import os

# The layout of the pinned data folder: the word list, the folder of the phoneme definitions, the voice files.
WORD_LIST = os.path.join("dictsource", "en_list")
PHONEME_FOLDER = "phsource"
VOICE_FOLDER = os.path.join("espeak-ng-data", "lang")

# The two variants and the `language` value their voice file declares.
VARIANT_LANGUAGE = {"us": "en-us", "gb": "en-gb"}

# A voice file with no `phonemes` line reads the phoneme table named after its language family (OBS-01).
DEFAULT_TABLE = "en"

# A phoneme with no `ipa` statement is written from its name, character by character, by this table (OBS-02);
# the documentation calls the names "based loosely on the Kirshenbaum (ASCII-IPA) representation", and the
# nasal mark `~` is that convention's. A character not listed and not a letter writes nothing, and a digit after
# the first character writes nothing (OBS-03). The tie mark goes before every letter of a written form but the
# first (OBS-04).
NAME_TO_WRITTEN = {
    "D": "ð", "T": "θ", "S": "ʃ", "Z": "ʒ", "N": "ŋ", "g": "ɡ", "I": "ɪ", "U": "ʊ", "E": "ɛ", "V": "ʌ",
    "O": "ɔ", "0": "ɒ", "@": "ə", "3": "ɜ", "A": "ɑ", ":": "ː", "~": "̃",
    ";": "ʲ",     # the palatal mark after a consonant (OBS-05)
    "||": " ",    # the word break inside a phoneme string writes one space (OBS-06)
}

# The tie mark written before every letter of a written form but the first when the caller names none (OBS-04).
DEFAULT_TIE = "^"

# The phoneme named `=` in the phoneme source gives the primary stress to the vowel before it (OBS-12).
STRESS_TO_PREVIOUS = "="

# A letter's name in the word list is the entry of the letter with this prefix, else of the bare letter (OBS-13).
LETTER_NAME_PREFIX = "_"

# The stress marks of the written form: primary, secondary (OBS-07).
PRIMARY_MARK = "ˈ"
SECONDARY_MARK = "ˌ"

# Stress levels, as the phoneme file's `stress_type` values name them (0 diminished, 1 unstressed, 2 secondary,
# 4 primary); the marks are written from level 4 (primary) and levels 2-3 (secondary).
DIMINISHED, UNSTRESSED, SECONDARY, PRIMARY = 0, 1, 2, 4

# A word whose entry places no primary stress takes it on its FIRST unmarked vowel that is not of the
# `unstressed` property; a word with no such vowel has no primary stress, and a lone marked secondary stays
# secondary (OBS-08). An unmarked full vowel with no stressed neighbour syllable, in a word that has a primary
# stress, takes a secondary stress (OBS-09).

# The list flags that make an entry unusable for a word read alone (they need the word before it, a dot after
# it, or a removed suffix), from the documentation of the flags.
FLAGS_NEEDING_CONTEXT = ("verb", "noun", "past", "hasdot", "stem")
# The flags that leave the word without a primary stress when it is read alone (the documentation of `$u`...).
FLAGS_UNSTRESSED = ("u", "u1", "u2", "u3", "unstressend")
# ... unless one of these is there too: a word read alone ends its clause (the documentation of `$strend`,
# `$strend2` and `$u+`; OBS-10).
FLAGS_STRESSED_AT_END = ("strend", "strend2", "u+", "u1+", "u2+", "u3+")

# Letters spelled one by one (`$abbrev` with no phoneme string): every letter but the last has its primary
# stress lowered to secondary (OBS-11).
