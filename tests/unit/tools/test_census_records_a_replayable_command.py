"""The command a census records must replay to the argv it ran.

Red on 2026-09-23: `certified_census` recorded `" ".join(cmd[2:])`. It HAS the argv list and
threw the quoting away, so any argument containing a space came back as several. Replaying
Kokoro-82M's recorded command gave

    neurobrix: error: unrecognized arguments: quick brown fox jumps over the lazy dog, ...

because `--prompt The quick brown fox ...` split at every space and only `The` reached the
flag. 45 of the 59 censused models carry a multi-word argument, so most of the census's
commands could not be replayed at all — including three of the ten verification cells, which
is how it was found.

`shlex.join` is the whole fix: it is the inverse of `shlex.split`, which is what a shell (and
`eval`) does to that string.
"""
import shlex


def _recorded(argv):
    """The line the census writes, as it writes it."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("tools").resolve()))
    src = pathlib.Path("tools/certified_census.py").read_text(encoding="utf-8")
    assert "shlex.join(cmd[2:])" in src, \
        'the census still records `" ".join(cmd[2:])`: a multi-word argument cannot be replayed'
    return shlex.join(argv)


def test_a_multi_word_prompt_replays_to_one_argument():
    argv = ["run", "--model", "Kokoro-82M",
            "--prompt", "The quick brown fox jumps over the lazy dog.",
            "--seed", "42", "--triton"]
    back = shlex.split(_recorded(argv))
    assert back == argv, f"replay changed the argv:\n  ran  {argv}\n  back {back}"
    assert back[back.index("--prompt") + 1] == "The quick brown fox jumps over the lazy dog."


def test_a_path_with_a_space_survives():
    argv = ["run", "--model", "x", "--input-image", "/tmp/a folder/apple 448.png"]
    assert shlex.split(_recorded(argv)) == argv


def test_an_ordinary_command_is_unchanged_in_meaning():
    argv = ["run", "--model", "swinir-classical-x2", "--triton"]
    assert shlex.split(_recorded(argv)) == argv
