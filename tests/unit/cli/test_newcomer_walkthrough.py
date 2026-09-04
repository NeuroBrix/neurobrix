"""CLI defects found by a newcomer walkthrough from an empty machine.

Reported 2026-09-03 against 0.5.2 from PyPI, on a 2 vCPU / 1.9 GB Debian box
with no GPU. The walkthrough finished — a real ×4 upscale in 1.93 s — but
four of the seven minutes went on avoidable CLI problems, and each one is
the kind that makes someone conclude the engine does not do something it
does.

Each test here is one reported item, named by its section in the report.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neurobrix.cli import create_parser
from neurobrix.cli.commands.registry import (
    _strip_build_stamp,
    _suggest_run_command,
)


# --- §4: the last line before a newcomer's first run ------------------------

def _cache_with_family(family: str) -> Path:
    directory = Path(tempfile.mkdtemp())
    (directory / "manifest.json").write_text(json.dumps({"family": family}))
    return directory


def test_import_suggests_a_command_that_can_actually_work():
    """`import` closed with `--prompt "..."` for EVERY family, so importing an
    upscaler ended with a command that answers
    `ZERO FALLBACK: family 'upscaler' requires --input-image`. It is the last
    line read before the first run."""
    suggestion = _suggest_run_command("Swin2SR-Classical-x4",
                                      _cache_with_family("upscaler"))
    assert "--input-image" in suggestion
    assert "--prompt" not in suggestion


@pytest.mark.parametrize("family,expected", [
    ("upscaler", "--input-image"),
    ("llm", "--prompt"),
    ("stt", "--audio"),
    ("image", "--prompt"),
])
def test_the_suggestion_comes_from_the_family_contract(family, expected):
    """Built from the family YAML's `inputs.required`, so a new family needs
    no change here — no family-name cascade in the CLI (R32)."""
    assert expected in _suggest_run_command("M", _cache_with_family(family))


def test_an_unreadable_manifest_falls_back_to_the_bare_command():
    """Better to suggest less than to invent flags the model may reject."""
    suggestion = _suggest_run_command("M", Path("/nonexistent-cache"))
    assert suggestion == "neurobrix run --model M"


# --- §5: list contradicting the import it just performed --------------------

@pytest.mark.parametrize("stem,expected", [
    ("Swin2SR-Classical-x4.20260828T183831", "Swin2SR-Classical-x4"),
    ("Model", "Model"),
    ("Name.With.Dots.20260101T000000", "Name.With.Dots"),
    ("Not.A.Stamp.2026", "Not.A.Stamp.2026"),
])
def test_build_stamp_is_stripped_before_comparing(stem, expected):
    """`list` compared store filenames (which carry a build stamp) against
    extracted cache directory names (which do not), so a freshly extracted
    model appeared as installed AND as "store only (not extracted)" on the
    same screen, with an instruction to import it again."""
    assert _strip_build_stamp(stem) == expected


# --- §6: one file, two flag names between sibling commands ------------------

def test_upscale_accepts_both_input_flag_names():
    """`upscale` took `--input`; `run` took `--input-image` for the same file
    and the same family. Which one worked depended only on which subcommand
    the user reached first."""
    parser = create_parser()
    for flag in ("--input", "--input-image"):
        args = parser.parse_args(["upscale", "--model", "M", flag, "in.png",
                                  "--output", "out.png"])
        assert args.input == "in.png", f"{flag} must reach the same destination"


# --- §7: "no models installed" without saying where it looked ---------------

def test_not_found_names_the_directory_it_searched(monkeypatch, tmp_path):
    """Models live under the INVOKING user's home. Running under sudo — which
    people reach for the moment anything looks like a permission problem —
    searches root's cache and reports that a machine full of models has none."""
    from neurobrix.cli import utils

    monkeypatch.setattr(utils, "CACHE_DIR", tmp_path / "empty-cache")

    with pytest.raises(FileNotFoundError) as excinfo:
        utils.find_model("Swin2SR-Classical-x4")

    message = str(excinfo.value)
    assert str(tmp_path / "empty-cache") in message, "must name the directory searched"
    assert "sudo" in message, "must name the trap that produces this state"


def test_not_found_still_lists_installed_models_when_there_are_some(monkeypatch, tmp_path):
    """The non-empty branch was already good; it must stay that way."""
    from neurobrix.cli import utils

    cache = tmp_path / "cache"
    (cache / "SomeModel").mkdir(parents=True)
    (cache / "SomeModel" / "manifest.json").write_text("{}")
    monkeypatch.setattr(utils, "CACHE_DIR", cache)

    with pytest.raises(FileNotFoundError) as excinfo:
        utils.find_model("Missing")

    message = str(excinfo.value)
    assert "SomeModel" in message
    assert "No models installed" not in message
