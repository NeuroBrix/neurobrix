"""tools/one_name.py: the `--model` a tool hands the command line, from the catalogue's `cache entry` / `hub slug`
columns and the entry's name record.

Injections (each RED, then GREEN; see the commit message): the record ignored (the entry name returned) — the
recorded case fails; a disagreement reconciled instead of refused — the disagreement case fails.
"""
from __future__ import annotations

import json

import pytest

import one_name


def _setup(tmp_path, recorded=None):
    cat = tmp_path / "catalogue.tsv"
    cat.write_text("unique name\tcache entry\thf_snapshots folder\tNAS path (models/)\thub slug\n"
                   "hexgrad/Kokoro-82M\tKokoro-82M\tKokoro-82M\ttts/Kokoro-82M\thexgrad/Kokoro-82M\n")
    entry = tmp_path / "cache" / "Kokoro-82M"
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text("{}")
    if recorded:
        (entry / "name.json").write_text(json.dumps({"name": recorded}))
    one_name.table.cache_clear()
    return str(cat), tmp_path / "cache"


def test_a_recorded_entry_is_passed_by_its_hub_slug(tmp_path):
    cat, cache = _setup(tmp_path, recorded="hexgrad/Kokoro-82M")
    assert one_name.cli_name("Kokoro-82M", cache=cache, catalogue=cat) == "hexgrad/Kokoro-82M"


def test_an_entry_without_a_record_keeps_its_name_until_it_has_one(tmp_path):
    cat, cache = _setup(tmp_path)
    assert one_name.cli_name("Kokoro-82M", cache=cache, catalogue=cat) == "Kokoro-82M"
    assert one_name.cli_name("not-installed", cache=cache, catalogue=cat) == "not-installed"


def test_the_catalogue_and_the_record_disagreeing_is_refused_by_name(tmp_path):
    cat, cache = _setup(tmp_path, recorded="someone/Kokoro-82M")
    with pytest.raises(one_name.NamesDisagree, match="hexgrad/Kokoro-82M"):
        one_name.cli_name("Kokoro-82M", cache=cache, catalogue=cat)


def test_a_tree_that_predates_the_one_name_is_given_the_entry(tmp_path):
    """A gate's base tree, or a bisection's older commit, knows models by their entry: the argv it runs carries the
    entry; a tree with nbx/names.py carries the one name."""
    cat, cache = _setup(tmp_path, recorded="hexgrad/Kokoro-82M")
    one_name.table.cache_clear()
    old, new = tmp_path / "old" / "src", tmp_path / "new" / "src"
    (old / "neurobrix" / "nbx").mkdir(parents=True)
    (new / "neurobrix" / "nbx").mkdir(parents=True)
    (new / "neurobrix" / "nbx" / "names.py").write_text("")
    import functools
    orig = one_name.table
    one_name.table = functools.lru_cache(maxsize=4)(lambda path=None: orig(cat))
    try:
        argv = ["run", "--model", "hexgrad/Kokoro-82M", "--prompt", "hi"]
        assert one_name.argv_for_tree(argv, old)[2] == "Kokoro-82M"
        assert one_name.argv_for_tree(argv, new) == argv
    finally:
        one_name.table = orig
