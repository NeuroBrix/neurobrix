"""No engine configuration file carries the same key twice in one mapping.

YAML keeps the LAST of two equal keys and says nothing. On 2026-09-26 the Apple profile held two top-level
`autotune:` blocks — the second a seed inherited from Volta — and the loader silently dropped the first one's
measured noise-floor bands (`autotune.loss_tolerance`), which tools/bucket_loss.py then read as absent; a
comment line missing its `#` also parsed as a second `metal_backend:`. A duplicate is a configuration nobody
reads, so it is refused here for every file under config/, at any depth.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config"


class _Strict(yaml.SafeLoader):
    pass


def _refuse_duplicates(loader, node, deep=False):
    seen = {}
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise AssertionError(
                f"{loader.name}: key {key!r} appears at line {seen[key]} and again at line "
                f"{key_node.start_mark.line + 1}; YAML silently keeps only the last")
        seen[key] = key_node.start_mark.line + 1
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _refuse_duplicates)
FILES = sorted(list(CONFIG.rglob("*.yml")) + list(CONFIG.rglob("*.yaml")))


def test_there_are_files_to_check():
    assert len(FILES) > 20, f"found only {len(FILES)} config files under {CONFIG}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(CONFIG)))
def test_no_mapping_repeats_a_key(path):
    with open(path, encoding="utf-8") as fh:
        yaml.load(fh, Loader=_Strict)
