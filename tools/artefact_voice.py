#!/usr/bin/env python3
"""The voice an artefact requires on the command line, or "" where it needs none.

An artefact that ships many voicepacks and declares no `voice` in
`runtime/defaults.json` is refused by the engine — "ZERO FALLBACK: this
artefact ships 54 voices and declares none as its default" — because picking
one by a literal used to happen on every run without a word. READ FROM THE
ARTEFACT, never written here: the first in sorted order, the enumeration the
engine does. One brick for the three harnesses that compose a request (the
regression cells, the warm-serve rows, the precision campaign): on 2026-09-14
the campaign composed its own and Kokoro's before arm produced no output —
"a harness that composes its own request silently excludes families".
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# The container cache has ONE door (`core.paths.cache_dir`), which reads
# NEUROBRIX_CACHE, then ~/.neurobrix/paths.json, then the default. A literal here is a
# SECOND answer to a question that already has one: on 2026-09-22 this file sent a
# whole census pass to an empty ~/.neurobrix/cache while the canonical catalogue sat on
# the shared mount the env var named, and every model died on a missing manifest.
from neurobrix.core.paths import cache_dir as _cache_dir
CACHE = _cache_dir()


def speaker_the_artefact_requires(model: str, cache: Path = CACHE) -> str:
    if not model:
        return ""
    voices = cache / model / "modules" / "voices"
    if not voices.is_dir():
        return ""
    try:
        declared = json.loads((cache / model / "runtime" / "defaults.json").read_text()).get("voice")
    except (OSError, ValueError):
        declared = None
    if declared:
        return ""
    available = sorted(p.stem for p in voices.glob("*.pt"))
    return available[0] if available else ""


def speaker_args(model: str, cache: Path = CACHE) -> list:
    """`["--speaker", voice]` where the artefact needs one, else `[]`."""
    voice = speaker_the_artefact_requires(model, cache)
    return ["--speaker", voice] if voice else []
