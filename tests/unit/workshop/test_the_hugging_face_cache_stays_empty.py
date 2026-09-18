"""The Hugging Face hub cache must stay empty. A model is downloaded once.

A snapshot is downloaded ONCE, into the configured snapshots root, and every
consumer reads it THERE — the build toolchain traces from it, vendor tools and
oracles load from it, the engine resolves its third-party fallbacks against it.
A second copy under `~/.cache/huggingface/hub` is a duplicate of NAS-resident
data on the one filesystem that cannot hold it.

On 2026-09-17 that cache held 4.0 G across 25 entries — and EVERY model in it
already had its snapshot on the NAS, verified by checksum, while the local disk
had reached 0 MB the same day. Four writers put it there: a bare hub id in the
audio output processor, a literal path with "otherwise try Hub" behind it in one
profiler, a bare hub id in another, and the R29 judging harness.

This gate fires when it refills. It reads MACHINE STATE, so it skips where there
is no such cache rather than failing on a developer's laptop.
"""

import os
from pathlib import Path

import pytest

CACHE = Path(os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache" / "huggingface" / "hub"))

#: Below this, an entry is bookkeeping (refs, version.txt, .locks), not a model.
BOOKKEEPING_BYTES = 1024 * 1024


def _models_with_bytes():
    if not CACHE.is_dir():
        return []
    found = []
    for entry in CACHE.iterdir():
        if not entry.name.startswith("models--"):
            continue
        total = 0
        for dirpath, _, names in os.walk(entry):
            for n in names:
                p = Path(dirpath) / n
                try:
                    if not p.is_symlink():
                        total += p.stat().st_size
                except OSError:
                    pass
        if total > BOOKKEEPING_BYTES:
            found.append((entry.name, total))
    return sorted(found, key=lambda t: -t[1])


@pytest.mark.skipif(not CACHE.is_dir(), reason="no Hugging Face hub cache on this machine")
def test_no_model_has_been_downloaded_into_the_hub_cache():
    offenders = _models_with_bytes()
    assert not offenders, (
        "the Hugging Face hub cache is holding model bytes:\n  "
        + "\n  ".join(f"{n}: {s / 2**20:.0f} MB" for n, s in offenders)
        + "\nA snapshot is downloaded once, into the configured snapshots root, "
          "and read there. Find what fetched this — a bare hub id handed to "
          "`from_pretrained`, or a path that falls back to the local disk — and "
          "fix it at that source rather than deleting the bytes and waiting."
    )
