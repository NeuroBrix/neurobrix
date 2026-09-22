"""No tool writes the container cache's path itself; they all read the one door.

`core.paths.cache_dir()` resolves NEUROBRIX_CACHE, then ~/.neurobrix/paths.json, then the
default — and its own docstring records that the answer used to live in four places reached by
two mechanisms, one of which called itself "single source of truth" and was not one.

It happened again: on 2026-09-22 `tools/precision_zoo_campaign.py` held
`CACHE = Path(os.path.expanduser("~")) / ".neurobrix" / "cache"`, so a census pass whose
environment named the shared NFS catalogue read an empty local directory instead and every
model died on a missing `topology.json`. Four other tools held the same literal.

The machine-local REPLAY cache (`~/.neurobrix/replay_cache`) is a different thing and is not
covered here: it is per-machine autotune state, not the catalogue.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
# a literal container-cache path: `.neurobrix` joined with `cache`, in any spelling
LITERAL = re.compile(r'\.neurobrix["\']?\s*(?:/|,)\s*["\']?cache|\.neurobrix/cache')


def test_no_tool_spells_the_container_cache_path():
    # Scoped to the census/certification CHAIN. The same literal is in about fifteen other
    # tools this chantier does not drive and cannot exercise; widening the gate to them would
    # commit a red test for everyone. They are recorded in docs/reference/owed-proofs.md
    # (2026-09-22) so the finding is not lost, and this gate grows as they are fixed.
    CHAIN = {"precision_zoo_campaign.py", "levers_byte_identity.py", "unroll_census_report.py",
             "artefact_voice.py", "ir_census.py", "certified_census.py", "certified_checkpoint.py",
             "bucket_loss.py"}
    offenders = []
    for f in sorted((REPO / "tools").rglob("*.py")):
        if f.name not in CHAIN:
            continue
        for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "replay_cache" in line:
                continue
            # prose in a docstring may NAME the path; only construction is the defect
            if not re.search(r"Path\s*\(|Path\.home\s*\(|expanduser", line):
                continue
            if LITERAL.search(line):
                offenders.append(f"{f.relative_to(REPO)}:{n}: {line.strip()}")
    assert not offenders, (
        "these spell the container cache instead of reading core.paths.cache_dir():\n  "
        + "\n  ".join(offenders))
