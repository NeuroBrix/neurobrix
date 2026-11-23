"""The ONE reading of which nbxtensor the engine is built against: the pin `pyproject.toml` states and the commit the
python actually imports.

Between two releases the engine pins the library by its git commit (`nbxtensor @ git+<repository>@<sha40>`): a version
string does not move between the library's commits (0.1.11 named both the release and def527d, which carries
`host_measure` and `chunk_rule` the release does not), so engine main could import a name its pin lacked (2026-11-23,
`one_device_env`). At a release the pin becomes the published version (`nbxtensor==X.Y.Z`) in the tag's commit
(docs/reference/RELEASING.md). Both forms are read here; any other is refused by name.
"""
from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

REPOSITORY = "https://github.com/NeuroBrix/nbxtensor.git"
_BY_COMMIT = re.compile(r"nbxtensor@git\+" + re.escape(REPOSITORY) + r"@([0-9a-f]{40})")
_BY_RELEASE = re.compile(r"nbxtensor==(\d+\.\d+\.\d+)")


class Pin(NamedTuple):
    commit: Optional[str]      # the library commit, between releases
    release: Optional[str]     # the published version, at a release


def pin_of(dependencies: Iterable[str]) -> Pin:
    """The engine's one nbxtensor pin among `dependencies` (pyproject's `project.dependencies`)."""
    found = [d.replace(" ", "") for d in dependencies if re.match(r"nbxtensor\b", d.strip())]
    if len(found) != 1:
        raise ValueError(f"the engine names nbxtensor {len(found)} times in its dependencies: {found}")
    m = _BY_COMMIT.fullmatch(found[0])
    if m:
        return Pin(m.group(1), None)
    m = _BY_RELEASE.fullmatch(found[0])
    if m:
        return Pin(None, m.group(1))
    raise ValueError(f"the nbxtensor pin {found[0]!r} is neither a commit of {REPOSITORY} (40 hex) nor an exact "
                     f"release (==X.Y.Z)")


def installed_commit() -> Optional[str]:
    """The commit of the nbxtensor this python imports: the source tree's HEAD when it is imported from a checkout,
    else the commit pip recorded for a git install (`direct_url.json`); None for an install from a published wheel."""
    import nbxtensor
    here = Path(str(nbxtensor.__file__)).resolve().parent
    for up in (here, *here.parents):
        if (up / ".git").exists():
            r = subprocess.run(["git", "-C", str(up), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
            return r.stdout.strip() or None if r.returncode == 0 else None
    try:
        raw = importlib.metadata.distribution("nbxtensor").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return ((json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None) or None


def installed_matches(pin: Pin) -> bool:
    """Whether the imported nbxtensor is the pinned one: the same commit, or the same published version."""
    if pin.commit is not None:
        return installed_commit() == pin.commit
    import nbxtensor
    return str(getattr(nbxtensor, "__version__", "")) == pin.release
