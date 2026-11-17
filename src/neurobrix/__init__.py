"""NeuroBrix - Universal Deep Learning Inference Engine."""

from pathlib import Path as _Path
import re as _re

# Single source of truth: pyproject.toml version field.
# Works in both dev mode (PYTHONPATH=src) and pip-installed mode.
_pyproject = _Path(__file__).resolve().parents[2] / "pyproject.toml"
if _pyproject.exists():
    _m = _re.search(r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), _re.MULTILINE)
    __version__ = _m.group(1) if _m else "0.0.0-dev"
else:
    # pip-installed: pyproject.toml not available, use package metadata
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("neurobrix")
    except Exception:
        __version__ = "0.0.0-dev"

# The engine's tensor library is the installed `nbxtensor` package, the release `pyproject.toml` pins (its own repository).
# The library's replay cache follows the engine's home: nbxtensor reads NEUROBRIX_REPLAY_CACHE, else ~/.neurobrix; a
# configured NEUROBRIX_HOME is told to it before it is imported (the Mac's new-user run, 2026-11-07)
import os as _os
if _os.environ.get("NEUROBRIX_HOME") and not _os.environ.get("NEUROBRIX_REPLAY_CACHE"):
    _os.environ["NEUROBRIX_REPLAY_CACHE"] = _os.path.join(_os.path.expanduser(_os.environ["NEUROBRIX_HOME"]), "replay_cache")

# The engine provides its tensor library with what the library does not own (src/neurobrix/_nbxtensor_host.py).
from neurobrix import _nbxtensor_host as _nbxtensor_host
_nbxtensor_host.install()
