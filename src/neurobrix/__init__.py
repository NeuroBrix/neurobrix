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
