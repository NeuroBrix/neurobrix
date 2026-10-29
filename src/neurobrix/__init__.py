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

# The engine's tensor library is the one of ITS tree: in a source checkout, the sibling `nbxtensor/src` of this package's
# `src/` comes first on the path, whatever editable install the environment carries (an install points at ONE tree, and a
# process of another tree that loaded it ran its engine against a different library: the R33 guard on d9fc8747, 2026-10-29).
# Installed from a wheel, there is no sibling and the installed dependency is used. A library already imported from
# elsewhere is refused by name: the engine never runs against a foreign one.
import sys as _sys
_sibling = _Path(__file__).absolute().parents[2] / "nbxtensor" / "src"   # the tree it was imported from (links kept)
if (_sibling / "nbxtensor" / "__init__.py").is_file():
    _already = _sys.modules.get("nbxtensor")
    if _already is not None and not _Path(_already.__file__).resolve().is_relative_to(_sibling.resolve()):
        raise ImportError(f"nbxtensor was imported from {_already.__file__} before this engine ({__file__}), whose tree "
                          f"carries its own at {_sibling}: import neurobrix first, or put {_sibling} on the path")
    if str(_sibling) not in _sys.path:
        _sys.path.insert(0, str(_sibling))

# The engine provides its tensor library with what the library does not own (src/neurobrix/_nbxtensor_host.py).
from neurobrix import _nbxtensor_host as _nbxtensor_host
_nbxtensor_host.install()
