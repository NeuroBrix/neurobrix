"""pytest plugin: record every device-file key the LIBRARY reads while a test suite runs (the dynamic half of the
device-key census, `device_key_census.py`). Activated with `-p nbxtensor.tools.device_key_trace` (or the file on
the path) and NBX_DEVICE_KEY_TRACE=<file>: each read `section.key` (or a top-level key) is appended once."""
from __future__ import annotations

import os

_SEEN: set = set()


def _note(path: str) -> None:
    out = os.environ.get("NBX_DEVICE_KEY_TRACE")
    if not out or path in _SEEN:
        return
    _SEEN.add(path)
    with open(out, "a") as f:
        f.write(path + "\n")


class _Tracked(dict):
    """A vendor file (or a section of it) that records the keys read from it."""

    def __init__(self, data, prefix=""):
        super().__init__(data)
        self._prefix = prefix

    def _wrap(self, key, value):
        path = f"{self._prefix}{key}"
        _note(path)
        return _Tracked(value, path + ".") if isinstance(value, dict) else value

    def get(self, key, default=None):
        if key in self:
            return self._wrap(key, dict.__getitem__(self, key))
        _note(f"{self._prefix}{key}")
        return default

    def __getitem__(self, key):
        return self._wrap(key, dict.__getitem__(self, key))


def pytest_configure(config):
    from nbxtensor import host
    real_need = host.need

    def need(name):
        fn = real_need(name)
        if name == "vendor_config":
            return lambda *a, **k: _Tracked(fn(*a, **k))
        if name == "vendor_param":
            def param(vendor, architecture, section, key, default):
                _note(f"{section}.{key}")
                return fn(vendor, architecture, section, key, default)
            return param
        return fn

    host.need = need
