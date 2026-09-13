"""The registry URL honours NEUROBRIX_REGISTRY, and the public name is the fallback.

On 2026-09-12 the public name `neurobrix.es` was unreachable from the hub's own
rack for an evening. It sits behind Cloudflare, and Spanish operators apply a
court order (Juzgado de lo Mercantil no 6 de Barcelona, 18 Dec 2024, in force
through the 2026/27 season) to Cloudflare's shared addresses during football
matches: the name resolved to a blocking device presenting a self-signed
certificate, and every publish failed certificate verification — correctly. It
returns every weekend.

A machine on the hub's network declares its internal entry point and never goes
out to the internet to come back to a box three metres away. A user anywhere
else keeps the public name, which is why it stays the fallback.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_the_registry_is_read_from_the_machine.py
"""
from __future__ import annotations

import importlib
import sys


def _fresh_utils(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("NEUROBRIX_REGISTRY", raising=False)
    else:
        monkeypatch.setenv("NEUROBRIX_REGISTRY", value)
    sys.modules.pop("neurobrix.cli.utils", None)
    return importlib.import_module("neurobrix.cli.utils")


def test_the_public_name_is_the_default(monkeypatch):
    """A user anywhere in the world reaches the hub through it."""
    assert _fresh_utils(monkeypatch, None).REGISTRY_URL == "https://neurobrix.es"


def test_the_machine_declaration_wins(monkeypatch):
    assert (_fresh_utils(monkeypatch, "http://10.0.0.39:3000").REGISTRY_URL
            == "http://10.0.0.39:3000")


def test_an_empty_declaration_is_not_a_registry(monkeypatch):
    """`NEUROBRIX_REGISTRY=` must not turn the hub into an empty string."""
    assert _fresh_utils(monkeypatch, "").REGISTRY_URL == "https://neurobrix.es"
