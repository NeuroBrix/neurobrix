"""The registry commands use ONE http client, and it is the one that carries CAs.

`neurobrix hub` reached for `urllib.request.urlopen` while every other registry
call in the same file used `requests`. urllib verifies against OpenSSL's default
CA file, and on a python.org macOS install that file does not exist until the
user runs `Install Certificates.command`. Measured 2026-09-18 on an M4 Pro:

    ssl default verify paths : .../Python.framework/.../etc/openssl/cert.pem
    that file exists         : False
    plain urlopen            : CERTIFICATE_VERIFY_FAILED
    requests (certifi)       : HTTP 200

so `hub` printed "Cannot connect to registry" on a machine whose network was
fine and whose registry answered `import` perfectly well. The failure named the
wrong thing, which is the part worth preventing: a user reads that and checks
their firewall.

This is structural rather than a network test — it reads the AST, so it needs no
registry and cannot be defeated by a comment or a docstring mentioning urllib.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[3] / "src/neurobrix/cli/commands/registry.py"


def _imported_names(tree: ast.AST) -> set[str]:
    """Every module actually IMPORTED, at any scope. Not text, not comments."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_registry_module_does_not_import_urllib_request():
    """`urllib.parse` is fine — it builds query strings and touches no socket.
    `urllib.request` is the one that opens a connection against a CA file that
    may not be there."""
    tree = ast.parse(REGISTRY.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.startswith("urllib.request")]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("urllib.request"):
                offenders.append(node.module)
    assert not offenders, (
        f"{REGISTRY.name} imports {offenders}. Registry calls go through "
        f"`requests`, which carries certifi; urllib.request verifies against an "
        f"OpenSSL CA file that does not exist on a stock macOS python.org "
        f"install, and reports the resulting TLS failure as a connectivity one.")


def test_no_call_reaches_urlopen():
    """Belt and braces: even reached via an alias, `urlopen` must not be called."""
    tree = ast.parse(REGISTRY.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "urlopen":
                calls.append(f.id)
            elif isinstance(f, ast.Attribute) and f.attr == "urlopen":
                calls.append(f.attr)
    assert not calls, f"{REGISTRY.name} calls urlopen {len(calls)} time(s)"


def test_requests_is_what_it_uses():
    """The positive half — a file that imported neither would pass the two
    tests above while being unable to talk to anything."""
    assert "requests" in _imported_names(ast.parse(REGISTRY.read_text(encoding="utf-8"))), (
        "registry.py imports neither urllib.request nor requests; it cannot "
        "reach the registry at all")
