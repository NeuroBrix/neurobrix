"""The grammar engine (nbxgrammar, its own package and repository: github.com/NeuroBrix/nbxgrammar) for the tests that
need it: the installed package — the wheel neurobrix's own dependency installs on every platform a wheel exists for.
Skipped BY NAME only where it is not installed (a platform without a wheel, or an environment made without the
engine's dependencies); the crate no longer lives in this repository, so nothing is built here."""
import importlib

import pytest


def load():
    try:
        return importlib.import_module("nbxgrammar")
    except ImportError as e:
        pytest.skip(f"nbxgrammar is not installed here (pip install nbxgrammar; no wheel exists for some platforms): {e}")
