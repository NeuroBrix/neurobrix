"""The processors' tests run against a private NeuroBrix home, cache and store, and a local host (tests/unit/processors/toy.py)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.processors import toy

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def places(tmp_path, monkeypatch):
    """A private home, cache and store, set in this process and given to the children."""
    env = {"NEUROBRIX_HOME": tmp_path / "home", "NEUROBRIX_CACHE": tmp_path / "cache",
           "NEUROBRIX_STORE": tmp_path / "store"}
    for key, path in env.items():
        path.mkdir()
        monkeypatch.setenv(key, str(path))
    return {k: Path(v) for k, v in env.items()}


def env_of(places, hub: str, hf: str) -> dict:
    """The children's environment: the private places, and the two hosts the engine knows."""
    return {**{k: str(v) for k, v in places.items()}, "NEUROBRIX_REGISTRY": hub, "HF_ENDPOINT": hf}


@pytest.fixture
def host():
    """A local host serving the toy artifact."""
    h = toy.Host({toy.FILE: toy.artifact()})
    yield h
    h.close()


def neurobrix(*argv, env=None, timeout=120):
    """The real command, in a child interpreter, its stdout and stderr captured (a pipe, not a terminal)."""
    child = dict(os.environ, PYTHONPATH=str(REPO / "src"),
                 **(env or {}))
    return subprocess.run([sys.executable, "-m", "neurobrix", *argv], capture_output=True, text=True, env=child,
                          timeout=timeout)
