"""A harness that compiles and does not own its cache refuses to start.

Six times in one day a measurement returned a zero that was the cache
answering: a census over kernels already stashed, a MEPT probe whose arms both
reached the lowerer zero times, four tests that began reporting no lowerer was
reached at all. Every layer of this stack caches compiled artefacts keyed by a
hash of the source, so any measurement that changes code and shares a cache
reads the old artefact and reports on it.

Six is where a discipline becomes a defect. This is the structural form: put
the harness in a state where it cannot do the harm, rather than measure
afterwards that it did not -- the same shape as the campaign refusing without
a frozen tree.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_owned_compilation_cache.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from check_measurement_environment import (                      # noqa: E402
    _CACHE_VARS,
    enforce_owned_cache,
    owned_cache_env,
    owned_cache_problems,
)


def test_an_unset_cache_is_a_problem():
    assert len(owned_cache_problems({})) == len(_CACHE_VARS), (
        "both cache variables must be reported, not the first one only: a "
        "harness that owns triton's cache and shares the MSL stash still reads "
        "a stale kernel")


def test_a_private_directory_is_accepted(tmp_path):
    assert owned_cache_problems(owned_cache_env(tmp_path)) == []


def test_a_path_inside_the_shared_cache_is_refused():
    """The subtle one, and the reason this is not a simple is-it-set check.

    `TRITON_CACHE_DIR=~/.triton/cache/run42` looks owned and reads the shared
    cache's contents, because triton keys on the hash below whatever root it
    is given.
    """
    shared = Path.home() / ".triton" / "cache" / "run42"
    problems = owned_cache_problems({
        "TRITON_CACHE_DIR": str(shared),
        "TRITON_MSL_CACHE_DIR": str(Path.home() / ".cache" / "triton_msl"),
        "NEUROBRIX_REPLAY_CACHE": str(Path.home() / ".neurobrix" / "replay_cache"),
    })
    assert len(problems) == 3, f"expected all three refused, got {problems}"


def test_the_replay_cache_is_one_of_the_layers():
    """The layer the first version of this guard missed, and it is the one
    that answers for the screen.

    A measurement taken after repairing the oracle replayed a refusal recorded
    before it, printed no `AUTOTUNE_SCREEN` line at all, and read exactly like
    a repair that had not worked. A guard over some of the caches is a guard
    over none, because the one it misses is the one that answers.
    """
    assert "NEUROBRIX_REPLAY_CACHE" in _CACHE_VARS
    assert "NEUROBRIX_REPLAY_CACHE" in owned_cache_env(Path("/tmp/x"))


def test_the_refusal_names_the_remedy_and_the_remedy_exists():
    """A refusal that points at a flag which does not exist is vacuous."""
    with pytest.raises(RuntimeError) as exc:
        enforce_owned_cache("a test")
    assert "--cache-env" in str(exc.value)
    tool = Path(__file__).resolve().parents[3] / "tools" / "check_measurement_environment.py"
    assert "--cache-env" in tool.read_text(), (
        "the refusal names `--cache-env`; the tool must implement it")


def test_enforce_passes_once_the_cache_is_owned(tmp_path, monkeypatch):
    """Both directions: the guard must also let a correct harness through."""
    for var, value in owned_cache_env(tmp_path).items():
        monkeypatch.setenv(var, value)
    enforce_owned_cache("a test")          # must not raise
