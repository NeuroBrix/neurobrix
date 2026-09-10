"""`neurobrix coverage` reads the container's own contract, and says zero honestly.

The command exists because a test that exercises nothing is green. It would be
a poor joke if the command itself answered from a key that had stopped being
true — the exact defect it was built to expose — so its scan key is pinned here
against the NBX format contract (R18: `graph.json` op records carry `op_type`).

Two behaviours are pinned, and they are the two that matter:

  * a symbol NO container carries returns zero AND says what zero means. A
    silent empty list reads as "nothing to worry about"; the whole point is
    that it means "no model run can validate this".
  * a symbol every container carries is found through both spellings, with and
    without the `aten::` prefix, because a census nobody can address by the
    name they have in their hand is a census nobody runs.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_coverage_reads_the_container_contract.py
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from neurobrix.cli.commands import coverage


def _container(root: Path, name: str, ops: list[str]) -> None:
    comp = root / name / "components" / "backbone"
    comp.mkdir(parents=True)
    (root / name / "manifest.json").write_text(json.dumps({"family": "llm"}))
    (comp / "graph.json").write_text(json.dumps({
        "component_name": "backbone",
        "ops": {f"{op}::{i}": {"op_type": op} for i, op in enumerate(ops)},
    }))


def _args(**kw):
    base = dict(symbol=None, rarest=0, unreached=False, field=None)
    base.update(kw)
    return Namespace(**base)


def test_the_scan_key_is_the_one_the_container_writes(tmp_path, monkeypatch, capsys):
    """If `op_type` ever stopped being the key, every answer would be zero and
    every zero would read as a real coverage gap."""
    _container(tmp_path, "modelA", ["aten::tril", "aten::mm"])
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path))

    assert coverage.cmd_coverage(_args(symbol="aten::tril")) == 0
    out = capsys.readouterr().out
    assert "1 of 1" in out and "modelA" in out


def test_a_bare_op_name_finds_the_same_thing_as_the_prefixed_one(tmp_path, monkeypatch, capsys):
    _container(tmp_path, "modelA", ["aten::tril"])
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path))

    coverage.cmd_coverage(_args(symbol="tril"))
    bare = capsys.readouterr().out
    coverage.cmd_coverage(_args(symbol="aten::tril"))
    assert capsys.readouterr().out == bare


def test_zero_says_what_zero_means(tmp_path, monkeypatch, capsys):
    """The control on this command's own honesty: an empty answer must carry
    its consequence, or it reads as reassurance."""
    _container(tmp_path, "modelA", ["aten::mm"])
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path))

    coverage.cmd_coverage(_args(symbol="aten::var"))
    out = capsys.readouterr().out
    assert "0 of 1" in out
    assert "validates nothing" in out, (
        "a zero that does not state its consequence is read as 'fine'")


def test_an_empty_cache_refuses_instead_of_answering_zero(tmp_path, monkeypatch):
    """Every answer would be zero, and every zero would be a lie of the same
    shape as the defect this command hunts."""
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path))
    assert coverage.cmd_coverage(_args(symbol="aten::tril")) == 1
