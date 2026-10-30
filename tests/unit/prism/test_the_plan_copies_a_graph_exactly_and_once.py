"""The plan's copies of a graph (the MoE-fused sizing graph, the branch-normalized cut graph) are exact and cheap.

copy.deepcopy dispatched per object: 13.0 s per copy of Qwen3-30B-A3B's graph (115 419 ops, 140 480 tensors), two
copies per plan, on one core; graph_copy makes the same copy in 7.1 s (2026-10-30). The copy must stay exact and
independent — a plan-time rewrite leaking into the container's graph is the defect normalize_for_branch exists to
prevent — and a graph holding anything but plain data is refused, never copied some other way.

What would this file do if the copy shared an entry with its source, or lost a type? The independence and equality
checks fail; if a non-data value were copied silently, the refusal check fails."""
from __future__ import annotations

import copy

import pytest

from neurobrix.core.optim.passes.normalize import graph_copy


def _graph():
    shape = ["s0", 64]
    return {"tensors": {"x": {"shape": shape, "dtype": "float16"}, "y": {"shape": shape, "dtype": "float16"}},
            "ops": {"aten.mm::0": {"op_type": "aten::mm", "input_tensor_ids": ["x"], "output_tensor_ids": ["y"],
                                   "attributes": {"alpha": 1.0, "dims": (0, 1), "flag": None, "on": True}}},
            "execution_order": ["aten.mm::0"], 7: "an int key"}


def test_the_copy_is_exact():
    g = _graph()
    c = graph_copy(g)
    assert c == copy.deepcopy(g) == g
    assert isinstance(c["ops"]["aten.mm::0"]["attributes"]["dims"], tuple), "a tuple came back as something else"
    assert c["tensors"]["x"]["shape"] is c["tensors"]["y"]["shape"], "a shared reference was not kept"


def test_the_copy_shares_nothing_with_its_source():
    g = _graph()
    c = graph_copy(g)
    c["ops"]["aten.mm::0"]["input_tensor_ids"].append("z")
    c["tensors"]["x"]["shape"][1] = 128
    c["ops"]["aten.mm::0"]["attributes"]["alpha"] = 2.0
    assert g == _graph(), "a rewrite of the copy reached the source graph"


def test_a_graph_holding_anything_but_plain_data_is_refused():
    g = _graph()
    g["ops"]["aten.mm::0"]["attributes"]["fn"] = object()
    with pytest.raises(TypeError, match="not plain data"):
        graph_copy(g)


def test_the_fused_graphs_end_with_the_planning():
    """The solver is held by the run that asked for its plan (cli run, serving): a fused graph kept on it would sit in
    host memory for the whole run. solve_smart lets every one go, whether its plan came back or was refused."""
    from types import SimpleNamespace

    from neurobrix.core.prism.solver import PrismSolver

    s = PrismSolver.__new__(PrismSolver)

    def _solve(*a, **k):
        s.__dict__.setdefault("_declared_fusions", {})[1] = ("graph", True, "fused")
        return SimpleNamespace(execution_door=None)

    s.solve = _solve
    s.solve_smart(None, None)
    assert "_declared_fusions" not in s.__dict__, "a fused graph outlived the planning"

    def _refuse(*a, **k):
        s.__dict__.setdefault("_declared_fusions", {})[1] = ("graph", True, "fused")
        raise RuntimeError("no plan fits")

    s.solve = _refuse
    with pytest.raises(RuntimeError):
        s.solve_smart(None, None)
    assert "_declared_fusions" not in s.__dict__, "a refused planning kept its fused graph"
