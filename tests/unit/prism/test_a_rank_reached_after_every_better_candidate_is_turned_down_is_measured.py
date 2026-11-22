"""The lazy measures are owed by the best physical rank present only (`owed_facts.measure_owed`): a streamed or
host-placed candidate behind a resident plan is never chosen by its time. But the selection loop of `_solve_at_rung`
turns candidates down AFTER that filter (its host side declined, the arena door, a KV cache that does not fit). A model
whose weights fit its card and whose KV cache does not loses every resident candidate there, and a streamed plan wins:
the rank it belongs to is measured and timed when the loop reaches it, and ordered by that time (qa 1231-2)."""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import owed_facts as OF
from neurobrix.core.prism.solver import STRATEGY_RANK, PrismSolver, order_candidates

PROFILE = NS(host_measured=None, devices=[NS(measured={"memory_gbps": 900.0})])
CANDIDATES = [(9.0, "single_gpu", {"lm": "cuda:0"}, None), (5.0, "zero3", {"lm": "zero3:cuda:0"}, None),
              (4.0, "layer_streaming", {"lm": "cuda:0"}, None), (1.0, "cpu_execution", {"lm": "cpu"}, None)]


def _select(monkeypatch, turned_down):
    """The loop of `_solve_at_rung` over `CANDIDATES`, `turned_down` refused as the KV cache refuses: (chosen, reload
    probes asked, groups priced)."""
    probes, priced = [], []
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-abc")
    monkeypatch.setattr(OF, "measure_reload_rates", lambda hw, e, c: probes.append(e) or ["BF16"])
    monkeypatch.setattr(OF, "measure_host_facts", lambda hw: [])
    monkeypatch.setattr("neurobrix.core.prism.loader.load_profile", lambda hw: NS(host_measured={"measured": True}))
    measured = {"layer_streaming": 2.0, "zero3": 3.0}    # what the streamed rank's times are once its rate is measured

    def _priced(group):
        priced.append([c[1] for c in group])
        OF.measure_owed("compiled", group, NS(cache_path="/x"), PROFILE)
        return {c[1]: (measured.get(c[1], 1.0) if probes else "not estimated: no reload rate") for c in group}
    estimates = _priced(order_candidates(CANDIDATES, {}))
    ranked = order_candidates(CANDIDATES, estimates)
    for best in OF.each_rank_priced(ranked, _priced, estimates):
        if best[1] in turned_down:
            continue                                     # "KV cache does not fit"
        return best[1], probes, priced
    return None, probes, priced


def test_the_loop_of_the_solver_walks_the_ranks_priced():
    assert "OF.each_rank_priced(ranked, _priced, self._time_estimates)" in inspect.getsource(PrismSolver._solve_at_rung)
    assert STRATEGY_RANK["single_gpu"] < STRATEGY_RANK["zero3"] == STRATEGY_RANK["layer_streaming"] \
        < STRATEGY_RANK["cpu_execution"]


def test_weights_that_fit_with_a_cache_that_does_not_measure_the_next_rank_and_order_it_by_that_time(monkeypatch):
    chosen, probes, priced = _select(monkeypatch, {"single_gpu"})
    assert probes == ["compiled"], "the streamed rank the loop reached owed its reload rate"
    assert priced[1] == ["zero3", "layer_streaming"] and len(priced) == 2, priced
    assert chosen == "layer_streaming", "ordered by its measured time (2.0 s against 3.0 s), not by its score"


def test_a_resident_plan_that_fits_measures_nothing_behind_it(monkeypatch):
    chosen, probes, priced = _select(monkeypatch, set())
    assert chosen == "single_gpu" and probes == [] and len(priced) == 1
