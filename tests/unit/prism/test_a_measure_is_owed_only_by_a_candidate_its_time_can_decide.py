"""The lazy measures of the ruling of 22/11/2026 17:43 (lines 4 and 5) are owed only by a candidate whose time can
decide the plan. Prism orders candidates by physical rank first and by the estimated time only inside a rank
(`order_candidates`), and every language model that fits its card is also a layer_streaming candidate (a lower rank):
asked for every streamed candidate in the running, the reload probe ran at every first launch of every model anyway (a
fresh Mac home: about 50 s per stored dtype, on NFS). A streamed or host-placed candidate behind a plan of a better
rank owes nothing; the same candidate in the best rank present owes its measure.

What each test would do if the code were wrong: a probe behind a resident plan of a better rank (red); no probe when
the streamed candidate is in the best rank present (red); the host measured for a host-placed candidate behind a
better rank (red)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import owed_facts as OF
from neurobrix.core.prism.solver import STRATEGY_RANK


@pytest.fixture
def asked(monkeypatch):
    calls = []
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-abc")
    monkeypatch.setattr(OF, "measure_reload_rates", lambda hw, e, c: calls.append("reload") or [])
    monkeypatch.setattr(OF, "measure_host_facts", lambda hw: calls.append("host") or [])
    return calls


def _owed(ranked):
    OF.measure_owed("compiled", [(0.0, n, a, None) for n, a in ranked],
                                    NS(cache_path="/x"), NS(host_measured=None, devices=[NS(measured={"memory_gbps": 900.0})]))


def test_a_streamed_or_host_placed_candidate_behind_a_better_rank_owes_nothing(asked):
    assert STRATEGY_RANK["single_gpu"] < STRATEGY_RANK["layer_streaming"] < STRATEGY_RANK["cpu_execution"]
    _owed([("single_gpu", {"lm": "cuda:0"}), ("layer_streaming", {"lm": "cuda:0"}), ("cpu_execution", {"lm": "cpu"})])
    assert asked == []


def test_the_same_candidates_in_the_best_rank_present_owe_their_measures(asked):
    _owed([("layer_streaming", {"lm": "cuda:0"}), ("lazy_sequential", {"lm": "cuda:0", "vae": "cpu"}),
           ("cpu_execution", {"lm": "cpu"})])
    assert asked == ["reload", "host"]
