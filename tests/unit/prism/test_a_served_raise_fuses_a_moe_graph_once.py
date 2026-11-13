"""A served session's prefill-chunk raise is ONE planning: its every solve reads the same fused copy of a MoE graph.

Each solve of the raise (`served_prefill.raise_while_the_plan_holds`) used to copy and fuse the graph again, the memo
dropped at the end of every `solve_smart`: Qwen3-Coder-30B-A3B served --compiled spent 10+ minutes of one core before
its card was touched (py-spy, 2026-11-13)."""
from types import SimpleNamespace

from neurobrix.core.prism import served_prefill
from neurobrix.core.prism.solver import PrismSolver


def test_the_served_solver_keeps_its_fused_graphs_across_the_raise(monkeypatch):
    seen = []

    def fake_solve_smart(self, container, profile, cfg, serve_mode=False, mode="compiled"):
        memo = self.__dict__.setdefault("_declared_fusions", {})
        seen.append((self, len(memo)))
        memo[len(memo)] = object()          # what a solve's fusion would add
        return SimpleNamespace(kv_cache_plan=None)

    monkeypatch.setattr(PrismSolver, "solve_smart", fake_solve_smart)
    served_prefill.solve_served(object(), object(), SimpleNamespace(seq_len=1))
    assert seen and seen[0][0].keeps_fusions is True


def test_a_plain_solver_drops_its_fused_graphs_after_each_planning():
    assert getattr(PrismSolver(), "keeps_fusions", False) is False


def test_a_kept_fusion_outlives_the_end_of_a_planning(monkeypatch):
    import neurobrix.core.runtime.graph.moe_fusion as mf
    calls = []
    monkeypatch.setattr(mf, "detect_and_fuse_moe", lambda g, *a, **k: calls.append(1) or g)
    graph = {"ops": {}, "tensors": {}}
    kept = PrismSolver()
    kept.keeps_fusions = True
    for solver, fused_each_time in ((kept, 1), (PrismSolver(), 2)):
        calls.clear()
        for _ in range(2):
            solver._fused_as_declared(graph, True)
            solver.__dict__.pop("_declared_fusions", None)     # what solve_smart's finally does after each solve
        assert len(calls) == fused_each_time
