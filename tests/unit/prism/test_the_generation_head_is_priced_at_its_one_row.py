"""The flow's generation head is priced at the ONE row both engines feed it (`core/flow/autoregressive.py`
`get_logits(hidden[:, -1:, :])`, the Triton strategies' `hidden.select(1, T - 1)`), and a serving session's prefill
chunk is raised while the plan's residents hold (`core/prism/served_prefill.py`).

Before (2026-11-10): the head's `seq_len` symbol was bound by name to the request's prompt rows. Qwen3-Coder-30B-A3B's
151 936-wide head was priced 1.2 MB a prompt row — 1 184 MB between a 128-row and a 1 024-row prefill chunk, for
logits no engine computes — so a served plan could price no chunk above the container's 128-row prompt margin without
giving up a resident expert block, and a 19 931-token turn took 156 passes of the whole graph, 322 s on two
V100-32GB (44 s at the 2 048 rows the plan holds once the head is priced right).

What this test does if the code is wrong:
1. the head's rows are the prompt's (no `head_rows`, or it answers for the language model, or a scoring request —
   whose head does run every position — is priced at one row) — red;
2. the profiler's map of the head's graph binds its sequence symbol to the request's rows — red on main (1024);
3. a served compiled plan keeps the prompt margin's chunk when larger ones hold the plan, raises past the first size
   that moves a resident, or a plan with certified times does not keep its chunk (it is priced once over the cached
   keys, served_prefill's contract since 2026-11-12) — red (on main: no such module);
4. `--set global.score_mode=1` is not read as a scoring request, or `=0` is — red.

No card. Run: python -m pytest tests/unit/prism/test_the_generation_head_is_priced_at_its_one_row.py
"""
import dataclasses
import json

from neurobrix.core.prism.served_prefill import solve_served as SP_solve_served
from tests.unit.prism._pinned_machine import container_root, profile, V100_16GB, pin_no_calibration

MODEL = "TinyLlama-1.1B-Chat-v1.0"


def _flow():
    from neurobrix.core.prism.flow_bindings import FlowBindings
    root = container_root(MODEL)
    return FlowBindings(json.loads((root / "topology.json").read_text()), root, MODEL), root


def test_the_head_takes_one_row_and_only_the_head():
    from neurobrix.core.prism import InputConfig
    fb, _ = _flow()
    assert fb.head_rows("lm_head") == 1
    assert fb.head_rows("lm_head", InputConfig(seq_len=1024)) == 1
    assert fb.head_rows("model") is None
    assert fb.head_rows("lm_head", InputConfig(seq_len=1024, scoring=True)) is None      # its head runs every position


def test_the_profiler_binds_the_heads_sequence_to_one_row():
    from neurobrix.core.prism import InputConfig
    from neurobrix.core.prism.profiler import ActivationProfiler
    fb, root = _flow()
    dag = json.loads((root / "components" / "lm_head" / "graph.json").read_text())
    seq = [s for s, i in dag["symbolic_context"]["symbols"].items() if i["name"] == "seq_len"]
    assert seq, "the head's graph names no seq_len symbol: the test reads nothing"
    prof = ActivationProfiler(dag)
    bound = prof.build_symbol_map(InputConfig(batch_size=1, seq_len=1024, dtype="float16", flow=fb))
    assert [bound[s] for s in seq] == [1] * len(seq), bound
    scored = prof.build_symbol_map(InputConfig(batch_size=1, seq_len=1024, dtype="float16", flow=fb, scoring=True))
    assert [scored[s] for s in seq] == [1024] * len(seq), scored
    model = json.loads((root / "components" / "model" / "graph.json").read_text())
    mseq = [s for s, i in model["symbolic_context"]["symbols"].items() if i["name"] == "seq_len"]
    mbound = ActivationProfiler(model).build_symbol_map(InputConfig(batch_size=1, seq_len=1024, dtype="float16", flow=fb))
    assert mseq and all(mbound[s] == 1024 for s in mseq), mbound         # the generating model keeps the prompt's rows


def _served(monkeypatch, mode):
    from neurobrix.core.prism import InputConfig, PrismSolver
    from neurobrix.nbx import NBXContainer
    pin_no_calibration(monkeypatch)
    fb, root = _flow()
    solver = PrismSolver()
    cfg = InputConfig(batch_size=1, dtype="float16", flow=fb)
    return solver, NBXContainer.load(str(root)), cfg


def test_a_served_compiled_plan_raises_its_chunk_while_the_plan_holds(monkeypatch):
    solver, c, cfg = _served(monkeypatch, "compiled")
    base = solver.solve_smart(c, profile(V100_16GB), cfg, serve_mode=True, mode="compiled")
    plan, used = SP_solve_served(c, profile(V100_16GB), cfg, mode="compiled")
    kv = plan.kv_cache_plan
    assert base.kv_cache_plan.prefill_chunk == 128, base.kv_cache_plan.prefill_chunk_reason   # the prompt margin
    # TinyLlama on a 16 GB card: every size up to the cache's reach holds the plan
    assert kv.prefill_chunk == used.seq_len == 2048 and kv.max_cache_len == 2048, (kv.prefill_chunk, used.seq_len)
    from neurobrix.core.prism.served_prefill import resident_signature, UNTIMED
    assert f"compiled engine keeps {UNTIMED}" in kv.prefill_chunk_reason
    assert resident_signature(plan) == resident_signature(base)
    assert used.flow is cfg.flow and used.prefill_keys == 2048 and dataclasses.replace(used, seq_len=None, prefill_keys=None, prefill_key_dtype=None) == cfg


def test_the_raise_ends_at_the_first_size_that_moves_a_resident(monkeypatch):
    from neurobrix.core.prism import served_prefill as SP
    solver, c, cfg = _served(monkeypatch, "compiled")
    real = SP.resident_signature
    # a resident gives way from 1 024 priced rows on (as Qwen3-Coder's expert block does at 4 096 on two V100-32GB)
    monkeypatch.setattr(SP, "resident_signature",
                        lambda plan: real(plan) + ((plan.kv_cache_plan.prefill_chunk or 0) >= 1024,))
    plan, used = SP_solve_served(c, profile(V100_16GB), cfg, mode="compiled")
    assert plan.kv_cache_plan.prefill_chunk == used.seq_len == 512, (plan.kv_cache_plan.prefill_chunk, used.seq_len)


def test_a_size_no_plan_fits_ends_the_raise(monkeypatch):
    from neurobrix.core.prism import PrismSolver
    solver, c, cfg = _served(monkeypatch, "compiled")
    real = PrismSolver.solve_smart

    def refusing(self, container, prof, input_config=None, **kw):
        if (getattr(input_config, "seq_len", None) or 0) >= 512:
            raise RuntimeError("This model cannot run on this machine")
        return real(self, container, prof, input_config, **kw)
    monkeypatch.setattr(PrismSolver, "solve_smart", refusing)
    plan, used = SP_solve_served(c, profile(V100_16GB), cfg, mode="compiled")
    assert plan.kv_cache_plan.prefill_chunk == used.seq_len == 256


def test_the_solver_says_untimed_with_the_modules_own_phrase(monkeypatch):
    """Reworded, the solver's reason would turn the raise off without a word: both are built from one constant."""
    from neurobrix.core.prism import served_prefill as SP
    monkeypatch.setattr(SP, "UNTIMED", "no timing at all")
    solver, c, cfg = _served(monkeypatch, "compiled")
    plan, used = SP_solve_served(c, profile(V100_16GB), cfg, mode="compiled")
    assert "no timing at all" in plan.kv_cache_plan.prefill_chunk_reason and used.seq_len == 2048


def test_an_engine_with_certified_times_keeps_its_own_choice(monkeypatch):
    solver, c, cfg = _served(monkeypatch, "triton")
    base = solver.solve_smart(c, profile(V100_16GB), cfg, serve_mode=True, mode="triton")
    plan, used = SP_solve_served(c, profile(V100_16GB), cfg, mode="triton")
    kv = plan.kv_cache_plan
    # its own choice kept, priced once at that chunk over the cached keys it attends
    assert (kv.prefill_chunk, kv.prefill_chunk_reason) == (base.kv_cache_plan.prefill_chunk,
                                                           base.kv_cache_plan.prefill_chunk_reason)
    assert (used.seq_len, used.prefill_keys, used.prefill_key_dtype) == (kv.prefill_chunk, kv.max_cache_len, str(kv.dtype))
    assert "no certified kernel times" not in plan.kv_cache_plan.prefill_chunk_reason


def test_a_scoring_request_is_read_from_its_set_flag():
    from types import SimpleNamespace as NS
    from neurobrix.cli.commands.run import scoring_request
    assert scoring_request(NS(set=["global.seed=3", "global.score_mode=1"]))
    assert scoring_request(NS(set=["global.score_mode = true"]))
    assert not scoring_request(NS(set=["global.score_mode=0"])) and not scoring_request(NS(set=["global.score_mode=False"]))
    assert not scoring_request(NS(set=None)) and not scoring_request(NS())
