"""A served session on the Triton engine with no certified prefill chunk runs its turns in chunks of the rows its plan
PRICES (core/prism/solver.py `_prefill_chunk`), the plan priced at that chunk over the cached keys
(core/prism/served_prefill.py) — never one pass of a turn's rows the plan did not price. Not raised: a chunk is padded
to its full rows, so a larger one slows every shorter turn (measured: raised to 16 384 rows, a short request's first
token took 51 s instead of 4.9 on one V100-32GB).

Before (2026-11-12): the served Qwen3-Coder-30B-A3B on one V100-32GB had no chunk record in the derived census, so its
plan said "one chunk per turn"; the container states no prompt margin, so the turn was priced at 128 rows while the
experts' cache took the rest of the card, and an agent's turn of 2 800 to 3 700 rows ran as one pass and out of memory.

What this test does if the code is wrong (TinyLlama served on the pinned V100-16GB, the census root redirected to an
empty one, so no chunk record exists):
1. the served plan's chunk is None ("one chunk per turn") — red on main;
2. its reason carries the phrase `served_prefill` raises on — red (the raise would pad every turn to its rows);
3. the returned configuration does not price the chunk's own rows over the session's reach — red;
4. a RUN (not served) on the same engine keeps its one pass of its own rows, priced at them — red if it changed;
5. the served chunk's attention over the cached keys routes where it writes no scores: on Volta, whose matrix unit
   runs the flash at the decoder's head dim, the plan states a math budget of 0 (red on main: no such field), and the
   price of that attention at the session's reach is the keys and values it gathers, no scores (red when the price
   keeps the device's math budget: 3 GB of scores at Qwen3-Coder's reach, measured 2026-11-12).

Run: python -m pytest tests/unit/prism/test_a_served_turn_is_never_one_unpriced_pass.py
"""
import json

from tests.unit.prism._pinned_machine import container_root, profile, V100_16GB, pin_no_calibration

MODEL = "TinyLlama-1.1B-Chat-v1.0"


def _served(monkeypatch, tmp_path):
    from neurobrix.core.prism import InputConfig
    from neurobrix.core.prism.flow_bindings import FlowBindings
    from neurobrix.core.prism.request_config import served_plan_fields
    from neurobrix.core.prism.served_prefill import solve_served
    from neurobrix.kernels import census_table as T
    from neurobrix.nbx import NBXContainer
    pin_no_calibration(monkeypatch)
    monkeypatch.setattr(T, "ROOT", tmp_path)            # an empty census: no chunk record for any model
    root = container_root(MODEL)
    c = NBXContainer.load(str(root))
    man = c.get_manifest() or {}
    defaults = json.loads((root / "runtime" / "defaults.json").read_text())
    ic = InputConfig(**served_plan_fields(man.get("family"), defaults, man), height=None, width=None, vae_scale=None,
                     flow=FlowBindings(json.loads((root / "topology.json").read_text()), root, man.get("model_name")))
    return solve_served(c, profile(V100_16GB), ic, mode="triton")


def test_a_served_turn_runs_in_priced_chunks(monkeypatch, tmp_path):
    from neurobrix.core.prism.served_prefill import UNTIMED
    plan, cfg = _served(monkeypatch, tmp_path)
    kv = plan.kv_cache_plan
    assert kv.prefill_chunk, kv.prefill_chunk_reason
    assert "holds no prefill-chunk record" in kv.prefill_chunk_reason and UNTIMED not in kv.prefill_chunk_reason
    assert int(cfg.seq_len) == int(kv.prefill_chunk), (cfg.seq_len, kv.prefill_chunk)
    assert int(cfg.prefill_keys) == int(kv.max_cache_len) and int(kv.prefill_chunk) <= int(kv.max_cache_len)


def test_the_served_chunk_routes_where_it_writes_no_scores(monkeypatch, tmp_path):
    plan, cfg = _served(monkeypatch, tmp_path)
    kv = plan.kv_cache_plan
    assert kv.chunk_scores_budget_bytes == 0, kv.chunk_scores_budget_bytes
    assert plan.to_dict("x")["kv_cache_plan"]["chunk_scores_budget_bytes"] == 0


def test_the_flash_routed_chunk_is_priced_at_its_gathered_keys_only():
    from tests.unit.prism._pinned_machine import profile, V100_16GB
    from neurobrix.core.prism import PrismSolver
    s = PrismSolver()
    s._mode = "triton"
    prof = profile(V100_16GB)
    base = s._attention_scratch(prof, "float16")
    b, h, C, D, keys = 1, 32, 128, 128, 65536
    assert base(b, h, C, keys, D=D, Dv=D, Hk=h, mask=True) > 2 ** 30          # the device's math route: over 1 GiB
    s._kv_decoder = "model"
    s._input_config = type("IC", (), {"prefill_keys": keys})()
    served = s._attention_scratch(prof, "float16", "model")
    assert served(b, h, C, C, D=D, Dv=D, Hk=h, mask=True) == b * h * keys * 2 * D * 2   # the gathered K and V, fp16


def test_a_run_keeps_its_own_pass(monkeypatch, tmp_path):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism import PrismSolver
    from neurobrix.kernels import census_table as T
    from neurobrix.nbx import NBXContainer
    pin_no_calibration(monkeypatch)
    monkeypatch.setattr(T, "ROOT", tmp_path)
    root = container_root(MODEL)
    c = NBXContainer.load(str(root))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "hi", "--max-tokens", "32", "--triton"])
    kv = PrismSolver().solve(c, profile(V100_16GB), request_input_config(args, man, man.get("family"), c.cache_path),
                             mode="triton").kv_cache_plan
    assert kv.prefill_chunk is None and "holds no prefill-chunk record" in kv.prefill_chunk_reason
    assert kv.chunk_scores_budget_bytes is None
