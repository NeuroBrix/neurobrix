"""Prism chooses a serving session's prefill chunk C from the certified times of each candidate's census keys
(docs/design/prism_kv.md §5bis; core/prism/prefill_chunk.py), never a constant; None, its reason stated, when no
candidate's keys are all certified.

Before (A, 2026-11-08): `KVCachePlan.prefill_chunk` was always None — "Prism chooses no C yet" — and the exactness
of "A then B" against "A+B" held only when a test set the field by hand.

What this test does if the code is wrong:
1. the rule on a record: a candidate with an uncertified key is chosen anyway, the dearer one is chosen, a tie is
   not broken to the smaller C, or None comes without the key that refused each candidate — red;
2. TinyLlama planned for the Triton engine on the pinned 16 GB card, with a census chunk record in a redirected
   census root and the certified times stubbed: the plan's C is the candidate the stub makes cheapest, and with the
   record's container hash stale the plan says no record holds this graph; the compiled engine states that it keeps
   no certified times (red on main: no chooser, the field always None, no reason field).

Run: python -m pytest tests/unit/prism/test_a_session_chunk_is_chosen_from_certified_times.py
"""
import json

from tests.unit.prism._pinned_machine import container_root, profile, V100_16GB, pin_no_calibration

MODEL = "TinyLlama-1.1B-Chat-v1.0"


def _record(costs_by_rows):
    """A chunk record: per candidate one graph key and one attention key per key length 1..4 pages."""
    cands = []
    for C in costs_by_rows:
        cands.append({"rows": C, "graph": [["k.mm", f"({C},)", "fp32", 2]],
                      "attention": [[tk, "k.sm", f"({C}, {tk})", "fp32", 1] for tk in range(C, 257, 64)]})
    return {"candidates": cands}


def test_the_cheapest_certified_candidate_is_chosen():
    from neurobrix.core.prism import prefill_chunk as PC
    times = {"(64,)": 1.0, "(128,)": 1.5}

    def served(kernel, dtype, key):
        if kernel == "k.mm":
            return times.get(key)
        return 0.1
    # a turn of 128 rows: C=64 costs 2 chunks x 2 ops x 1.0 + attention at 64 and 128; C=128 one chunk x 2 x 1.5 + one
    C, why = PC.choose(_record([64, 128]), [64, 128], 128, served)
    assert C == 128 and "C=128" in why and "C=64" in why, why          # 3.1 ms against 4.2 ms
    times["(128,)"] = 3.0
    assert PC.choose(_record([64, 128]), [64, 128], 128, served)[0] == 64
    del times["(64,)"]                                                  # C=64's graph key no longer certified
    C, why = PC.choose(_record([64, 128]), [64, 128], 128, served)
    assert C == 128 and "C=64: 1 key(s) not certified, first mm (64,)" in why, why
    del times["(128,)"]
    C, why = PC.choose(_record([64, 128]), [64, 128], 128, served)
    assert C is None and why.startswith("one chunk per turn: no candidate's keys are all certified"), why
    assert "C=128: 1 key(s) not certified" in why
    assert PC.choose(None, [64], 128, served) == (None, "one chunk per turn: the derived census holds no "
                                                        "prefill-chunk record for this model")


def test_a_tie_takes_the_smaller_chunk():
    from neurobrix.core.prism import prefill_chunk as PC
    rec = {"candidates": [{"rows": 64, "graph": [["k", "a", "fp32", 1]], "attention": []},
                          {"rows": 128, "graph": [["k", "b", "fp32", 1]], "attention": []}]}
    assert PC.choose(rec, [64, 128], 128, lambda k, d, key: 2.0 if key == "b" else 1.0)[0] == 64


def _plan(monkeypatch, tmp_path, mode, served, container=None):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism import PrismSolver, prefill_chunk as PC
    from neurobrix.kernels import census_table as T
    from neurobrix.nbx import NBXContainer
    pin_no_calibration(monkeypatch)
    root = container_root(MODEL)
    rec = dict(_record([64, 128]), model=root.name, mode="triton", container=container or T.container_sha(root))
    monkeypatch.setattr(T, "ROOT", tmp_path)
    path = T.table_path("nvidia", "volta", 16)
    path.parent.mkdir(parents=True)
    T.chunks_path(path).write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(PC, "certified_times", lambda vendor, arch, cls: served)
    c = NBXContainer.load(str(root))
    man = c.get_manifest() or {}
    flag = "--triton" if mode == "triton" else "--compiled"
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "hi", "--max-tokens", "32", flag])
    return PrismSolver().solve(c, profile(V100_16GB), request_input_config(args, man, man.get("family"), c.cache_path),
                               mode=mode).kv_cache_plan


def test_the_plan_carries_the_chosen_chunk_and_its_reason(monkeypatch, tmp_path):
    served = lambda kernel, dtype, key: {"(64,)": 1.0, "(128,)": 1.5}.get(key, 0.1)
    kv = _plan(monkeypatch, tmp_path, "triton", served)
    assert kv.prefill_chunk == 128, (kv.prefill_chunk, kv.prefill_chunk_reason)
    assert kv.prefill_chunk_reason.startswith("C=128: the cheapest certified turn of 128 rows"), kv.prefill_chunk_reason


def test_a_record_of_another_graph_is_none_and_said(monkeypatch, tmp_path):
    kv = _plan(monkeypatch, tmp_path, "triton", lambda *a: 1.0, container="0000000000000000")
    assert kv.prefill_chunk is None and "holds no prefill-chunk record" in kv.prefill_chunk_reason


def test_the_compiled_engine_states_why_it_plans_one_chunk(monkeypatch, tmp_path):
    kv = _plan(monkeypatch, tmp_path, "compiled", lambda *a: 1.0)
    assert kv.prefill_chunk is None and "compiled engine keeps no certified kernel times" in kv.prefill_chunk_reason
