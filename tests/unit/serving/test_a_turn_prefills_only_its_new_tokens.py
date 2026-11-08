"""A serving session keeps its KV cache between turns, and turn 2 served from it is the same bytes as the two turns
prefilled together in the same chunks — on the Triton engine and on the compiled engine (docs/design/prism_kv.md
§5bis, scope A; the owner's option C, 2026-11-08).

Before: every chat turn re-prefilled the whole conversation (`serving/session.py`: "V1 Strategy: Re-prefill entire
history each turn"); the Triton cache was cleared at each request, the compiled one built anew per request.

Device test: TinyLlama-1.1B-Chat-v1.0 from the engine cache, one child process per (engine, chunk) running
`session_kv_driver.py` on the card the caller made visible (CUDA_VISIBLE_DEVICES), with the machine's own profile and
certified kernels only. Each request decodes one token, so a cache holds exactly what was prefilled.

(a) The rows the language model's executor RAN for turn 2 are its new tokens only — counted at the executor's `run`,
    outside the session code that decides them: len(full) - len(A) with no chunk planned; the whole chunks of C rows
    covering them with C planned.
(b) With the plan's chunk C (KVCachePlan.prefill_chunk, a multiple of the page), "A then B" from the session and
    "A+B" in one request give the same sha256 for every layer's K and V over the conversation's positions and for the
    turn's first-token logits. C = 320 is the chunk the 2026-11-08 experiment chose for TinyLlama on this rack from
    the certified times (campaign 2026_11_08_chunked_prefill, choose_C_tinyllama.txt), five 64-position pages.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODEL = "TinyLlama-1.1B-Chat-v1.0"
CHUNK = 320
REPO = HERE.parents[2]


def _cache_root() -> Path:
    sys.path.insert(0, str(REPO / "tools"))
    import atelier
    return Path(atelier.root("cache"))


def _tmp_root() -> Path:
    sys.path.insert(0, str(REPO / "tools"))
    import atelier
    return Path(atelier.root("tmp"))


def _run(mode: str, chunk: int) -> dict:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("device test: name the card with CUDA_VISIBLE_DEVICES")
    if not (_cache_root() / MODEL).exists():
        pytest.skip(f"{MODEL} is not in the engine cache")
    tmp = _tmp_root() / f"session_kv_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / f"{mode}_{chunk}.json"
    env = dict(os.environ, PYTHONNOUSERSITE="1", NBX_AUTOTUNE_CERTIFIED_ONLY="1", NBX_SESSION_KV_TMP=str(tmp),
               PYTHONPATH=f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}")
    p = subprocess.run([sys.executable, str(HERE / "session_kv_driver.py"), "--model", MODEL, "--mode", mode,
                        "--chunk", str(chunk), "--conv", str(HERE / "data" / "two_turns_garden.json"),
                        "--out", str(out)], env=env, capture_output=True, text=True, timeout=1800, cwd=str(tmp))
    if p.returncode != 0:
        raise AssertionError(f"driver {mode} chunk {chunk} rc {p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}")
    return json.loads(out.read_text())


_RESULTS: dict = {}


def _result(mode: str, chunk: int) -> dict:
    if (mode, chunk) not in _RESULTS:
        _RESULTS[(mode, chunk)] = _run(mode, chunk)
    return _RESULTS[(mode, chunk)]


@pytest.mark.parametrize("mode", ["triton", "compiled"])
def test_turn_two_prefills_only_its_new_tokens(mode):
    r = _result(mode, 0)
    new = r["len_full"] - r["len_A"]
    assert 0 < r["len_A"] < r["len_full"]
    t2 = r["split_turn2"]
    assert t2["ledger"]["matched"] == r["len_A"], t2["ledger"]
    assert t2["rows_run"] == new, f"turn 2 ran {t2['rows_run']} rows for {new} new tokens ({r['len_full']} in all)"
    assert r["whole"]["rows_run"] == r["len_full"], "the fresh session prefilled less than its conversation"
    if mode == "triton":
        assert not r["torch_loaded"], "R33: the Triton session's process loaded torch"


@pytest.mark.parametrize("mode", ["triton", "compiled"])
def test_a_then_b_is_a_plus_b_in_the_same_chunks(mode):
    r = _result(mode, CHUNK)
    new = r["len_full"] - r["len_A"]
    assert r["split_turn2"]["rows_run"] == -(-new // CHUNK) * CHUNK
    diff = [i for i, (s, w) in enumerate(zip(r["split_kv"], r["whole_kv"])) if s != w]
    assert len(r["split_kv"]) == len(r["whole_kv"]) and not diff, f"layers whose K/V differ: {diff}"
    assert r["split_turn2"]["logits_sha"] == r["whole"]["logits_sha"], (r["split_turn2"], r["whole"])
