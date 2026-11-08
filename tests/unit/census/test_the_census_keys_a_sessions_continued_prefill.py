"""The derived census keys a serving session's CONTINUED prefill (docs/design/prism_kv.md §5bis): a turn's chunk
attends over the cache's keys gathered as whole pages with an explicit 0/-inf mask (triton/kv_cache.py
`_prefill_over_pages`), keyed through the launch-key functions the wrappers use (`_op_launches`, decode_kv
`continued`), and records each prefill-chunk candidate's keys for Prism (`_session_chunks`).

Before (2026-11-08): `neurobrix run --turns` on TinyLlama, Triton, certified-only, stopped at turn 2 on
`baddbmm_kernel fp32 (M_BUCKET=32 N_BUCKET=192 K_BUCKET=64 ...)` — 28 new rows over 192 whole-page keys — a key the
census never formed: it keyed a prefill's attention only where its query rows equal its keys.

What this test does if the code is wrong (TinyLlama, the 16 GB profile, its top rung, derived — nothing runs):
1. no chunk planned: the candidates' continuation is not derived — a 64-row chunk's softmax over 192 whole-page keys
   absent — or no chunk record names the candidates (whole pages up to the session's priced prefill rows) with their
   graph and their attention at every whole-page key length up to the reach — red (red on main: no such key, no
   record). A turn running its own 28 rows in one chunk is NOT keyed (the crossed grid of every turn length with every
   key length is not derived): its turn-2 key stays absent, said;
2. the plan's chunk C = 128: the continuation is keyed at 128 rows only — the 64-row key absent, the 128-row softmax
   over 192 keys present.

Run: python -m pytest tests/unit/census/test_the_census_keys_a_sessions_continued_prefill.py
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODEL = "TinyLlama-1.1B-Chat-v1.0"
HW = "default-ff6008b7"
TURN2 = ("nbxtensor.ops.baddbmm_op.baddbmm_kernel", "(32, 192, 64, True, False, False, 'fp32', 'fp32', 'fp32', 'fp32')")
# at 128 rows the scores product takes the matrix unit (no autotune key); its softmax over 192 keys (a next power of
# two of 256 columns, 32 heads x 128 rows) is the keyed launch
AT128 = ("nbxtensor.ops.softmax.softmax_forward_kernel", "(4096, 256, False, 'fp32', 'fp32')")
AT64 = ("nbxtensor.ops.softmax.softmax_forward_kernel", "(2048, 256, False, 'fp32', 'fp32')")


def _derive(monkeypatch, chunk=None):
    sys.path.insert(0, str(REPO / "tools"))
    import certified_census as CC
    import derived_census as D
    if not (D.CACHE / MODEL / "manifest.json").exists():
        pytest.skip(f"{MODEL} is not in the engine cache")
    try:
        from neurobrix.core.prism.loader import load_profile
        load_profile(HW)
    except Exception as e:                  # a machine without this rack's profile
        pytest.skip(f"profile {HW}: {e}")
    if chunk is not None:
        real = D.plan_record

        def with_chunk(*a, **k):
            p = real(*a, **k)
            p["kv_cache"]["prefill_chunk"] = chunk
            return p
        monkeypatch.setattr(D, "plan_record", with_chunk)
    D.SESSION_CHUNKS.clear()
    req = CC.census_requests(MODEL, CC._family(MODEL), [], None)[0]
    derived, _plan, _P, unhandled = D.derive_keys(MODEL, HW, "triton", CC.rungs_for(HW)[-1], req)
    assert not [w for w in unhandled if not w.startswith(("NOTE ", D.KEYLESS))], unhandled
    return {(q, k) for _u, q, k in derived}, D.SESSION_CHUNKS.get("triton")


def test_the_candidates_continuation_is_derived_and_recorded(monkeypatch):
    keys, rec = _derive(monkeypatch)
    assert AT64 in keys and AT128 in keys, "the candidates' continuation over 192 whole-page keys is not derived"
    assert TURN2 not in keys, "a 28-row one-chunk turn is keyed: the crossed grid is derived after all"
    assert rec is not None and rec["page_tokens"] == 64 and rec["reach"] == 2048, rec and rec["page_tokens"]
    assert [c["rows"] for c in rec["candidates"]] == [64, 128]
    for c in rec["candidates"]:
        tks = sorted({a[0] for a in c["attention"]})
        assert tks == list(range(c["rows"], 2049, 64)), (c["rows"], tks[:3], tks[-3:])
        assert c["graph"] and all(len(r) == 4 and r[3] >= 1 for r in c["graph"])
        assert not any("baddbmm" in r[0] and r[1].startswith(f"({c['rows']}, {c['rows']},") for r in c["graph"]), \
            "the chunk's graph carries the one-pass prefill's attention"


def test_a_planned_chunk_keys_its_rows_only(monkeypatch):
    keys, rec = _derive(monkeypatch, chunk=128)
    assert AT128 in keys and AT64 not in keys and TURN2 not in keys
    assert rec["prefill_chunk"] == 128
