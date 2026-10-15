"""An attention whose device writes its full scores is priced at what it writes and run in query chunks.

PyTorch's MPS scaled_dot_product_attention is the fused math path (pytorch/pytorch#179294): it writes the whole
[B, H, Tq, Tk] scores. Allegro native at 720x1280 (79 200 tokens) added ~13 GB in the transformer's first second on an
M4 Pro (the Mac, 2026-10-15) while Prism priced the memory-efficient workspace (no scores). Each vendor file now states
`memory.compiled_sdpa_full_scores`; where it is true, the compiled engine runs such an attention in query chunks within
the vendor's budget, and Prism prices one chunk's scores.

What this test does if the code is wrong: the exactness cells fail when the chunked attention differs from the whole
one (plain, masked, causal); the vendor cell fails when a vendor file does not state the capability (refused, never a
default) or when Apple does not state it true and NVIDIA false; the pricing cell fails when a device that writes its
scores is priced at nothing, or one that writes none is priced at anything; the Prism cell fails when the estimate of a
graph with one attention does not rise by its chunk's scores on such a device; the entry cells fail when an attention on
such a device runs whole beyond the budget, or runs at all without the plan's budget or without hardware; the plan cell
fails when a component placed on such a device does not carry the budget its attention was priced at.

Run: python -m pytest tests/unit/prism/test_an_attention_that_writes_its_scores_is_priced_and_chunked.py
"""
import glob
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from neurobrix.core.runtime.graph.device_attention import attention_in_chunks

VENDORS = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors"


def _qkv(tq=37, tk=41):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(2, 3, tq, 16, generator=g), torch.randn(2, 3, tk, 16, generator=g),
            torch.randn(2, 3, tk, 16, generator=g))


@pytest.mark.parametrize("rows", [1, 5, 16, 37])
def test_the_chunked_attention_is_the_whole_one(rows):
    q, k, v = _qkv()
    torch.testing.assert_close(attention_in_chunks(q, k, v, None, 0.0, False, None, rows),
                               F.scaled_dot_product_attention(q, k, v))


@pytest.mark.parametrize("heads,rows", [(1, 5), (2, 16), (4, 37), (5, 3)])
def test_groups_of_heads_by_bands_of_rows_are_the_whole_attention(heads, rows):
    q, k, v = _qkv()
    mask = torch.randn(2, 3, 37, 41)
    torch.testing.assert_close(attention_in_chunks(q, k, v, mask, 0.0, False, None, rows, heads=heads),
                               F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


def test_a_masked_and_a_causal_attention_chunk_exactly():
    q, k, v = _qkv(23, 23)
    mask = torch.randn(2, 1, 23, 23)
    torch.testing.assert_close(attention_in_chunks(q, k, v, mask, 0.0, False, None, 7),
                               F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    torch.testing.assert_close(attention_in_chunks(q, k, v, None, 0.0, True, None, 6),
                               F.scaled_dot_product_attention(q, k, v, is_causal=True))


def test_every_vendor_file_states_whether_its_compiled_attention_writes_its_scores():
    for f in sorted(glob.glob(str(VENDORS / "*" / "*.yml"))):
        m = (yaml.safe_load(open(f)) or {}).get("memory") or {}
        assert "compiled_sdpa_full_scores" in m, f
        assert m["compiled_sdpa_full_scores"] is ("/apple/" in f), f
        if m["compiled_sdpa_full_scores"]:
            assert 0 < m["compiled_sdpa_scores_device_fraction"] < 1 and m["sdpa_math_min_chunk_rows"] > 0, f
            assert m["compiled_sdpa_chunks_in_flight"] >= 1, f


def test_the_price_is_what_a_device_writes():
    from neurobrix.core.prism.attention_scratch import scores_budget, scratch_bytes
    apple = {"compiled_sdpa_full_scores": True, "compiled_sdpa_scores_device_fraction": 0.1,
             "sdpa_math_min_chunk_rows": 128, "compiled_sdpa_chunks_in_flight": 2}
    budget = scores_budget(apple, 18186)
    assert budget == int(18186 * 1024 * 1024 * 0.1)
    from neurobrix.core.prism.attention_scratch import chunk_plan
    full = 2 * 24 * 79200 * 79200 * 4
    # Allegro at 720x1280: 79 200 tokens, 2 x 24 heads — one 128-row band of all heads is over budget, so the cut is
    # two chunks in flight share the 1.91 GB: one head by bands of 2 944 rows (0.95 GB / 316 800 B a row, in 128-row
    # blocks), 48 x 27 chunks; the price is the two chunks the device holds at once
    assert chunk_plan(apple, budget, 2, 24, 79200, 79200) == (1, 2944)
    got = scratch_bytes(apple, budget, 2, 24, 79200, 79200)
    assert got == 2 * 2944 * 79200 * 4 and 0 < got <= budget < full, (got, budget, full)
    with pytest.raises(KeyError, match="compiled_sdpa_chunks_in_flight"):
        chunk_plan({k: v for k, v in apple.items() if k != "compiled_sdpa_chunks_in_flight"}, budget, 2, 24, 79200, 79200)
    assert scratch_bytes(apple, budget, 1, 2, 64, 64) == 1 * 2 * 64 * 64 * 4, "small scores: the whole, unchunked"
    assert scores_budget({"compiled_sdpa_full_scores": False}, None) is None
    with pytest.raises(KeyError, match="states no memory.compiled_sdpa_full_scores"):
        scores_budget({}, 18186)


def test_prism_adds_the_scores_to_the_attentions_in_op_figure():
    from neurobrix.core.prism.profiler import ActivationProfiler
    S = {"type": "symbol", "id": "s0"}
    graph = {"tensors": {"q": {"shape": [1, 2, 64, 16], "dtype": "float32", "is_input": True},
                         "k": {"shape": [1, 2, 64, 16], "dtype": "float32", "is_input": True},
                         "v": {"shape": [1, 2, 64, 16], "dtype": "float32", "is_input": True},
                         "o": {"shape": [1, 2, 64, 16], "dtype": "float32", "producer_op_uid": "aten.sdpa::0"}},
             "ops": {"aten.sdpa::0": {"op_type": "aten::scaled_dot_product_attention",
                                      "input_tensor_ids": ["q", "k", "v"], "output_tensor_ids": ["o"],
                                      "attributes": {"args": []}}},
             "execution_order": ["aten.sdpa::0"], "input_tensor_ids": ["q", "k", "v"], "output_tensor_ids": ["o"]}
    base = ActivationProfiler(graph).estimate_peak_memory(dtype_bytes=4).peak_bytes
    with_scores = ActivationProfiler(graph).estimate_peak_memory(
        dtype_bytes=4, attention_scratch=lambda b, h, tq, tk: b * h * tq * tk * 4).peak_bytes
    assert with_scores - base == 1 * 2 * 64 * 64 * 4


def test_the_entry_cuts_at_the_plans_budget_and_refuses_without_it(monkeypatch):
    """On the meta device (shapes only): the entry routes; the exactness cells above prove the cut."""
    from neurobrix.core.runtime.graph import device_attention as DA
    seen = []
    monkeypatch.setattr(DA, "attention_in_chunks", lambda *a, **k: seen.append((a[7], k.get("heads"))) or a[0])
    apple, nvidia = ("apple", "apple_m4_pro"), ("nvidia", "volta")
    q = torch.empty(2, 24, 4096, 64, device="meta")
    whole = 2 * 24 * 4096 * 4096 * 4
    DA.scaled_dot_product_attention(q, q, q, hardware=apple, budget=whole // 4)
    assert seen and seen[-1][0] % 128 == 0, "over the budget on a device that writes its scores: cut"
    n = len(seen)
    DA.scaled_dot_product_attention(q, q, q, hardware=apple, budget=whole)
    DA.scaled_dot_product_attention(q, q, q, hardware=nvidia, budget=None)
    assert len(seen) == n, "within the budget, or on a device writing no scores: whole"
    DA.scaled_dot_product_attention(q[:, 0], q[:, 0], q[:, 0], hardware=apple, budget=whole // 96)
    assert len(seen) == n + 1, "a 3-D attention is cut as one head"
    with pytest.raises(ValueError, match="states no attention_scores_budget_bytes"):
        DA.scaled_dot_product_attention(q, q, q, hardware=apple, budget=None)
    with pytest.raises(ValueError, match="was given no hardware"):
        DA.scaled_dot_product_attention(q, q, q, hardware=None, budget=whole)


def test_a_broadcast_key_head_is_kept_and_a_foreign_head_count_refused():
    q, k, v = _qkv()
    k1, v1 = k[:, :1], v[:, :1]
    torch.testing.assert_close(attention_in_chunks(q, k1, v1, None, 0.0, False, None, 5, heads=2),
                               F.scaled_dot_product_attention(q, k1.expand_as(k), v1.expand_as(v)))
    with pytest.raises(ValueError, match="neither the query's heads nor a broadcast one"):
        attention_in_chunks(q, torch.cat([k, k[:, :1]], 1), torch.cat([v, v[:, :1]], 1), None, 0.0, False, None, 5)


def test_the_plan_states_the_budget_its_attention_was_priced_at():
    from neurobrix.core.prism import PrismSolver
    from neurobrix.core.prism.solver import ComponentAllocation, ExecutionPlan
    from neurobrix.core.prism.attention_scratch import scores_budget, vendor_memory
    from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, profile
    s = PrismSolver()
    s._mode = "compiled"
    for spec, expect in ((APPLE_M4_PRO, True), (V100_16GB, False)):
        prof = profile(spec)
        dev = prof.devices[0]
        plan = ExecutionPlan.__new__(ExecutionPlan)
        kw = dict(memory_mb=1.0, architecture=dev.architecture, vendor=spec["vendor"])
        plan.components = {"t": ComponentAllocation(name="t", devices=["mps:0" if expect else "cuda:0"], dtype="float16", **kw),
                           "h": ComponentAllocation(name="h", devices=["cpu"], dtype="float32", **kw)}
        s._assign_attention_budgets(plan, prof)
        want = scores_budget(vendor_memory(dev), dev.memory_mb) if expect else None
        assert plan.components["t"].attention_scores_budget_bytes == want, spec["id"]
        assert plan.components["h"].attention_scores_budget_bytes is None, "a host component computes no device attention"
        if expect:
            assert s._attention_scratch(prof)(2, 24, 79200, 79200) <= want
    s._mode = "triton"
    assert s._attention_budget(profile(APPLE_M4_PRO)) is None, "the Triton engine runs its own attention kernels"


def test_the_compiled_resolvers_attention_op_reaches_the_cut(monkeypatch):
    """The compiled engine's own op (`CompiledOpResolver`, the graph's `_scaled_dot_product_efficient_attention`, as
    Allegro's transformer traces it) cuts at the budget it was given — the path a run takes, not the entry alone."""
    from neurobrix.core.runtime.graph import device_attention as DA
    from neurobrix.core.runtime.graph.compiled_ops import CompiledOpResolver
    seen = []
    monkeypatch.setattr(DA, "attention_in_chunks", lambda *a, **k: seen.append((a[7], k.get("heads"))) or a[0])
    q = torch.empty(2, 24, 4096, 64, device="meta", dtype=torch.float16)
    r = CompiledOpResolver(torch.device("meta"), torch.float16, hardware=("apple", "apple_m4_pro"),
                           attention_budget=2 * 24 * 4096 * 4096 * 4 // 8)
    fn = r.get_op_func("_scaled_dot_product_efficient_attention", {"args": []})
    fn(q, q, q, None, False)
    assert seen, "the resolver's attention ran whole on a device that writes its scores, over the plan's budget"


def test_the_engine_waits_for_the_device_every_in_flight_chunks(monkeypatch):
    """An asynchronous device reuses a chunk's freed scores only once it ran it (MPS, the Mac 2026-10-15 08:01: 672
    chunks enqueued -> 21.3 GB held): the cut waits every `in_flight` chunks, and its result is the whole attention."""
    waits = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: waits.append(device))
    q, k, v = _qkv()
    out = attention_in_chunks(q, k, v, None, 0.0, False, None, 5, heads=2, in_flight=3)
    chunks = (1 + 2 + 1) * 8            # 6 slices by 2: the middle group spans both batch entries (2 calls); 8 bands
    assert len(waits) == chunks // 3, (len(waits), chunks)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))
    waits.clear()
    attention_in_chunks(q, k, v, None, 0.0, False, None, 5, heads=2)
    assert not waits, "no stated in-flight bound, no wait (a device that writes no scores never cuts)"
