"""A serving session's prefill chunk attends its own rows AND every cached key: the ONE reserve it has for that
(`price.session_chunk_attention_bytes`: rows x reach at the item size of its dense mask — the cache's dtype on the
compiled engine, float32 on the Triton engine) is priced into the decoder's activations
(`served_prefill.chunk_mask_bytes`, the keys and their dtype stated by core/prism/served_prefill.py), and it is the
figure both engines' wrappers spend: a plain chunk on its mask, a chunk over spilled positions on its scores.

Before (2026-11-10): the walk priced the pass over its own rows only. Once the head's wrongly priced room was freed,
Qwen3-Coder-30B-A3B's served chunk rose to 4 096 rows on two V100-32GB and a 79 931-token turn ran its card out of
memory twice: the 15th chunk's 480 MiB mask (4 096 x 61 440 keys, fp16), then a spilled chunk's 3.2 GB of scores.

What this file does if the code is wrong:
1. the reserve is not rows x keys x the mask's item size per engine, or a non-positive extent is taken — red;
2. TinyLlama's served plan with the keys stated prices its decoder exactly that reserve more than without them — fp16
   on the compiled engine, float32 on the Triton engine — red on a tree that prices nothing (difference 0);
3. the plan `solve_served` returns was not solved with the cached keys and their dtype stated — red;
4. a run's plan (no cached keys) moves — red.

No card. Run: python -m pytest tests/unit/prism/test_a_served_chunk_is_priced_with_the_cached_keys_it_attends.py
"""
import dataclasses

import pytest

from neurobrix.core.prism.served_prefill import solve_served as SP_solve_served
from tests.unit.prism._pinned_machine import profile, V100_16GB
from tests.unit.prism.test_the_generation_head_is_priced_at_its_one_row import _served


def test_the_reserve_is_rows_by_keys_at_the_engines_mask_item_size():
    from neurobrix.core.state.price import session_chunk_attention_bytes as reserve
    assert reserve(4096, 61440, "float16", "compiled") == 503316480 == 480 * 2 ** 20      # the 480 MiB of the failure
    assert reserve(4096, 61440, "float16", "triton") == 2 * 503316480                      # its mask is float32
    assert reserve(128, 2048, "float32", "compiled") == 128 * 2048 * 4
    for bad in ((0, 8), (8, 0)):
        with pytest.raises(ValueError):
            reserve(*bad, "float16", "compiled")


@pytest.mark.parametrize("mode,itemsize", [("compiled", 2), ("triton", 4)])
def test_the_decoder_is_priced_the_mask_more_when_the_keys_are_stated(monkeypatch, mode, itemsize):
    solver, c, cfg = _served(monkeypatch, mode)
    at = dataclasses.replace(cfg, seq_len=512)
    without = solver.solve_smart(c, profile(V100_16GB), at, serve_mode=True, mode=mode)
    with_keys = solver.solve_smart(c, profile(V100_16GB),
                                   dataclasses.replace(at, prefill_keys=2048, prefill_key_dtype="float16"), serve_mode=True,
                                   mode=mode)
    more = with_keys.component_memory["model"].activation_bytes - without.component_memory["model"].activation_bytes
    assert more == 512 * 2048 * itemsize, (more, 512 * 2048 * itemsize)
    assert with_keys.component_memory["lm_head"].activation_bytes == without.component_memory["lm_head"].activation_bytes


def test_the_served_plan_is_solved_with_the_keys_it_attends(monkeypatch):
    solver, c, cfg = _served(monkeypatch, "compiled")
    plan, used = SP_solve_served(c, profile(V100_16GB), cfg, mode="compiled")
    kv = plan.kv_cache_plan
    assert used.prefill_keys == kv.max_cache_len == 2048 and used.seq_len == kv.prefill_chunk
    assert used.prefill_key_dtype == str(kv.dtype)


def test_a_run_states_no_cached_keys_and_its_plan_is_untouched(monkeypatch):
    solver, c, cfg = _served(monkeypatch, "compiled")
    at = dataclasses.replace(cfg, seq_len=512)
    assert at.prefill_keys is None
    a = solver.solve_smart(c, profile(V100_16GB), at, serve_mode=False, mode="compiled")
    solver2, c2, _ = _served(monkeypatch, "compiled")
    seen = []
    monkeypatch.setattr("neurobrix.core.state.price.session_chunk_attention_bytes", lambda *a_: seen.append(a_) or 0)
    b = solver2.solve_smart(c2, profile(V100_16GB), at, serve_mode=False, mode="compiled")
    assert not seen and a.component_memory["model"].activation_bytes == b.component_memory["model"].activation_bytes
