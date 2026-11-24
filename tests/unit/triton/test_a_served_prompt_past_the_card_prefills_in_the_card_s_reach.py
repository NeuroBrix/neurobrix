"""The Triton twin of tests/unit/runtime/test_a_served_prompt_past_the_card_prefills_in_the_card_s_reach.py: a served
session whose plan leaves ONE page of 64 positions on the card and the rest to the host spill (Qwen2.5-Coder-7B served
on one V100-16GB, register row 526: turn 2 refused "a prefill chunk larger than the card's pages"). A prompt past the
card's pages, and a second turn that starts inside a page, prefill in chunks cut to what the card holds
(triton/session.TritonLMSession._prefill_chunks) — the older pages spill to the host as the chunks write — and every
real row's attention equals causal attention over every position so far (fp64 oracle). Zero torch. What this test
would do if the code were wrong: a chunk of 128 over a card of 64, a chunk of a page from a position inside a page, a
one-pass prompt longer than the card (each refused by the library's cache), spilled positions read wrong (the oracle)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.host_spill import HostSpill
from nbxtensor.memory.paged_kv import PagedTritonKVCache
from nbxtensor.tensor import NBXTensor

LAYERS, HK, H, D, PT, HID, POSITIONS = 2, 2, 8, 64, 64, 16, 512
#: the Dell's figures (Qwen2.5-Coder-7B, gate 1393, turn 2): 28 query heads, 4 KV heads of 128, pages of 64 — the
#: library's spilled chunk reads its blocks at most the card part's own keys (memory/paged_kv._chunk_attention_spilled)
DELL = dict(h=28, hk=4, d=128)


@pytest.fixture(autouse=True)
def _accelerator():
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _dev(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a)).to(device="cuda:0")


class _Graph:
    """A two-layer graph: each run attends its rows through the interceptor as the Triton sequence does (a chunk over
    the cache's whole pages, else the one-pass append) with q/k/v read from fixed per-position tables."""

    def __init__(self, wrapper, seed: int, h: int = H, hk: int = HK, d: int = D):
        rng = np.random.default_rng(seed)
        self.w, self.h, self.hk, self.d = wrapper, h, hk, d
        self.q = [rng.standard_normal((1, h, POSITIONS, d)).astype(np.float16) for _ in range(LAYERS)]
        self.k = [rng.standard_normal((1, hk, POSITIONS, d)).astype(np.float16) for _ in range(LAYERS)]
        self.v = [rng.standard_normal((1, hk, POSITIONS, d)).astype(np.float16) for _ in range(LAYERS)]
        self.rows_run, self.got, self.hidden = [], {}, None

    def enable_hidden_states_capture(self):
        """The session asks it before a prefill; the fake keeps its hidden states always."""

    def run(self, inputs, skip_kills=False):
        """One prefill run of the rows at `position_ids`: every layer through the interceptor's prefill path."""
        pos = [int(p) for p in inputs["position_ids"].numpy()[0]]
        self.rows_run.append(len(pos))
        for layer in range(LAYERS):
            q, k, v = (_dev(t[layer][:, :, pos]) for t in (self.q, self.k, self.v))
            if self.w._whole_pages:
                out = self.w._prefill_over_pages(q, k, v, k, v, False, layer, 0.0, self.d ** -0.5).to(NBXDtype.float32)
                out = out.numpy()
                for i in range(self.w._chunk_real):
                    self.got[(layer, self.w._chunk_start + i)] = out[0, :, i].astype(np.float64)
            else:                                                   # the one-pass prefill's append
                self.w.cache.prefill(layer, k, v)
        self.hidden = _dev(np.zeros((1, len(pos), HID), dtype=np.float16))
        return {}

    def get_hidden_states(self, expected_hidden_dim, expected_batch_size):
        """The last run's hidden states, one row per chunk row."""
        return self.hidden

    def want(self, layer: int, p: int):
        """Causal attention of position p over positions 0..p, fp64."""
        q = self.q[layer][0, :, p].astype(np.float64)
        k = np.repeat(self.k[layer][0, :, :p + 1].astype(np.float64), self.h // self.hk, 0)
        v = np.repeat(self.v[layer][0, :, :p + 1].astype(np.float64), self.h // self.hk, 0)
        s = np.einsum("hd,hnd->hn", q, k) * self.d ** -0.5
        e = np.exp(s - s.max(-1, keepdims=True))
        return np.einsum("hn,hnd->hd", e / e.sum(-1, keepdims=True), v)


def _session(card_pages: int, seed: int, h: int = H, hk: int = HK, d: int = D):
    from neurobrix.triton.kv_cache import TritonAttentionInterceptor
    from neurobrix.triton.session import TritonLMSession
    cache = PagedTritonKVCache(LAYERS, hk, d, d, max_cache_len=100000, dtype=NBXDtype.float16, page_tokens=PT,
                               async_release=False, initial_cache_len=PT, card_pages=card_pages,
                               host_spill=HostSpill(LAYERS, hk, d, d, 1 << 30))
    w = TritonAttentionInterceptor(cache, h)
    graph = _Graph(w, seed, h, hk, d)
    return TritonLMSession(graph, w, HID, ["input_ids", "position_ids"], uses_embeds=False,
                           uses_absolute_position=True), graph


@pytest.mark.parametrize("dims", [{}, DELL], ids=["small", "dell"])
@pytest.mark.parametrize("chunk", [128, 64, None])
def test_two_turns_past_one_card_page_prefill_within_it_and_equal_the_oracle(chunk, dims):
    """Turn 2 runs over spilled blocks, at the small figures and the Dell's."""
    session, graph = _session(1, 7 + (chunk or 0), **dims)
    ids = list(range(1000, 1200))
    session.prefill(_dev(np.asarray([ids[:130]], np.int64)), 1, start=0, chunk=chunk, ids=ids[:130])   # turn 1
    cache = session.kv_wrapper.cache
    assert cache.get_cache_len() == 130 and cache.spilled > 0
    session.prefill(_dev(np.asarray([ids], np.int64)), 1, start=130, chunk=chunk, ids=ids)  # turn 2: inside a page
    assert cache.get_cache_len() == 200
    assert all(r <= PT for r in graph.rows_run), graph.rows_run                    # every chunk within the card's page
    assert session.prefilled_rows == 70
    for layer in range(LAYERS):
        for p in range(200):
            assert np.allclose(graph.got[(layer, p)], graph.want(layer, p), atol=3e-3), (chunk, layer, p)


def test_a_card_that_holds_the_prompt_keeps_the_plan_s_chunk():
    """Nothing changes where the card holds the chunk: the plan's chunk of 128 on a card of four pages runs whole."""
    session, graph = _session(4, 3)
    ids = list(range(130))
    session.prefill(_dev(np.asarray([ids], np.int64)), 1, start=0, chunk=128, ids=ids)
    assert graph.rows_run == [128, 128] and session.padded_rows == 126
