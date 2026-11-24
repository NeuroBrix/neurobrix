"""A served session whose plan leaves ONE page of 64 positions on the card and the rest to the host spill (Qwen2.5-Coder-
7B served on one V100-16GB, register rows 525/526): a prompt past the card's pages, and a second turn that starts inside
a page, prefill in chunks cut to what the card holds — the older pages spill to the host as the chunks write — and
every real row's attention equals causal attention over every position so far (fp64 oracle). The compiled branch
(core/flow/autoregressive.GraphLMSession._prefill_chunks); the Triton twin is
tests/unit/triton/test_a_served_prompt_past_the_card_prefills_in_the_card_s_reach.py. What this test would do if the
code were wrong: the plan's chunk of 128 written whole over a card of 64 ("a prompt past the card's reach", turn 1), a
chunk of a page from a position inside a page ("past the card's reach", turn 2), a one-pass prompt longer than the card,
spilled positions read wrong (the oracle)."""
import pytest

torch = pytest.importorskip("torch")

from nbxtensor.memory.host_spill import HostSpill  # noqa: E402
from neurobrix.core.flow.autoregressive import GraphLMSession  # noqa: E402
from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper, KVCacheConfig  # noqa: E402

LAYERS, HK, H, D, PT, HID, POSITIONS = 2, 2, 8, 64, 64, 16, 512


class _Graph:
    """A two-layer graph: each run attends its rows through the KV interceptor as the compiled sequence does (a chunk
    over the cache's whole pages, else the one-pass append) with q/k/v read from fixed per-position tables."""

    device = "cpu"

    def __init__(self, wrapper, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.w = wrapper
        self.q = [torch.randn(1, H, POSITIONS, D, generator=g).half() for _ in range(LAYERS)]
        self.k = [torch.randn(1, HK, POSITIONS, D, generator=g).half() for _ in range(LAYERS)]
        self.v = [torch.randn(1, HK, POSITIONS, D, generator=g).half() for _ in range(LAYERS)]
        self.rows_run, self.got, self.hidden = [], {}, None

    def enable_hidden_states_capture(self):
        """The session asks it before a prefill; the fake keeps its hidden states always."""

    def run(self, inputs):
        """One prefill run of the rows at `position_ids`: every layer through the interceptor's prefill path."""
        pos = [int(p) for p in inputs["position_ids"][0].tolist()]
        self.rows_run.append(len(pos))
        for layer in range(LAYERS):
            q, k, v = (t[layer][:, :, pos] for t in (self.q, self.k, self.v))
            if self.w._whole_pages:
                out = self.w._prefill_over_pages(q, k, v, None, 0.0, D ** -0.5, layer)
                for i in range(self.w._chunk_real):
                    self.got[(layer, self.w._chunk_start + i)] = out[0, :, i].double()
            else:                                                   # the one-pass prefill's append
                self.w.cache.update(layer, k, v)
        self.hidden = torch.zeros(1, len(pos), HID)

    def get_hidden_states(self, expected_hidden_dim, expected_batch_size):
        """The last run's hidden states, one row per chunk row."""
        return self.hidden

    def want(self, layer: int, p: int):
        """Causal attention of position p over positions 0..p, fp64."""
        q = self.q[layer][0, :, p].double()                                         # [H, D]
        k = self.k[layer][0, :, :p + 1].double().repeat_interleave(H // HK, 0)      # [H, p+1, D]
        v = self.v[layer][0, :, :p + 1].double().repeat_interleave(H // HK, 0)
        s = torch.einsum("hd,hnd->hn", q, k) * D ** -0.5
        return torch.einsum("hn,hnd->hd", torch.softmax(s, -1), v)


def _session(card_pages: int, seed: int):
    cfg = KVCacheConfig(num_layers=LAYERS, num_kv_heads=HK, k_head_dim=D, v_head_dim=D, max_cache_len=100000,
                        dtype="float16", initial_cache_len=PT, page_tokens=PT)
    w = KVCacheAttentionWrapper(cfg)
    w.cache.card_pages = card_pages
    w.cache.host_spill = HostSpill(LAYERS, HK, D, D, 1 << 30)
    graph = _Graph(w, seed)
    return GraphLMSession(graph, w, HID, ["input_ids", "position_ids"], uses_embeds=False,
                          uses_absolute_position=True), graph


@pytest.mark.parametrize("chunk", [128, 64, None])
def test_two_turns_past_one_card_page_prefill_within_it_and_equal_the_oracle(chunk):
    session, graph = _session(1, 7 + (chunk or 0))
    ids = list(range(1000, 1200))
    session.prefill(torch.tensor([ids[:130]]), 1, start=0, chunk=chunk)                # turn 1: past the card's page
    assert session.kv_wrapper.cache.get_seq_len() == 130 and session.kv_wrapper.cache.spilled > 0
    session.prefill(torch.tensor([ids]), 1, start=130, chunk=chunk)                     # turn 2: from inside a page
    assert session.kv_wrapper.cache.get_seq_len() == 200
    assert all(r <= PT for r in graph.rows_run), graph.rows_run                    # every chunk within the card's page
    assert session.prefilled_rows == 70
    for layer in range(LAYERS):
        for p in range(200):
            assert torch.allclose(graph.got[(layer, p)], graph.want(layer, p), atol=3e-3), (chunk, layer, p)


def test_a_card_that_holds_the_prompt_keeps_the_plan_s_chunk():
    """Nothing changes where the card holds the chunk: the plan's chunk of 128 on a card of four pages runs whole."""
    session, graph = _session(4, 3)
    session.prefill(torch.tensor([list(range(130))]), 1, start=0, chunk=128)
    assert graph.rows_run == [128, 128] and session.padded_rows == 126
