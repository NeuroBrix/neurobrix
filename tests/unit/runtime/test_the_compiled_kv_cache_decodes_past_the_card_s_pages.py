"""The compiled engine's paged cache past the card's pages (R30 mirror of nbxtensor's mid-decode spill): the oldest
written pages move to the host spill, the card holds the newest, and a decode step's attention — the card's partial in
torch merged with the host's (nbxtensor ops/host_decode_partial) — equals attention over every position the sequence
holds (fp64 oracle). What this test would do if the code were wrong: spilled positions dropped or doubled, the card's
positions not shifted (an append over a live page), the host partial left out, a truncation inside the spilled part
keeping stale positions, a prompt past the card's reach accepted (red)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nbxtensor.memory.host_spill import HostSpill  # noqa: E402
from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper, KVCacheConfig  # noqa: E402

LAYERS, HK, H, D, PT = 2, 2, 8, 64, 64


def _devices():
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def _wrapper(card_pages, device):
    cfg = KVCacheConfig(num_layers=LAYERS, num_kv_heads=HK, k_head_dim=D, v_head_dim=D, max_cache_len=100000,
                        dtype="float16", initial_cache_len=PT, page_tokens=PT)
    w = KVCacheAttentionWrapper(cfg)
    w.cache.card_pages = card_pages
    w.cache.host_spill = HostSpill(LAYERS, HK, D, D, 1 << 30)
    return w


def _oracle(q, ks, vs):
    q, k, v = (x.cpu().double()[0] for x in (q, torch.cat(ks, 2), torch.cat(vs, 2)))   # [H, 1, D], [HK, T, D]
    k, v = k.repeat_interleave(H // HK, 0), v.repeat_interleave(H // HK, 0)
    s = (q @ k.transpose(1, 2)) * D ** -0.5
    return (torch.softmax(s, -1) @ v)                                                     # [H, 1, D]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("prompt, steps, card_pages, truncate_at", [(100, 160, 2, None), (64, 200, 1, None),
                                                                      (70, 160, 2, 40)])
def test_a_spilled_decode_equals_attention_over_every_position(device, prompt, steps, card_pages, truncate_at):
    g = torch.Generator().manual_seed(prompt + steps)
    rnd = lambda *s: torch.randn(*s, generator=g).to(torch.float16).to(device)           # noqa: E731
    w = _wrapper(card_pages, device)
    held = [([], []) for _ in range(LAYERS)]
    for layer in range(LAYERS):
        k, v = rnd(1, HK, prompt, D), rnd(1, HK, prompt, D)
        w.cache.update(layer, k, v)
        held[layer][0].append(k), held[layer][1].append(v)
    for step in range(steps):
        if truncate_at is not None and step == steps // 2:
            w.cache.truncate(truncate_at)
            for layer in range(LAYERS):
                for lst in held[layer]:
                    whole = torch.cat(lst, 2)[:, :, :truncate_at]
                    lst[:] = [whole]
        for layer in range(LAYERS):
            q, k, v = rnd(1, H, 1, D), rnd(1, HK, 1, D), rnd(1, HK, 1, D)
            k_card, v_card = w.cache.update(layer, k, v)
            held[layer][0].append(k), held[layer][1].append(v)
            if not w.cache.spilled:
                continue
            got = w._decode_over_spill(q, k_card, v_card, None, D ** -0.5, layer)
            want = _oracle(q, held[layer][0], held[layer][1])
            assert torch.allclose(got[0].cpu().double(), want, atol=3e-3), (device, step, layer)
    assert w.cache.spilled > 0 and w.cache.get_seq_len() == (truncate_at or 0) + (steps - steps // 2 if truncate_at
                                                                                   else prompt + steps)


def test_past_the_reach_is_refused_by_name():
    w = _wrapper(1, "cpu")
    with pytest.raises(RuntimeError, match="past the card's reach"):
        w.cache.update(0, torch.zeros(1, HK, 130, D, dtype=torch.float16), torch.zeros(1, HK, 130, D, dtype=torch.float16))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("prompt, chunk, card_pages", [(400, 64, 2), (300, 100, 3)])
def test_a_prompt_past_the_card_prefills_chunk_by_chunk_over_both_parts(device, prompt, chunk, card_pages):
    """A prompt longer than the card's pages in the plan's chunks: each chunk's attention over the card's pages and the
    spilled ones equals causal attention over every position so far (fp64 oracle)."""
    g = torch.Generator().manual_seed(prompt)
    rnd = lambda *s: torch.randn(*s, generator=g).to(torch.float16).to(device)           # noqa: E731
    w = _wrapper(card_pages, device)
    q, k, v = rnd(1, H, prompt, D), rnd(1, HK, prompt, D), rnd(1, HK, prompt, D)
    for a in range(0, prompt, chunk):
        b = min(a + chunk, prompt)
        for layer in range(LAYERS):
            w.cache.update(layer, k[:, :, a:b], v[:, :, a:b])
            if not w.cache.spilled:
                continue
            got = w._prefill_over_spill(q[:, :, a:b], layer, a, b - a, D ** -0.5)[0].cpu().double()
            qq, kk, vv = (x.cpu().double()[0] for x in (q[:, :, a:b], k[:, :, :b], v[:, :, :b]))
            kk, vv = kk.repeat_interleave(H // HK, 0), vv.repeat_interleave(H // HK, 0)
            s = (qq @ kk.transpose(1, 2)) * D ** -0.5
            causal = torch.arange(b)[None, :] <= (a + torch.arange(b - a))[:, None]
            want = torch.softmax(s.masked_fill(~causal, float("-inf")), -1) @ vv
            assert torch.allclose(got, want, atol=3e-3), (device, a, layer, (got - want).abs().max())
    assert w.cache.spilled > 0
