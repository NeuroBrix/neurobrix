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


def test_a_chunk_reads_the_spilled_positions_on_the_device_in_blocks_no_larger_than_its_own_keys(monkeypatch):
    """A prefill chunk's rows over the spilled positions are the device's work, block by block (the library's
    HostSpill.blocks, at most the card part's own key count each): the host partial — the decode step's one row — is
    not asked (it was half of each spilled chunk's time on the 79 931-token turn, 2026-11-10)."""
    prompt, chunk, card_pages = 640, 64, 2
    g = torch.Generator().manual_seed(11)
    rnd = lambda *s: torch.randn(*s, generator=g).to(torch.float16)                       # noqa: E731
    w = _wrapper(card_pages, "cpu")
    q, k, v = rnd(1, H, prompt, D), rnd(1, HK, prompt, D), rnd(1, HK, prompt, D)

    def refuse(*a, **kw):
        raise AssertionError("the host partial was asked for a prefill chunk")
    monkeypatch.setattr(w.cache.host_spill, "partial", refuse)
    seen, real = [], w.cache.host_spill.blocks

    def blocks(layer, positions):
        for kb, vb in real(layer, positions):
            seen.append((int(kb.shape[1]), int(positions)))
            yield kb, vb
    monkeypatch.setattr(w.cache.host_spill, "blocks", blocks)
    for a in range(0, prompt, chunk):
        b = a + chunk
        for layer in range(LAYERS):
            w.cache.update(layer, k[:, :, a:b], v[:, :, a:b])
            if not w.cache.spilled:
                continue
            got = w._prefill_over_spill(q[:, :, a:b], layer, a, chunk, D ** -0.5)[0].double()
            qq, kk, vv = (x.double()[0] for x in (q[:, :, a:b], k[:, :, :b], v[:, :, :b]))
            kk, vv = kk.repeat_interleave(H // HK, 0), vv.repeat_interleave(H // HK, 0)
            causal = torch.arange(b)[None, :] <= (a + torch.arange(chunk))[:, None]
            want = torch.softmax(((qq @ kk.transpose(1, 2)) * D ** -0.5).masked_fill(~causal, float("-inf")), -1) @ vv
            assert torch.allclose(got, want, atol=3e-3), (a, layer, (got - want).abs().max())
    assert w.cache.spilled > card_pages * PT                               # more spilled than one block holds
    assert seen and all(n <= positions <= card_pages * PT for n, positions in seen)


@pytest.mark.parametrize("budget, rows_at_once", [(None, PT), (12 * H * 2 * PT * 16, 16), (12 * H * 3 * PT, 1)])
def test_a_spilled_chunk_larger_than_its_scores_budget_is_cut_in_rows(monkeypatch, budget, rows_at_once):
    """The spilled chunk's math writes its fp32 scores: a chunk of many rows (the served plan's larger chunks, 2 048
    rows x 65 664 keys = 17 GB a matrix, 2026-11-10) is cut to the rows the plan's `spill_scores_bytes` holds — one
    page of rows when the plan states none — and equals causal attention over every position."""
    prompt, chunk, card_pages = 6 * PT, 2 * PT, 2
    g = torch.Generator().manual_seed(5)
    rnd = lambda *s: torch.randn(*s, generator=g).to(torch.float16)                       # noqa: E731
    w = _wrapper(card_pages, "cpu")
    w.spill_scores_bytes = budget
    q, k, v = rnd(1, H, prompt, D), rnd(1, HK, prompt, D), rnd(1, HK, prompt, D)
    widest, real_exp = [0], torch.exp

    def exp(x):
        if x.dim() == 3:                                                   # a scores matrix [H_kv, groups * rows, keys]
            widest[0] = max(widest[0], int(x.shape[1]) // (H // HK))
        return real_exp(x)
    monkeypatch.setattr(torch, "exp", exp)
    for a in range(0, prompt, chunk):
        b = a + chunk
        for layer in range(LAYERS):
            w.cache.update(layer, k[:, :, a:b], v[:, :, a:b])
            if not w.cache.spilled:
                continue
            got = w._prefill_over_spill(q[:, :, a:b], layer, a, chunk, D ** -0.5)[0].double()
            qq, kk, vv = (x.double()[0] for x in (q[:, :, a:b], k[:, :, :b], v[:, :, :b]))
            kk, vv = kk.repeat_interleave(H // HK, 0), vv.repeat_interleave(H // HK, 0)
            causal = torch.arange(b)[None, :] <= (a + torch.arange(chunk))[:, None]
            want = torch.softmax(((qq @ kk.transpose(1, 2)) * D ** -0.5).masked_fill(~causal, float("-inf")), -1) @ vv
            assert torch.allclose(got, want, atol=3e-3), (a, layer, (got - want).abs().max())
    assert w.cache.spilled > 0 and widest[0] == rows_at_once


def test_a_scores_budget_that_holds_no_row_is_refused_by_name():
    w = _wrapper(1, "cpu")
    w.spill_scores_bytes = 1024
    g = torch.Generator().manual_seed(1)
    rnd = lambda *s: torch.randn(*s, generator=g).to(torch.float16)                       # noqa: E731
    for a in range(0, 3 * PT, PT):
        for layer in range(LAYERS):
            w.cache.update(layer, rnd(1, HK, PT, D), rnd(1, HK, PT, D))
    with pytest.raises(RuntimeError, match="one row of a spilled prefill chunk writes"):
        w._prefill_over_spill(rnd(1, H, PT, D), 0, 2 * PT, PT, D ** -0.5)
