"""The decode cache of a decoder whose weight files are dealt over two cards is sized against the card that holds it.

Under `weight_sharding` the decoder's weightless ops — the cache's appends and reads — run on its first card
(`RuntimeFactory._extract_device`), so the whole cache lives there. The check summed the room of every card the
decoder's files sit on: Qwen3-Coder-30B-A3B on two V100-32GB (Triton engine, 2026-11-11) was planned with a
3 084 MB cache beside 29 016 MB of weights on the first card, and the run ran out of memory at layer 43 of its first
prefill (31 854 MB live on that card), 2 800 MB of cache above its weights.

The files were accepted against the USABLE part of the card; the cache beside them is granted from that same part
(granted from the whole rung it took the margin: the PyTorch engine ran out of memory at 31.7 GB on that plan).

What this test would do if the code were wrong: the cache planned with the first card's files exceeds what one
component may hold there — RED (seen twice: 3 084 MB + 28 999 MB summed over both cards; then 2 366 MB + 28 999 MB
against the whole rung)."""
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.loader import load_profile
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import container_root, pin_dedicated_card, pin_host, pin_no_calibration

MODEL = "Qwen3-Coder-30B-A3B-Instruct"
PROFILE = "v100-32g-x2-nvlink"           # the committed profile: two V100-SXM2-32GB


def test_the_cache_fits_the_card_that_holds_it(monkeypatch):
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 32501, 306, "a dedicated V100-32GB as its runtime reads it")
    pin_no_calibration(monkeypatch)
    c = NBXContainer.load(str(container_root(MODEL)))
    p = PrismSolver().solve_smart(c, load_profile(PROFILE), InputConfig(batch_size=1, seq_len=17, max_tokens=40,
                                                                   decode_tokens=40, dtype="float16"), mode="triton")
    model = p.components["model"]
    assert model.strategy == "weight_sharding", f"this request no longer plans the dealt files: {p.strategy}"
    first = sorted(set(model.shard_map.values()))[0]
    files = c.get_shard_sizes()["model"]
    held_mb = sum(b for f, b in files.items() if model.shard_map[f] == first) / 2**20
    usable_mb = 29377            # the part of the pinned card one component may hold (`_usable_mb`): its rung less the margin
    assert held_mb + p.kv_cache_plan.memory_mb <= usable_mb, (
        f"{first} holds {held_mb:,.0f} MB of the decoder's files and is given a {p.kv_cache_plan.memory_mb:,.0f} MB "
        f"cache: over the {usable_mb:,.0f} MB one component may hold there")
