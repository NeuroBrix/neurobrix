"""The state engine's price — the one place the bytes of a model's kept state are computed (docs/design/state-engine.md
§4.2, step 1, validated 2026-10-15).

Pure arithmetic over the container's declaration (`topology.json` `state`, written by the build: the KV its decoder keeps,
its layers, key-value heads, k and v head dims, window) and the request. Prism asks it; the engines size what they
allocate from the plan it produced. Torch-free: Prism and both engines import it.

A flow that decodes with a component the container declares no state for is refused by name: the state is the
container's fact, never re-derived from a model's config at run time (the config said 1 500 positions for Whisper's
decoder; its graph reads a 448-row position table).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from neurobrix.core.dtype import itemsize as _itemsize


class StateUndeclared(RuntimeError):
    pass


def kv_declaration(topology: Mapping[str, Any], component_names) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(decoder, its `kv` entry) of a container whose flow decodes, or None when the flow decodes nothing. Refused by
    name when the flow decodes with a component the container declares no KV for."""
    from neurobrix.core.runtime.lm_facts import decode_lm_component
    decoder = decode_lm_component(topology, list(component_names))
    state = topology.get("state") or {}
    if decoder is None:
        return None
    entries = [e for e in (state.get(decoder) or []) if e.get("kind") == "kv"]
    if not entries:
        raise StateUndeclared(
            f"ZERO FALLBACK: the flow decodes with '{decoder}' and the container declares no KV state for it "
            f"(topology.json `state`, written by the build) — its cache cannot be priced; the container's single "
            f"write declares it (forge tools/repropagate_in_place.py --apply)")
    entry = entries[0]
    missing = [k for k in ("layers", "kv_heads", "k_dim", "v_dim") if not entry.get(k)]
    if missing:
        raise StateUndeclared(f"ZERO FALLBACK: '{decoder}' declares a KV state without {', '.join(missing)}")
    return decoder, entry


def kv_token_bytes(entry: Mapping[str, Any], dtype: str, sequences: int) -> int:
    """Bytes one cached position costs across the `sequences` the cache holds at once:
    layers x key-value heads x (k + v head dims) x the compute dtype's width x sequences."""
    return (int(entry["layers"]) * int(entry["kv_heads"]) * (int(entry["k_dim"]) + int(entry["v_dim"]))
            * _itemsize.itemsize(dtype) * int(sequences))


#: the prompt positions a cache is sized for beside the generated tokens when the package declares no `prompt_margin`
#: (runtime/defaults.json) — ONE figure for the cache's price and for the language model's prefill pass
#: (`flow_bindings.FlowBindings.prompt_prefill_length`)
DEFAULT_PROMPT_MARGIN = 128


def kv_min_tokens(max_tokens: int, prompt_margin: int, serve: bool) -> int:
    """The fewest cached positions a plan is accepted with: one turn (max_tokens + margin), two when serving;
    64 when the container declares no max_tokens."""
    if not max_tokens:
        return 64
    return int(max_tokens) * (2 if serve else 1) + int(prompt_margin)


def kv_estimate_tokens(entry: Mapping[str, Any], max_tokens: int, prompt_margin: int) -> int:
    """The positions a strategy is priced for before the cache is sized: max_tokens + margin, else the window."""
    if max_tokens:
        return int(max_tokens) + int(prompt_margin)
    window = entry.get("window")
    if not window:
        raise StateUndeclared("ZERO FALLBACK: no max_tokens and the KV state declares no window — nothing bounds it")
    return int(window)


def kv_cache_size(entry: Mapping[str, Any], token_bytes: int, max_tokens: int, prompt_margin: int, serve: bool,
                  remaining_bytes: int, floor_max_tokens: Optional[int] = None) -> Dict[str, int]:
    """{max_cache_len, initial_cache_len, memory_bytes}: as many positions as the remaining bytes afford, bounded by the
    request (run: max_tokens + margin) or the declared window (serve), never beyond the window. Refused by name below
    the minimum."""
    window = int(entry.get("window") or 0)
    if serve:
        if not window:
            raise StateUndeclared("ZERO FALLBACK: serving plans the decoder's whole window, and its KV state declares none")
        upper = window
    else:
        upper = (int(max_tokens) + int(prompt_margin)) if max_tokens else window
    if window:
        upper = min(upper, window)
    affordable = int(remaining_bytes) // int(token_bytes) if token_bytes > 0 else 0
    length = min(affordable, upper)
    # the floor is the request's when it names fewer tokens than the package (a run's --max-tokens)
    floor = min(kv_min_tokens(floor_max_tokens if floor_max_tokens else max_tokens, prompt_margin, serve), upper)
    if length < floor:
        raise RuntimeError(
            f"ZERO FALLBACK: KV cache budget too small: {remaining_bytes / 1e6:.0f} MB remain, "
            f"{floor * token_bytes / 1e6:.0f} MB are needed for {floor} positions.")
    initial = min(int(max_tokens) + int(prompt_margin), length) if (serve and max_tokens) else 0
    return {"max_cache_len": length, "initial_cache_len": initial, "memory_bytes": length * int(token_bytes)}


def page_tokens(token_bytes_per_layer: int, tile_candidates, granule_bytes: int, rung_bytes: int) -> int:
    """Positions per state page (step 2; owner 2026-10-15 Q2: derived, never a hand value; agreed with the Dell 10:16).
    Every input is named and sourced, no constant inside:
    - `tile_candidates`: the position tiles the paged attention read declares (a kernel fact, one source for every
      backend). The page starts at the largest, so every candidate divides it and a tile never straddles two pages; the
      tile is then chosen among candidates that divide the page, never autotuned against it.
    - `granule_bytes`: the backend's allocation granule (a cited vendor fact per backend).
    - `token_bytes_per_layer`: kv_heads x (k_dim + v_dim) x itemsize of the declared KV, one layer's position.
    The page is that largest tile doubled until one layer's page reaches the granule; refused by name when one page of
    every layer exceeds the rung."""
    tiles = [int(c) for c in tile_candidates]
    if not tiles or min(tiles) <= 0 or token_bytes_per_layer <= 0 or granule_bytes <= 0 or rung_bytes <= 0:
        raise ValueError(f"page_tokens: token_bytes_per_layer {token_bytes_per_layer}, tile_candidates {tiles}, "
                         f"granule_bytes {granule_bytes}, rung_bytes {rung_bytes}")
    tokens = max(tiles)
    if any(tokens % c for c in tiles):
        raise ValueError(f"page_tokens: the tile candidates {tiles} do not all divide the largest")
    while tokens * int(token_bytes_per_layer) < int(granule_bytes):
        tokens *= 2
    if tokens * int(token_bytes_per_layer) > int(rung_bytes):
        raise StateUndeclared(
            f"ZERO FALLBACK: one state page ({tokens} positions x {token_bytes_per_layer} B) exceeds the rung's "
            f"{rung_bytes / 1e6:.0f} MB")
    return tokens


def paged_cache_params(num_kv_heads: int, k_head_dim: int, v_head_dim: int, dtype: str, vendor_memory,
                       cache_bytes: int, tile_candidates) -> Tuple[int, bool]:
    """(positions per page, whether a released page waits for a fence) of a paged KV cache — the one derivation both
    engines take (step 2): one layer's position bytes from the declared geometry and the compute dtype, the paged read's
    tile candidates, the card's page granule (vendor file memory.state_page_granule_bytes) and the cache's bytes as the
    bound; the release rule is the vendor file's memory.async_release. A key the vendor file does not state is refused
    by name."""
    memory = vendor_memory or {}
    missing = [k for k in ("state_page_granule_bytes", "async_release") if k not in memory]
    if missing:
        raise StateUndeclared(f"ZERO FALLBACK: the executing card's vendor file states no memory."
                              f"{', memory.'.join(missing)} — the paged KV cache's page size and release rule are vendor "
                              f"facts, never assumed")
    token_bytes = int(num_kv_heads) * (int(k_head_dim) + int(v_head_dim)) * _itemsize.itemsize(dtype)
    tokens = page_tokens(token_bytes, tile_candidates, int(memory["state_page_granule_bytes"]), int(cache_bytes))
    return tokens, bool(memory["async_release"])
