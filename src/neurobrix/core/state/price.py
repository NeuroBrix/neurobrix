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
                  remaining_bytes: int) -> Dict[str, int]:
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
    floor = min(kv_min_tokens(max_tokens, prompt_margin, serve), upper)
    if length < floor:
        raise RuntimeError(
            f"ZERO FALLBACK: KV cache budget too small: {remaining_bytes / 1e6:.0f} MB remain, "
            f"{floor * token_bytes / 1e6:.0f} MB are needed for {floor} positions.")
    initial = min(int(max_tokens) + int(prompt_margin), length) if (serve and max_tokens) else 0
    return {"max_cache_len": length, "initial_cache_len": initial, "memory_bytes": length * int(token_bytes)}


def page_tokens(token_bytes: int, position_tile: int, allocation_granule: int, rung_bytes: int) -> int:
    """Positions per state page (step 2; owner 2026-10-15 Q2: derived, never a hand value). Every input is named and
    sourced, no constant inside: `position_tile` is the paged attention read's tile on the position axis (a kernel fact,
    so a page never splits a tile), `allocation_granule` the device's allocation unit (a vendor fact), `token_bytes` this
    container's declared KV per position (kv_token_bytes). The page is the smallest power-of-two multiple of the tile whose
    bytes reach the granule (no page smaller than what the device allocates), refused by name when one page exceeds the
    rung."""
    if min(token_bytes, position_tile, allocation_granule, rung_bytes) <= 0:
        raise ValueError(f"page_tokens: token_bytes {token_bytes}, position_tile {position_tile}, "
                         f"allocation_granule {allocation_granule}, rung_bytes {rung_bytes}")
    tokens = int(position_tile)
    while tokens * int(token_bytes) < int(allocation_granule):
        tokens *= 2
    if tokens * int(token_bytes) > int(rung_bytes):
        raise StateUndeclared(
            f"ZERO FALLBACK: one state page ({tokens} positions x {token_bytes} B) exceeds the rung's "
            f"{rung_bytes / 1e6:.0f} MB")
    return tokens
