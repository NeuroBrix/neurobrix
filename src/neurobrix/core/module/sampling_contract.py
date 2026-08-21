"""Sampling-parameter contract — refuse loudly rather than ignore silently.

Zero imports on purpose: this is a contract check, not compute, so both
engines can call it without either subtree pulling on the other.

=== WHY THIS EXISTS ===

`CombinedSampler` is the sampling engine, but four flow paths sample on
their own — `core/flow/encoder_decoder.py`, `core/flow/audio_utils.py`,
`core/flow/tts_llm.py`, `triton/flow/audio_llm.py` — and each implements
a different subset of the parameters:

    site                      temperature  rep_penalty  top_p  top_k  min_p
    CombinedSampler (engine)       yes         yes       yes    yes     no
    encoder_decoder                yes         yes        no     no     no
    audio_utils                    yes         yes       yes     no     no
    tts_llm                        yes          -        yes     no    yes
    triton audio_llm               yes         yes        no     no     no

`top_k` is honoured by the engine and by nothing else. MiniCPM-o's
registry declares `top_k=20` and the voice path never applies it. A user
who sets top-k, top-p or a repetition penalty and whose setting is
dropped depending on which flow ran gets no error, no warning, and
plausible-looking output of the wrong quality.

That is the same family as the top-k merge defect: it does not crash, it
is invisible, and it degrades quality silently. The doctrine that answers
it is the one already used for unsupported capabilities — **refuse at the
boundary, naming what is missing** — applied to parameters rather than to
model families.

=== WHAT THIS IS AND IS NOT ===

It is a stopgap that makes the divergence LOUD. It is not the fix. The
fix is consolidation onto the engine's sampler, tracked as
P-SAMPLING-CONSOLIDATION; until that lands, a dropped parameter raises
instead of passing.

Neutral values never raise: a parameter at its disabled default is not a
request. `top_k=0`, `top_p=1.0`, `repetition_penalty=1.0`, `min_p=0.0`
all mean "not asked for".
"""

from __future__ import annotations

# Value at which each parameter is a no-op, i.e. not actually requested.
_NEUTRAL = {
    "temperature": None,        # no neutral value; always implemented
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "min_p": 0.0,
}

_CHANTIER = "P-SAMPLING-CONSOLIDATION"


def _requested(name: str, value) -> bool:
    """True when the caller actually asked for this parameter."""
    if value is None:
        return False
    neutral = _NEUTRAL.get(name, None)
    if neutral is None:
        return False
    try:
        return float(value) != float(neutral)
    except (TypeError, ValueError):
        return False


def enforce_sampling_support(site: str, supported, config,
                             explicit=None) -> None:
    """Refuse or report when `config` asks for something `site` ignores.

    Two cases are deliberately NOT treated the same, because conflating
    them would either brick working models or hide real requests:

    - **explicitly requested** (the user passed `--top-k`, or a serving
      request set it): RAISE. Ignoring what the user just asked for is
      the silent-quality-change class this guard exists to end.
    - **inherited from the registry**: WARN once on stderr. A vendor
      `generation_config` routinely carries fields meant for the model's
      TEXT decoder; the voice or codec path is a different decoder, and
      the field was never about it. MiniCPM-o declares `top_k=20` and
      reaches sampling through `triton/flow/audio_llm.py`, which has no
      top-k. Refusing there would replace a field that does not apply
      with a hard failure of a working model.

    Args:
        site: human-readable path name, used in the message.
        supported: iterable of parameter names this path implements.
        config: mapping of parameter name -> configured value.
        explicit: iterable of parameter names the CALLER set explicitly
            (CLI / request overrides). None means "unknown", which is
            treated as non-explicit — the conservative direction.

    Raises:
        RuntimeError when an explicitly requested parameter is ignored.
    """
    sup = set(supported)
    exp = set(explicit or ())
    ignored = [k for k in _NEUTRAL
               if k not in sup and _requested(k, config.get(k))]
    if not ignored:
        return

    hard = sorted(f"{k}={config.get(k)!r}" for k in ignored if k in exp)
    soft = sorted(f"{k}={config.get(k)!r}" for k in ignored if k not in exp)

    if hard:
        raise RuntimeError(
            f"sampling: {site} does not implement {', '.join(hard)} — you "
            f"asked for it explicitly and this path would ignore it, "
            f"silently changing output quality with no error. Sampling is "
            f"implemented in six places and only the engine's "
            f"CombinedSampler covers the full set; consolidating them is "
            f"{_CHANTIER}. Until then this refuses rather than drop your "
            f"setting. Re-run without it, or on a path that implements it."
        )

    if soft:
        key = (site, tuple(soft))
        if key not in _WARNED:
            _WARNED.add(key)
            import sys
            print(f"[NeuroBrix] sampling: {site} ignores {', '.join(soft)} "
                  f"inherited from the model registry — that path does not "
                  f"implement it. Not an error (the value was not requested "
                  f"for this path), but the divergence is real and is "
                  f"tracked as {_CHANTIER}.", file=sys.stderr, flush=True)


_WARNED = set()
