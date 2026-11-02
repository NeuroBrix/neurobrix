"""Sampling-parameter contract — refuse loudly rather than ignore silently.

Zero imports on purpose: this is a contract check, not compute, so both
engines can call it without either subtree pulling on the other.

=== WHAT IT HOLDS ===

`resolve_sampling` — THE resolution of a decode's four sampling parameters (request > container > family file), read
by every text, vlm and audio_llm decode of both branches, which then draw with their branch's one `CombinedSampler`
(`CombinedSampler(sampler_config(...))` in each branch, no compute shared; the supervisor 2026-11-02 06:40).
Before it, four flow paths sampled on their own and each implemented a different subset (the vlm and audio_llm decodes
had no top-k; the compiled vlm dropped top-p; audio_llm read the container's temperature whatever the request said).

`enforce_sampling_support` — refuse loudly rather than ignore silently, for the decodes that still sample on their own
(encoder_decoder, tts_llm, dual_ar) and for a parameter no sampler implements (min_p):

    site                      temperature  rep_penalty  top_p  top_k  min_p
    CombinedSampler (both)         yes         yes       yes    yes     no
    encoder_decoder                yes         yes        no     no     no
    tts_llm                        yes          -        yes     no    yes

A user who sets a parameter a path drops gets an error when they asked for it, a one-time warning when it was
inherited from the registry; consolidating those last paths is the rest of P-SAMPLING-CONSOLIDATION.

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


# The parameters every LLM-headed decode samples with, through its branch's ONE sampler (core/module/autoregressive/
# samplers.CombinedSampler, triton/samplers.CombinedSampler): text, vlm and audio_llm decodes of both branches.
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")
_CASTS = {"temperature": float, "top_k": int, "top_p": float, "repetition_penalty": float}
#: What the sampler implements — the contract enforce_sampling_support checks every other configured key against.
SAMPLER_IMPLEMENTS = SAMPLING_KEYS


def resolve_sampling(resolver, container_defaults, site: str) -> dict:
    """The sampling one decode runs with, resolved ONE way for every text, vlm and audio_llm decode of both branches.

    Order, the first that carries a value wins:
      1. the request — `global.<key>`, set by the CLI flag (`--temperature`, `--top-k`, `--top-p`,
         `--repetition-penalty`) or a serving request;
      2. the container — its `runtime/defaults.json` (the vendor's generation_config as the build embedded it);
      3. the family file — `config/families/<family>.yml` `defaults:` (and its generation-type block), the documented
         defaults (docs/reference/cli.md, "Sampling").
    A null is no value: a container that declares `top_k: null` leaves the family's value in force. A key that no
    source carries is refused by name — never a literal in the code.

    The executor injects the merged defaults as `global.temperature/top_p/top_k` when the request set none; the order
    above is unchanged by it (the merged value IS the container's over the family's), and a null injected there is no
    value either.
    """
    resolved = getattr(resolver, "resolved", None) or {}
    family = getattr(resolver, "family_defaults", None) or {}
    container = container_defaults or {}
    out, missing = {}, []
    for key in SAMPLING_KEYS:
        value = resolved.get(f"global.{key}")
        if value is None:
            value = container.get(key)
        if value is None:
            value = family.get(key)
        if value is None:
            missing.append(key)
            continue
        out[key] = _CASTS[key](value)
    if missing:
        raise RuntimeError(
            f"ZERO FALLBACK: sampling: {site} has no value for {', '.join(missing)} — neither the request "
            f"(--{missing[0].replace('_', '-')}), the container's runtime/defaults.json nor the family file "
            f"(config/families/<family>.yml `defaults:`) declares it.")
    return out


def sampler_config(resolver, container_defaults, site: str) -> dict:
    """What a decode's CombinedSampler is built from, either branch (`CombinedSampler(sampler_config(...))`): the four
    resolved parameters (resolve_sampling) and the top token always kept (min_tokens_to_keep 1, the HF convention: a
    top-p cut never empties the candidate set)."""
    return dict(resolve_sampling(resolver, container_defaults, site), min_tokens_to_keep=1)
