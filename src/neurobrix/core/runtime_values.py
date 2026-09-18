"""Runtime values come from the request, the container, or the family config.

A literal standing in for a runtime dimension or parameter is a silent wrong
answer waiting to happen. It looks like a default and behaves like a claim: the
engine proceeds as if someone had asked for 1024x1024, a batch of 2 or 512
tokens, and every downstream estimate inherits that invention.

Measured 2026-09-17: `cli/commands/run.py` planned the SAME 278 MB for a 448x448
request and a 160x112 one, because height and width fell through to a literal
1024. The per-cell memory gate was fed that number and could refuse nothing; the
cell went on to hold 8408 MB live and take the machine to 127 MB.

So there is exactly one way to obtain such a value, and it refuses by name when
nothing provides it.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


class MissingRuntimeValue(RuntimeError):
    """A runtime value nobody declared. Carries what was missing and where it
    was looked for, so the fix is a declaration, never a guess."""


_SENTINEL = object()


def resolve(name: str,
            request: Any = None,
            container: Optional[Mapping[str, Any]] = None,
            family: Optional[Mapping[str, Any]] = None,
            *,
            request_attr: Optional[str] = None,
            container_key: Optional[str] = None,
            family_key: Optional[str] = None,
            extra: Sequence[tuple] = (),
            default: Any = _SENTINEL,
            why: str = "") -> Any:
    """The request, then the container, then the family config, then refusal.

    `default` exists only for values that are genuinely OPTIONAL — a dimension
    that does not apply to a model (num_frames on an image model). Passing a
    default for a value the model needs is the very thing this module exists to
    stop, so a default of None means "absent is a legitimate answer", and any
    other default must be justified at the call site.

    `extra` is an ordered sequence of (source_name, value) pairs consulted
    between the request and the container — the place for a value read from the
    request's own payload, such as an input image's dimensions.
    """
    attr = request_attr or name
    if request is not None:
        v = getattr(request, attr, None)
        if v is not None:
            return v
    for src_name, v in extra:
        if v is not None:
            return v
    if container is not None:
        v = container.get(container_key or name)
        if v is not None:
            return v
    if family is not None:
        v = family.get(family_key or name)
        if v is not None:
            return v
    if default is not _SENTINEL:
        return default
    looked = ", ".join(
        [f"request.{attr}"]
        + [s for s, _ in extra]
        + [f"container runtime/defaults.json[{container_key or name!r}]",
           f"family config[{family_key or name!r}]"])
    raise MissingRuntimeValue(
        f"{name!r} is required to run this model and nothing declares it. "
        f"Looked in: {looked}."
        + (f" {why}" if why else "")
        + " Declare it in the container's runtime/defaults.json or the family "
          "config, or pass it with the request — the engine will not invent a "
          "value for it.")


def require_max_tokens(defaults, override=None):
    """`max_tokens` from the request, then the container — never a literal.

    `defaults.get("max_tokens", 512)` stood at five sites and `..., 2048)` at
    four others: two literals disagreeing with each other inside one engine. A
    decode bound nobody declared is not a default, it is a claim — it silently
    truncates a long generation, or reserves a cache nobody asked for.

    Every container on this rack that generates declares it (Kokoro 4096,
    whisper 448, TinyLlama through its lm_config), so the value exists; it was
    simply not read as required.
    """
    return resolve("max_tokens", container=(defaults or {}),
                   extra=[("the request", override)],
                   why="It bounds this generation.")
