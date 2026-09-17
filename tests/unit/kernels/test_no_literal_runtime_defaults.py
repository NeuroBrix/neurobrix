"""No literal may stand in for a runtime dimension or parameter.

A literal default here is not a convenience, it is a claim about a request
nobody made, and the engine then plans, sizes and truncates against it.

Measured 2026-09-17 before this was enforced:
  * `height`/`width` fell through to 1024, so Prism planned the SAME 278 MB for
    a 448x448 request and a 160x112 one — 22.4x the pixels. The per-cell memory
    gate is fed that number; it could refuse nothing, and the cell went on to
    hold 8408 MB live and take the machine to 127 MB before the OS killed it.
  * `batch_size=2` was passed for EVERY model with the comment "CFG effectively
    doubles batch" — applied to upscalers and speech models that run no
    guidance, and overriding the `batch_size: 1` diffusion containers declare.
  * `dtype="float16"` was passed for every model, including containers
    declaring float32 (swin2SR) and bfloat16 (TinyLlama).
  * `max_tokens` fell back to 512 at five sites and 2048 at four others — two
    literals disagreeing with each other inside one engine.

Values come from the request, then the container's runtime/defaults.json (or its
manifest), then the family config; when none provides one the engine refuses by
name with what was missing. Absent is legitimate ONLY for a dimension the model
does not have: no VAE, no vae_scale; no spatial extent, no height.

Vendored reference kernels under kernels/triton_kernels_ref are excluded — they
are third-party code, not our dispatch path.
"""
from __future__ import annotations

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
EXCLUDE = ("triton_kernels_ref",)

#: names that describe a runtime dimension or parameter
RUNTIME_NAMES = ("height", "width", "batch_size", "max_tokens", "seq_len",
                 "sequence_length", "num_frames", "vae_scale_factor",
                 "temporal_compression_ratio")

#: `.get("<runtime name>", <a literal that is not None>)`
LITERAL_GET = re.compile(
    r'\.get\(\s*["\'](' + "|".join(RUNTIME_NAMES) + r')["\']\s*,\s*(?!None)([0-9"\'])')


def _is_display_placeholder(line, match):
    """`.get("seq_len", "?")` inside a log line is a placeholder, not a claim.

    The distinction is whether the value could be COMPUTED with. "?" cannot; a
    number or a dtype name can, and those stay flagged.
    """
    tail = line[match.end() - 1:]
    return tail.startswith(("'?'", '"?"'))


def _files():
    for f in sorted(SRC.rglob("*.py")):
        if any(x in str(f) for x in EXCLUDE):
            continue
        yield f


def test_no_literal_default_for_a_runtime_dimension():
    offenders = []
    for f in _files():
        for n, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue                                # a comment describing the old code
            if '"""' in line or "`" in line:
                continue                                # docstrings quoting the old pattern
            m = LITERAL_GET.search(line)
            if m and _is_display_placeholder(line, m):
                continue                                # "?" in a log line, not a value
            if m:
                offenders.append(f"{f.relative_to(SRC)}:{n}: {stripped[:96]}")
    assert not offenders, (
        "a literal default for a runtime dimension or parameter is back:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse core.runtime_values.resolve — request, then container, then "
          "family config, then a refusal naming what was missing.")


def test_input_config_declares_no_literal_defaults():
    from neurobrix.core.prism.profiler import InputConfig
    import dataclasses
    for f in dataclasses.fields(InputConfig):
        if f.name in ("batch_size", "height", "width", "dtype", "vae_scale",
                      "seq_len", "num_frames", "temporal_compression"):
            assert f.default is None, (
                f"InputConfig.{f.name} defaults to {f.default!r}; a config built "
                f"without it would plan for a request nobody made")


def test_the_resolver_refuses_by_name_with_what_was_missing():
    from neurobrix.core.runtime_values import resolve, MissingRuntimeValue
    with pytest.raises(MissingRuntimeValue) as exc:
        resolve("max_tokens", request=None, container={}, family={})
    msg = str(exc.value)
    assert "max_tokens" in msg
    assert "runtime/defaults.json" in msg and "family config" in msg
    assert "will not invent" in msg
