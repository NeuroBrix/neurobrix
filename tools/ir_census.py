#!/usr/bin/env python3
"""The brick under every "what does a model actually reach" count.

It was written twice before it was named: once for the broadcast-shape
collision of upstream issue #9, once for the per-element `other` of a masked
load. Both do the same three things and only the middle one differs — wrap
triton's compiler in-process, run a catalogue model, apply a predicate to each
compiled kernel's TTIR. So the wrapper is the brick and the predicate is the
argument, rather than a second copy of the wrapper with a different regex in
the middle.

What it refuses to do is report a zero it did not earn. A run that compiled
nothing returns a REFUSAL, not "0 carriers": the sentence "no kernel carries
it" is true of an empty set and says nothing about the question, which is the
vacuous guard in its purest form.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def speaker_for(model: str) -> str:
    """The voice this artefact requires, or "" — first in sorted order, which
    is the enumeration the engine itself does. An artefact that ships voices
    and declares no default is refused before a single kernel compiles, and
    that refusal must not be read as a census of zero."""
    cache = Path.home() / ".neurobrix" / "cache" / model
    voices = cache / "modules" / "voices"
    if not voices.is_dir():
        return ""
    try:
        if json.loads((cache / "runtime" / "defaults.json").read_text()).get("voice"):
            return ""
    except (OSError, ValueError):
        pass
    available = sorted(p.stem for p in voices.glob("*.pt"))
    return available[0] if available else ""


def compile_census(model: str, arm: str, inspect, extra_argv=None) -> dict:
    """Run `model` and return {kernel_name: {"compilations": n, "findings": [...]}}.

    `inspect(ttir) -> list` is applied to every kernel compiled. Findings are
    de-duplicated per kernel, so a kernel compiled ten times with the same
    shape counts once as a carrier and ten as compilations.
    """
    # BOTH bindings. `triton/compiler/__init__.py` does `from .compiler import
    # compile`, so `triton.compiler.compile` is a SEPARATE name bound at import
    # — and it is the one the engine's launcher imports. Patching only the
    # module the function lives in wraps nothing, and the first run of the
    # broadcast census reported a census over an empty set for exactly that.
    # A census compiles. It therefore owns its cache or it does not run: a
    # stashed kernel makes the compiler return without reaching the code the
    # census is about, and the census then counts zero and says so.
    from check_measurement_environment import enforce_owned_cache
    enforce_owned_cache("an IR census")

    import triton.compiler as tc
    import triton.compiler.compiler as tcc

    seen: dict[str, dict] = {}
    real = tcc.compile

    def wrapped(src, target=None, options=None):
        name = getattr(getattr(src, "fn", None), "__name__", None) or str(src)
        try:
            compiled = real(src, target=target, options=options)
        except BaseException:
            # A kernel the backend REFUSES never reaches `compiled.asm`, so an
            # inspection that only runs on success is blind to exactly the
            # kernels a blocked model is blocked on -- the census would report
            # a clean zero over the population it was pointed at. It cannot be
            # inspected from here, so it is COUNTED and named instead: a hole
            # that is stated is a different object from one that is silent.
            row = seen.setdefault(name, {"compilations": 0, "findings": [],
                                         "refused": 0})
            row["refused"] = row.get("refused", 0) + 1
            raise
        try:
            row = seen.setdefault(name, {"compilations": 0, "findings": [],
                                         "refused": 0})
            row["compilations"] += 1
            for f in inspect(compiled.asm.get("ttir", "") or ""):
                if f not in row["findings"]:
                    row["findings"].append(f)
        except Exception as exc:                       # never break a run
            print(f"[census] inspection failed: {exc}", file=sys.stderr)
        return compiled

    tcc.compile = wrapped
    tc.compile = wrapped
    try:
        from neurobrix.cli import main as cli_main
        argv = ["neurobrix", "run", "--model", model]
        argv += list(extra_argv or ["--prompt", "a red apple on a wooden table",
                                    "--max-tokens", "8"])
        voice = speaker_for(model)
        if voice:
            argv += ["--speaker", voice]
        if arm == "triton":
            argv.append("--triton")
        sys.argv = argv
        try:
            cli_main()
        except SystemExit:
            pass
    finally:
        tcc.compile = real
        tc.compile = real
    return seen


def refuse_if_empty(seen: dict, model: str, arm: str, question: str) -> int | None:
    if seen:
        return None
    print(f"NOTHING WAS COMPILED by {model} ({arm}): the run reached no Triton "
          f"compilation at all, so this says nothing about {question}. Check "
          f"that the run started — a census over an empty set is not a result.")
    return 2
