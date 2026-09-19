"""A certified configuration is a property of the compiler that produced it.

`entry_for_memory_class` has refused to serve an entry across memory classes
since register 56: a setting proven on a 16 GB card is not served to a 32 GB one.
The same argument applies to the code generator, and the engine already half
agreed with itself -- `entry_covers(..., need_generator=...)` encodes the rule and
is called from exactly ONE place, `autotune_certify.py`, which is the CERTIFIER
deciding what to re-prove. Nothing asked the question at the serving side.

MEASURED 2026-09-19, which is why this matters now. All 9,655 NVIDIA entries on
`main` carry `triton 3.6.0`. The production stack runs 3.6.0; the candidate stack
runs 3.8.0; and a fifteen-hour re-proof under 3.8.0 produced 10,030 entries that
sit in `campaigns/2026_09_16_converge/stack/reproof32`, unmerged. Both sets live at
the same paths, so merging replaces: without this check, whichever set is on disk
is served to whichever compiler is running.

WHAT IS DELIBERATELY NOT CLAIMED. The owner's note of 2026-09-16, recorded in
`proof_backend`'s own docstring, says a setting stays CORRECT under any generator
-- the fp64 oracle proved the source, not the compiler -- and what a newer
generator may age is its RANK as the fastest. So this is not a correctness guard.
It refuses to present a stale rank as a certified one, and `NBX_AUTOTUNE_ANY_GENERATOR=1`
serves them anyway for whoever wants the old behaviour, named so it reads as
deliberate.

SEEN RED: with the generator block removed from `lookup()` so it ends at
`return entry_for_memory_class(entry, memory_class)`,
`test_an_entry_proven_under_another_compiler_is_not_served` fails.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_a_certified_setting_is_served_only_to_the_compiler_that_proved_it.py
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import autotune_certified as C

_CFG = {"kwargs": {"BLOCK_M": 32}, "num_warps": 2, "num_stages": 4}


def _entry(triton_version, mem_mb=16384, variants=None):
    e = {"config": _CFG,
         "proof": {"date": "2026-09-14T00:00:00+00:00",
                   "engine_version": "0.5.4",
                   "backend": {"triton": triton_version, "name": "cuda"},
                   "machine": {"device": {"memory_mb": mem_mb}},
                   "deviation": 1e-7, "tolerance": 1e-4}}
    if variants:
        e["variants"] = variants
    return e


def test_the_running_generator_is_nameable():
    """Nothing can be refused for not matching an answer that does not exist."""
    gen = C.running_generator()
    assert gen is None or (isinstance(gen, str) and gen.startswith("triton ")), gen


def test_an_entry_proven_under_the_running_compiler_is_served():
    gen = C.running_generator()
    if gen is None:
        pytest.skip("no Triton here, so there is no generator to match")
    ver = gen.split()[1]
    assert C.entry_for_generator(_entry(ver), gen) is not None


def test_an_entry_proven_under_another_compiler_is_not_served():
    """The refusal itself."""
    gen = C.running_generator()
    if gen is None:
        pytest.skip("no Triton here")
    ver = gen.split()[1]
    other = "9.9.9" if ver != "9.9.9" else "8.8.8"
    assert C.entry_for_generator(_entry(other), gen) is None, (
        f"an entry proven under triton {other} was served to {gen}")


def test_a_variant_proven_under_the_running_compiler_is_found():
    """Same shape as `entry_for_memory_class`: the primary, or a variant.

    Without this the check would refuse an entry that DOES carry a matching
    proof, just not in its primary slot -- turning a correct serve into a
    needless runtime sweep.
    """
    gen = C.running_generator()
    if gen is None:
        pytest.skip("no Triton here")
    ver = gen.split()[1]
    e = _entry("9.9.9", variants={"g16": _entry(ver)})
    got = C.entry_for_generator(e, gen)
    assert got is not None
    assert got["proof"]["backend"]["triton"] == ver


def test_an_unanswerable_question_refuses_nothing():
    """`running_generator()` None means the engine cannot say what it runs.

    Refusing everything because the question is unanswerable is worse than
    serving: it would empty the directory on any machine whose Triton import
    fails, and that is a silence dressed as caution.
    """
    assert C.entry_for_generator(_entry("3.6.0"), None) is None
    # and lookup's own guard is the thing that matters -- it returns the entry
    # rather than refusing when the generator cannot be named. Pinned by
    # reading the source, because constructing that state needs Triton absent.
    import inspect
    src = inspect.getsource(C.lookup)
    assert "if gen is None:" in src and "return cert" in src, (
        "lookup() must serve, not refuse, when the running generator cannot "
        "be named")


def test_the_opening_is_named_and_deliberate():
    """One opening, spelled so a reader sees it was meant."""
    import inspect
    src = inspect.getsource(C.lookup)
    assert "NBX_AUTOTUNE_ANY_GENERATOR" in src


def test_the_directory_on_disk_is_all_one_generator_and_it_is_named():
    """A census, so the docstring's numbers are recomputed rather than trusted."""
    import json, glob, collections
    c = collections.Counter()
    for f in glob.glob("src/neurobrix/config/autotune/nvidia/**/*.json", recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in (d.get("entries", d) or {}).items():
            if isinstance(v, dict) and isinstance(v.get("proof"), dict):
                c[C.proof_backend(v["proof"])] += 1
    assert c, "no NVIDIA certified entries found — this cell would pass vacuously"
    assert len(c) == 1, f"the NVIDIA directory mixes generators: {dict(c)}"


def test_lookup_itself_refuses_the_mismatched_entry(monkeypatch, capsys):
    """End to end through `lookup`, which is what the launch site calls.

    `entry_for_generator` being right is necessary and not sufficient: the
    refusal has to reach the serving path. This drives `lookup` with a directory
    of one entry and no card involved.
    """
    gen = C.running_generator()
    if gen is None:
        pytest.skip("no Triton here")
    ver = gen.split()[1]
    other = "9.9.9" if ver != "9.9.9" else "8.8.8"

    class _Tuner:  # only its identity is used by the code paths we touch
        pass

    key = (10, 1024, 1024, True, True, "fp32", "fp16", "fp32")
    monkeypatch.setattr(C, "enabled", lambda: True)
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))
    monkeypatch.setattr(C, "output_dtype", lambda tuner, k: "fp32")
    monkeypatch.setattr(C, "executing_memory_class", lambda *a, **k: 16)

    def _entries(version):
        return {C.key_repr(key): _entry(version)}

    # (a) the matching generator is served
    monkeypatch.setattr(C, "_load", lambda *a, **k: _entries(ver))
    assert C.lookup("matmul_kernel", _Tuner(), key, memory_class=16) is not None

    # (b) the mismatched one is NOT, and the engine says so
    C._GEN_REFUSED.clear()
    monkeypatch.setattr(C, "_load", lambda *a, **k: _entries(other))
    assert C.lookup("matmul_kernel", _Tuner(), key, memory_class=16) is None, (
        "lookup served a setting proven under another compiler")
    said = capsys.readouterr().out
    assert "certified under" in said and other in said and ver in said, (
        f"the refusal has to be said in clear; got: {said!r}")

    # (c) it is said ONCE per kernel, not once per shape: 9,655 entries would
    #     otherwise print 9,655 lines.
    C.lookup("matmul_kernel", _Tuner(), key, memory_class=16)
    assert capsys.readouterr().out == "", "the refusal repeated for the same kernel"

    # (d) the named opening serves it anyway
    monkeypatch.setenv("NBX_AUTOTUNE_ANY_GENERATOR", "1")
    assert C.lookup("matmul_kernel", _Tuner(), key, memory_class=16) is not None
