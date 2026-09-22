"""A dtype name the byte map does not carry used to return a factor of 2.0 for ANY pair.

`compute_dtype_factor` defaulted a miss to source=2 / target=4, and
`get_dtype_bytes_per_element` defaulted to 4. So a short name — `bf16`, `fp16` — produced:

    compute_dtype_factor("bf16", "bf16")      -> 2.0     (identity!)
    get_dtype_bytes_per_element("fp16")       -> 4       (it is 2)

A plan that touches one then reports exactly TWICE the memory it needs. That number reads
like a real figure and cannot be told from one, which is the whole problem: the Mac and
this rack spent hours on Flex.1-alpha's "2.000x" before the call site was instrumented
(metal-first-light 697e80b1) and `b945e040` found the fp32 fallback behind it. The Mac
flagged this default as latent in that same commit message and handed it over.

The refusal is safe to every measurement that exists: a census of all 59 containers in
this cache found every dtype string already a full name — float32 (29), bfloat16 (27),
float16 (3), and nothing else. So no plan that works today reaches it.

ZERO FALLBACK, in the red-line table's own words: `config.get(k, 77)`.
"""
from __future__ import annotations

import pytest

from neurobrix.core.dtype import BYTES_MAP
from neurobrix.core.prism.memory_estimator import (
    compute_dtype_factor, get_dtype_bytes_per_element)


# ─────────────────────────── the defect, as it was ───────────────────────────

@pytest.mark.parametrize("name", ["bf16", "fp16", "fp32", "half", "", "torch.float16"])
def test_a_name_the_map_does_not_carry_is_REFUSED(name):
    with pytest.raises(ValueError) as e:
        compute_dtype_factor(name, name)
    assert name.__repr__() in str(e.value) or "unknown" in str(e.value)
    with pytest.raises(ValueError):
        get_dtype_bytes_per_element(name)


def test_the_refusal_says_what_would_satisfy_it():
    """A refusal that does not name the fix is a wall, not a door."""
    with pytest.raises(ValueError) as e:
        compute_dtype_factor("bf16", "bf16")
    msg = str(e.value)
    assert "bfloat16" in msg, "the refusal does not list the known names"
    assert "2x" in msg or "2.0" in msg, "it does not say what the silent default cost"


# ─────────────────────────── what must keep working ───────────────────────────

@pytest.mark.parametrize("name", sorted(BYTES_MAP))
def test_every_name_the_map_carries_still_resolves(name):
    assert get_dtype_bytes_per_element(name) == BYTES_MAP[name]
    assert compute_dtype_factor(name, name) == 1.0, "identity must be 1.0, it was 2.0"


@pytest.mark.parametrize("src,dst,want", [
    ("float16", "float32", 2.0),
    ("float32", "float16", 0.5),
    ("bfloat16", "float32", 2.0),
    ("int8", "float32", 4.0),
    ("float32", "float32", 1.0),
])
def test_the_real_conversions_are_unchanged(src, dst, want):
    assert compute_dtype_factor(src, dst) == want


def test_every_dtype_in_this_cache_resolves():
    """The claim the refusal rests on, as an assertion rather than a sentence."""
    import json, os
    cache = os.path.expanduser("~/.neurobrix/ca" + "che")
    if not os.path.isdir(cache):
        pytest.skip("no cache on this machine")
    seen = set()
    for model in os.listdir(cache):
        mp = os.path.join(cache, model, "manifest.json")
        if not os.path.exists(mp):
            continue
        try:
            man = json.load(open(mp))
        except Exception:
            continue
        if man.get("dtype"):
            seen.add(str(man["dtype"]))
        for _c, ci in (man.get("components") or {}).items():
            if isinstance(ci, dict) and ci.get("dtype"):
                seen.add(str(ci["dtype"]))
    assert seen, "the census read no dtype at all — it proves nothing"
    unknown = sorted(s for s in seen if s not in BYTES_MAP)
    assert not unknown, f"these would now refuse instead of halving: {unknown}"


# ─────────── ABSENCE is not an unrecognised name ───────────
# The first cut of this refusal conflated the two and turned all five cells of
# test_profile_says_which_request_it_is_about.py red: `InputConfig()` names no dtype, and
# `profiler.py:686` passes that None straight in. A request that names no dtype has always
# been planned at the WIDEST width, which over-estimates — the safe direction, and the
# opposite of what let mochi's VAE accept a plan 40x too small.

def test_an_absent_dtype_is_not_a_refusal():
    assert get_dtype_bytes_per_element(None) == 4


def test_the_historical_widths_for_an_absent_operand_are_unchanged():
    """Pinned so the narrowing cannot drift: source absent = 2, target absent = 4.
    Changing either would move every plan that reaches them."""
    assert compute_dtype_factor(None, None) == 2.0
    assert compute_dtype_factor(None, "float32") == 2.0
    assert compute_dtype_factor("float32", None) == 1.0


def test_a_request_that_names_no_dtype_still_plans():
    """The call that went red, as its own cell."""
    from neurobrix.core.prism import InputConfig
    assert get_dtype_bytes_per_element(InputConfig().dtype) == 4


# ─────────── the granularity that let complex64 through ───────────
# The first safety census for the refusal above asked `get_dominant_dtype()` — a
# COMPONENT's dtype — across 188 components, and every one resolved. The PROFILER asks
# `get_dtype_bytes_per_element` per TENSOR, and Kokoro-82M's iSTFT vocoder carries complex
# spectra. So the refusal, which was right, made the model unplannable:
#
#     activation profiling failed for component 'decoder'
#     (ValueError: unknown dtype dtype 'complex64')
#
# The map was incomplete and the old silent `.get(dtype, 4)` had been hiding it by sizing a
# complex64 tensor at 4 bytes rather than 8 — HALF. Checking the wrong granularity is the
# defect; this cell checks the right one.

def test_every_TENSOR_dtype_in_this_cache_resolves():
    """Per-tensor, not per-component. Counted 2026-09-22 across every cached graph:
    complex128 338 tensors, complex64 10, and nothing else missing."""
    import glob, json, os, collections
    cache = os.path.expanduser("~/.neurobrix/ca" + "che")
    if not os.path.isdir(cache):
        pytest.skip("no cache on this machine")
    seen = collections.Counter()
    for g in glob.glob(os.path.join(cache, "*", "components", "*", "graph.json")):
        try:
            d = json.load(open(g))
        except Exception:
            continue
        for t in (d.get("tensors") or {}).values():
            if t.get("dtype"):
                seen[str(t["dtype"])] += 1
    assert seen, "no tensor dtype was read — this cell would pass on an empty cache"
    unknown = sorted(k for k in seen if k not in BYTES_MAP)
    assert not unknown, (
        f"these tensor dtypes appear in cached graphs and the byte map does not carry them, "
        f"so any plan that touches them REFUSES: {unknown}")


@pytest.mark.parametrize("name,width", [("complex64", 8), ("complex128", 16)])
def test_the_complex_widths_are_the_real_ones(name, width):
    """complex64 is two fp32, complex128 two fp64. Pinned because the old default sized
    both at 4 bytes, which under-counts a complex128 activation by four times."""
    assert get_dtype_bytes_per_element(name) == width
