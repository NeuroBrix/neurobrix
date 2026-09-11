"""The detector separates a per-element `other` from a splat, in both senses.

It exists because the refusal it measures is OURS. Upstream lowers
`tt.load(ptr, mask, other)` with a per-element `other` correctly on the
ordinary path — it indexes `other[i]` beside `ptr[i]`. Only the cooperative
staged fill cannot, because it visits elements no thread loaded and so has no
`i`; the repair for that fill's missing mask therefore refuses the case. A
refusal we add ourselves can only take away shapes that worked the day before,
so its reach is counted before it lands, and a count is worth exactly what its
detector is worth.

Three ways to be worthless here, and the third is the one that bites:

  * finding nothing — the census then reports zero over an empty question.
  * finding everything — every `dense<0.0>` counted as per-element makes the
    refusal look enormous and the wrong decision gets taken.
  * finding what is uniform ONE CAST AWAY. `arith.sitofp %splat` produces a
    tensor whose elements are all equal. A detector that stops at the
    defining op calls it per-element and over-reports. That is the same class
    as the staged-fill detector that read the emission line while the masking
    happened in a helper: reading where the value is NAMED, not where it is
    WRITTEN.

The IR below is not invented. It is what triton emitted for a kernel written
to carry all four forms at once, read back from `compiled.asm["ttir"]`.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_masked_other_detector.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from masked_other_census import per_element_other, _is_uniform, _defs  # noqa: E402


#: Emitted by triton for a kernel with, in order: a load whose `other` is
#: `offs.to(tl.float32)` (varies), one whose `other` is `3.0` (a splat), one
#: whose `other` is a splat behind a cast, and one with no mask at all.
EMITTED = """
%b = arith.constant dense<3.000000e+00> : tensor<64xf32> loc(#loc16)
%m = tt.splat %n : i32 -> tensor<64xi32> loc(#loc18)
%arr = arith.sitofp %offs : tensor<64xi32> to tensor<64xf32> loc(#loc19)
%u = arith.sitofp %m : tensor<64xi32> to tensor<64xf32> loc(#loc23)
%a_2 = tt.load %a_1, %m_0, %arr : tensor<64x!tt.ptr<f32>> loc(#loc21)
%b_3 = tt.load %a_1, %m_0, %b : tensor<64x!tt.ptr<f32>> loc(#loc16)
%u_4 = tt.load %a_1, %m_0, %u : tensor<64x!tt.ptr<f32>> loc(#loc23)
%c = tt.load %a_1 : tensor<64x!tt.ptr<f32>> loc(#loc22)
"""


def _names(ttir):
    return sorted(f["other"] for f in per_element_other(ttir))


def test_a_genuinely_varying_other_is_seen():
    assert "%arr" in _names(EMITTED), (
        "`other = offs.to(tl.float32)` differs in every element and is exactly "
        "the case the staged fill cannot re-index. A detector that misses it "
        "reports a refusal as free when it is not.")


def test_a_dense_splat_is_not_counted():
    assert "%b" not in _names(EMITTED), (
        "`other=3.0` lowers to `arith.constant dense<3.0>` — one value in "
        "every element, which the staged fill emits without indexing anything. "
        "Counting it would inflate the refusal's reach.")


def test_a_splat_one_cast_away_is_not_counted():
    """The row the first version of this detector got wrong.

    `arith.sitofp` of a `tt.splat` is still uniform. Stopping at the defining
    op — reading where the value is named rather than where it is written —
    calls it per-element.
    """
    assert "%u" not in _names(EMITTED), (
        "a splat behind a cast is uniform; the detector must chase the cast "
        "chain to its root, not read the op that names the value")


def test_an_unmasked_load_is_not_counted():
    assert _names(EMITTED) == ["%arr"], (
        "`tt.load %ptr` has one operand: no mask, so no `other`, so nothing "
        "for the refusal to bite on")


def test_what_it_cannot_resolve_is_not_called_uniform():
    """Unknown must mean not-uniform, never uniform.

    The two errors are not symmetric. Over-reporting shows up as a list to
    read; under-reporting shows up as a landed refusal breaking a model, and
    nothing in the census would have said so.
    """
    assert _is_uniform("%never_defined", _defs(EMITTED)) is False


def test_the_detector_is_not_vacuous_on_its_own_corpus():
    """Both directions on one text: it must find one and reject three.

    Either half alone is satisfiable by a detector that is always wrong: one
    that answers yes to everything passes the first test, one that answers no
    to everything passes the other three.
    """
    found = _names(EMITTED)
    assert found == ["%arr"], f"expected exactly ['%arr'], got {found}"


# ── the static scan: it must be able to say `per_element` at all ───────────
#
# The scan returned "255 of 255 scalar, 0 per-element" over the whole kernel
# tree. That is the answer that most needs a mutation behind it: a classifier
# hard-wired to return "scalar" produces exactly the same line, and the
# refusal would then land on a count that was never taken.

import ast                                                       # noqa: E402

from masked_other_census import _classify_other                   # noqa: E402


def _verdict(src: str, expr: str) -> str:
    tree = ast.parse(src)
    node = ast.parse(expr, mode="eval").body
    return _classify_other(node, tree)[0]


_TREE = """
min_val = float('-inf')
_MIN = tl.constexpr(-2147483648)
boxed = _MIN
arr = offs[:, None]
"""


def test_the_scan_can_say_per_element():
    assert _verdict(_TREE, "arr") == "per_element", (
        "a subscript with a None axis is the archetype of a per-element "
        "`other`; a classifier that cannot return this verdict makes every "
        "zero it reports meaningless")


def test_the_scan_follows_a_name_to_its_value():
    assert _verdict(_TREE, "min_val") == "scalar", (
        "`other=min_val` must be resolved to `float('-inf')`, not reported as "
        "an unknown name — reading where the value is named is the fourth "
        "vacuous-guard form")


def test_the_scan_chases_a_constexpr_box():
    assert _verdict(_TREE, "boxed") == "scalar", (
        "`tl.constexpr(-2147483648)` is a scalar, through two levels: the "
        "name `boxed`, then the box itself")


def test_what_it_cannot_chase_is_never_called_scalar():
    assert _verdict(_TREE, "who_knows") == "unresolved", (
        "an unchaseable `other` must surface as unresolved. Calling it scalar "
        "under-reports, and an under-reported count is a refusal landing "
        "blind")
