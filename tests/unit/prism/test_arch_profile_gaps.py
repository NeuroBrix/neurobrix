"""A card without a profile is REFUSED by name, never given a neighbour's.

Three NVIDIA profiles exist — volta (7.0), ampere (8.0), hopper (9.0) — and
architectures are resolved by capability MAJOR. That silently gave every part
sharing a major the datacentre part's profile, and its values are wrong for
them in the UNSAFE direction: a shared-memory budget larger than the card has
offers autotune tiles it cannot hold.

Measured 2026-09-04, against each vendor's published figures:

    T4 (7.5)          64 KB   was inheriting volta's   96 KB
    A10 / 3090 (8.6) 100 KB   was inheriting ampere's 164 KB
    L4 / 4090 (8.9)  100 KB   was inheriting ampere's 164 KB

Latent until the autotune tile filter began reading that value. The remedy is
NOT to estimate the right numbers — guessing a block size produces a wrong
result rather than an error, which is the whole reason `get_vendor_config` is
zero-fallback. It is to say plainly that the profile is missing, name it, and
say what closing the gap needs.
"""

from __future__ import annotations

import pytest

from neurobrix.core.config.loader import (
    UnsupportedArchitectureError,
    get_vendor_config,
)
from neurobrix.core.prism.autodetect import _nvidia_cc_to_arch


@pytest.mark.parametrize("cc,arch,card", [
    ("7.5", "turing", "T4 / RTX 2080"),
    ("8.6", "ampere_consumer", "A10 / RTX 3090"),
    ("8.9", "ada", "L4 / RTX 4090"),
])
def test_an_unprofiled_card_is_named_and_refused(cc, arch, card):
    assert _nvidia_cc_to_arch(cc) == arch, (
        f"{card} (cc {cc}) must resolve to its OWN architecture name, not to "
        f"the datacentre part that shares its major version"
    )
    with pytest.raises(UnsupportedArchitectureError) as excinfo:
        get_vendor_config("nvidia", arch)
    assert arch in str(excinfo.value), "the refusal must name what is missing"


@pytest.mark.parametrize("cc,arch", [("7.0", "volta"), ("8.0", "ampere"),
                                     ("9.0", "hopper")])
def test_the_profiled_cards_still_load(cc, arch):
    """The refusals above must not have cost the three cards that ARE
    described. This is the regression that would matter."""
    assert _nvidia_cc_to_arch(cc) == arch
    cfg = get_vendor_config("nvidia", arch)
    assert cfg["memory"]["max_shared_memory_per_block"] > 0


def test_the_refusal_says_what_would_close_the_gap():
    """A refusal that only says "no" leaves the reader with no next step. It
    must name the file to write and the keys it must carry — and say the
    values come from the vendor's specification, not from a neighbour."""
    with pytest.raises(UnsupportedArchitectureError) as excinfo:
        get_vendor_config("nvidia", "turing")
    msg = str(excinfo.value)
    for needed in ("max_shared_memory_per_block", "compute_capability",
                   "block_sizes", "pipelining.max_num_stages", "volta.yml"):
        assert needed in msg, f"the refusal does not mention {needed!r}"


def test_no_profile_claims_to_describe_a_card_it_does_not():
    """ampere.yml is the A100. If someone later widens it to cover 8.6 and 8.9
    by editing its compute_capability, this fails — those parts have a
    different shared-memory budget and need their own file."""
    cfg = get_vendor_config("nvidia", "ampere")
    assert str(cfg["compute_capability"]).strip() == "8.0"
