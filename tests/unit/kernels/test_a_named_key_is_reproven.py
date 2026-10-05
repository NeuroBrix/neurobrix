"""`--reprove-keys`: the keys a file names are swept again even when this class holds them.

A timing taken while another job shared the card (card 2, 2026-10-05 04:45: a certify beside a
20 GB model run, 13 keys) may rank the wrong configuration first; the certificate is still
correct (its deviation is checked) but its choice is suspect. Coverage must not let those keys
by, every other key keeps `--only-missing`'s rule, and a named key the census table does not
hold is refused by name — a re-prove that silently matches nothing is the empty gate.

Injection (2026-10-05): with `skips_key` ignoring `named`, test_a_named_covered_key_is_not_passed_by
went RED; restored, green.
"""
import pytest

from neurobrix.kernels import autotune_certify as A


class _Tuner:
    keys = ["M_BUCKET", "N_BUCKET", "K_BUCKET", "IEEE_PRECISION"]
    arg_names = ["a_ptr", "b_ptr", "c_ptr"]


QUAL = "neurobrix.kernels.ops.baddbmm.baddbmm_kernel"
KEY = (384, 384, 384, True, "fp32", "fp32", "fp32")
OTHER = (768, 384, 384, True, "fp32", "fp32", "fp32")


def test_the_identity_is_the_log_line():
    assert A.key_identity(QUAL, "fp32", _Tuner(), KEY) == \
        "baddbmm_kernel fp32 M_BUCKET=384 N_BUCKET=384 K_BUCKET=384 IEEE_PRECISION=True fp32,fp32,fp32"


def test_a_log_line_is_read_back_as_its_identity():
    named = A.named_key_set([
        "[certify] baddbmm_kernel fp32 M_BUCKET=384 N_BUCKET=384 K_BUCKET=384 IEEE_PRECISION=True fp32,fp32,fp32",
        "# a comment", "",
        "baddbmm_kernel fp32 M_BUCKET=768 N_BUCKET=384 K_BUCKET=384 IEEE_PRECISION=True fp32,fp32,fp32: {'BLOCK_M': 64}",
    ])
    tuners = {QUAL: _Tuner()}
    assert named == A.census_identities({QUAL: [KEY, OTHER]}, tuners)


def test_a_named_covered_key_is_not_passed_by():
    ident = A.key_identity(QUAL, "fp32", _Tuner(), KEY)
    assert A.skips_key(True, ident, {ident}) is False


def test_an_unnamed_covered_key_is_passed_by_and_a_missing_one_is_not():
    ident = A.key_identity(QUAL, "fp32", _Tuner(), KEY)
    assert A.skips_key(True, ident, {"something else"}) is True
    assert A.skips_key(False, ident, {"something else"}) is False
    assert A.skips_key(True, ident, None) is True


def test_an_empty_list_is_refused_by_name():
    with pytest.raises(RuntimeError, match="names no key"):
        A.named_key_set(["", "# only a comment"])


def test_a_named_key_outside_the_table_is_visible_to_the_door():
    named = A.named_key_set(["baddbmm_kernel fp32 M_BUCKET=1 N_BUCKET=1 K_BUCKET=1 IEEE_PRECISION=True fp32,fp32,fp32"])
    assert named - A.census_identities({QUAL: [KEY]}, {QUAL: _Tuner()})
