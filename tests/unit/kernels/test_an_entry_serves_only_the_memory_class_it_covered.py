"""Register entry 56 — a directory keyed by a profile two memory classes share.

This rack carries two V100 SKUs under ONE vendor profile (`nvidia/volta`):
16 GB cards 0 and 1, 32 GB cards 2 and 3. The certified directory was keyed
by (vendor, profile, kernel, dtype, shape) and served any entry to any card.
The rule: a proof says which card's memory it was made on, and an entry
serves only the memory class it covered. A class it does not cover sweeps at
runtime, announced — never silently served from a proof made elsewhere.

Shapes: memory classes 16 and 32 are the rack's; a third class (24) stands
for a card no proof was ever made on. Each test names why its class differs
from the proof's — a test at the proof's own class is green for the reason
that blinds it.
"""
import json
from pathlib import Path

import pytest

from neurobrix.kernels import autotune_certified as C


def _proof(memory_mb=None, hardware_profile=None):
    p = {"date": "2026-09-13T22:00:00+00:00", "engine_version": "0.5.3",
         "backend": {"name": "cuda", "triton": "3.6.0"}, "shape": [10, 1536, 1536],
         "deviation": 1e-6, "tolerance": 1e-4, "oracle": "fp64", "machine": {"hostname": "t"},
         "built": {"gpu": True}}
    if hardware_profile is not None:
        p["machine"]["hardware_profile"] = hardware_profile
    if memory_mb is not None:
        p["machine"]["device"] = {"ordinal": 0, "visible_devices": "2", "name": "Tesla V100-SXM2-32GB", "memory_mb": memory_mb}
    return p


def _entry(config_kw, memory_mb=None, hardware_profile=None):
    return {"config": {"kwargs": config_kw, "num_warps": 4, "num_stages": 3},
            "proof": _proof(memory_mb, hardware_profile), "excluded": []}


# --- the class a proof covers -------------------------------------------------

def test_a_proof_with_a_device_record_names_its_memory_class():
    assert C.proof_memory_class(_proof(memory_mb=32768)) == 32
    assert C.proof_memory_class(_proof(memory_mb=16384)) == 16
    # the same rounding as the auto-profile name (autodetect: round(mb / 1024))
    assert C.proof_memory_class(_proof(memory_mb=16160)) == 16


def test_a_legacy_single_card_profile_name_is_read_as_its_class():
    assert C.proof_memory_class(_proof(hardware_profile="auto-v100-16gb-16g")) == 16
    assert C.proof_memory_class(_proof(hardware_profile="auto-v100-32gb-32g")) == 32


def test_a_legacy_rig_profile_name_does_not_say_which_card_ran():
    """`auto-4xv100-16gb-96.0g` names the SUM and the first card's model; the
    certifying card is unknown — `?`, never a plausible reconstruction."""
    assert C.proof_memory_class(_proof(hardware_profile="auto-4xv100-16gb-96.0g")) is None
    assert C.proof_memory_class(_proof()) is None


# --- serving: only the class covered -----------------------------------------

def test_an_entry_proven_on_32g_is_not_served_to_a_16g_card():
    e = _entry({"BLOCK_M": 64}, memory_mb=32768)
    assert C.entry_for_memory_class(e, 32) is e
    assert C.entry_for_memory_class(e, 16) is None
    assert C.entry_for_memory_class(e, 24) is None


def test_an_entry_proven_on_16g_is_not_served_to_a_32g_card_either():
    """The rule is coverage, not 'more memory is fine': a 16g proof covers 16g."""
    e = _entry({"BLOCK_M": 64}, memory_mb=16384)
    assert C.entry_for_memory_class(e, 16) is e
    assert C.entry_for_memory_class(e, 32) is None


def test_a_variant_serves_its_own_class_with_its_own_config():
    e = _entry({"BLOCK_M": 64}, memory_mb=16384)
    e["variants"] = {"32g": _entry({"BLOCK_M": 128}, memory_mb=32768)}
    served = C.entry_for_memory_class(e, 32)
    assert served is not None and served["config"]["kwargs"] == {"BLOCK_M": 128}
    assert C.entry_for_memory_class(e, 16)["config"]["kwargs"] == {"BLOCK_M": 64}
    assert C.covered_memory_classes(e) == {16, 32}


def test_an_unknown_class_proof_is_served_to_no_card():
    e = _entry({"BLOCK_M": 64}, hardware_profile="auto-4xv100-16gb-96.0g")
    assert C.covered_memory_classes(e) == set()
    assert C.entry_for_memory_class(e, 16) is None
    assert C.entry_for_memory_class(e, 32) is None


def test_an_unknown_executing_class_is_served_nothing():
    """The engine that cannot say which card it is on cannot say it is covered."""
    e = _entry({"BLOCK_M": 64}, memory_mb=16384)
    assert C.entry_for_memory_class(e, None) is None


# --- the file gate: a variant's slot name must be its proof's class ------------

def test_validate_refuses_a_variant_filed_under_another_class(tmp_path):
    root = tmp_path / "nvidia" / "volta"; root.mkdir(parents=True)
    path = root / "matmul_kernel.fp32.json"
    e = _entry({"BLOCK_M": 64}, memory_mb=16384)
    e["variants"] = {"32g": _entry({"BLOCK_M": 128}, memory_mb=16384)}   # filed as 32g, proven on 16g
    doc = {"format": "nbx-autotune-certified/2", "vendor": "nvidia", "profile": "volta",
           "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "dtype": "fp32",
           "entries": {"(10, 1536, 1536)": e}}
    path.write_text(json.dumps(doc))
    problems = C.validate(doc, path)
    assert any("variant 32g" in p and "16" in p for p in problems), problems
    e["variants"]["32g"] = _entry({"BLOCK_M": 128}, memory_mb=32768)
    assert not [p for p in C.validate(doc, path) if "variant" in p]


# --- the certifier's only-missing: missing FOR the certifying class ----------

def test_only_missing_is_asked_per_memory_class():
    entries = {"k": _entry({"BLOCK_M": 64}, memory_mb=16384)}
    assert C.entry_covers(entries, "k", 16) is True
    assert C.entry_covers(entries, "k", 32) is False
    assert C.entry_covers(entries, "absent", 16) is False


def test_filing_a_certification_places_it_by_class():
    entries = {}
    C.file_certification(entries, "k", _entry({"BLOCK_M": 64}, memory_mb=16384))
    assert entries["k"]["config"]["kwargs"] == {"BLOCK_M": 64} and "variants" not in entries["k"]
    C.file_certification(entries, "k", _entry({"BLOCK_M": 128}, memory_mb=32768))
    assert entries["k"]["config"]["kwargs"] == {"BLOCK_M": 64}          # the primary stays
    assert entries["k"]["variants"]["32g"]["config"]["kwargs"] == {"BLOCK_M": 128}
    C.file_certification(entries, "k", _entry({"BLOCK_M": 32}, memory_mb=16384))   # re-proof of the primary's class
    assert entries["k"]["config"]["kwargs"] == {"BLOCK_M": 32}
    assert entries["k"]["variants"]["32g"]["config"]["kwargs"] == {"BLOCK_M": 128}


def test_a_certification_without_a_device_record_is_refused_from_the_file():
    """A new proof that cannot say its card would be an entry no card is ever served."""
    with pytest.raises(ValueError):
        C.file_certification({}, "k", _entry({"BLOCK_M": 64}))


# --- the ONE rule: the profile name's grammar and its reading are the same function ---

def test_the_profile_name_and_its_reading_share_one_rule():
    """Register 56 review: `round(mb / 1024)` and the `auto-<model>-<N>g`
    grammar were written twice (autodetect builds, the directory parsed). Now
    the builder and the parser live together in Prism's structure module and
    this pins their round trip — for the rack's two cards and an odd MB count."""
    from neurobrix.core.prism.structure import (memory_class_gb, single_card_profile_id,
                                                rig_profile_id, memory_class_from_profile_id)
    for mb in (16384, 16160, 32768, 32510, 24576):
        assert memory_class_from_profile_id(single_card_profile_id("v100-16gb", mb)) == memory_class_gb(mb)
    assert memory_class_from_profile_id(rig_profile_id(4, "v100-16gb", 96.0)) is None
    assert memory_class_from_profile_id("auto-m4pro-cpu-48g") == 48   # a host profile's memory is still a class
    assert memory_class_from_profile_id("") is None and memory_class_from_profile_id(None) is None


# --- the path this rack takes: unpinned, two memory classes visible -----------

class _Tuner:
    keys = ["M", "N", "K", "IEEE_PRECISION", "PROMOTE_B"]
    arg_names = ["a_ptr", "b_ptr", "c_ptr", "M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn",
                 "stride_cm", "stride_cn", "IEEE_PRECISION", "PROMOTE_B"]
    base_fn = type("F", (), {"__name__": "matmul_kernel"})

    def __init__(self):
        self.cache = {}
        self.nargs = {}


KERNEL = "neurobrix.kernels.ops.matmul.matmul_kernel"
KEY = (1500, 1280, 1280, True, True, "fp32", "fp16", "fp32")


def _tensor_on(idx):
    return type("T", (), {"_device": "cuda", "_device_idx": idx, "dtype": "fp16"})()


@pytest.fixture
def two_class_rig(tmp_path, monkeypatch):
    """A profile with a 16 GB card at ordinal 0 and a 32 GB card at ordinal 1,
    and a directory whose only entry was proven on the 16 GB card."""
    from neurobrix.kernels import wrappers as W
    before = (W._NBX_HW_PROFILE, W._NBX_HAS_NATIVE_BF16)
    d16 = type("Dev", (), {"index": 0, "name": "Tesla V100-SXM2-16GB", "memory_mb": 16384})()
    d32 = type("Dev", (), {"index": 1, "name": "Tesla V100-SXM2-32GB", "memory_mb": 32768})()
    W.set_hardware_profile(type("Prof", (), {"devices": [d16, d32], "has_native_bf16": False})())
    monkeypatch.setenv("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR", str(tmp_path))
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))
    root = tmp_path / "nvidia" / "volta"; root.mkdir(parents=True)
    e = _entry({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, memory_mb=16384)
    e["proof"]["shape"] = list(KEY)
    (root / "matmul_kernel.fp32.json").write_text(json.dumps(
        {"format": C.FORMAT, "vendor": "nvidia", "profile": "volta", "kernel": KERNEL, "dtype": "fp32",
         "entries": {C.key_repr(KEY): e}}))
    C.reset()
    yield
    C.reset()
    W._NBX_HW_PROFILE, W._NBX_HAS_NATIVE_BF16 = before


def test_on_a_two_class_rig_the_card_is_unknown_at_seed_time(two_class_rig):
    assert C.executing_memory_class() is None, "two classes visible, no tensor: unknown"
    assert C.executing_memory_class([_tensor_on(0)]) == 16
    assert C.executing_memory_class([_tensor_on(1)]) == 32


def test_an_evicted_seed_is_served_certified_on_the_proven_card(two_class_rig):
    """Seed time: the replay cache seeded KEY; the directory certifies it for 16 GB;
    the card is unknown → evicted, not applied. Launch on the 16 GB card → the
    certified config is applied and the seed is no longer counted local."""
    import triton
    t = _Tuner()
    local = triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2)
    t.cache[KEY] = local
    C.note_local(1)
    assert C.override_seeded([(KERNEL, t)]) == 0, "nothing applied at seed time: the card is unknown"
    assert KEY not in t.cache, "the seed is evicted so the launch site decides with the card in hand"
    mcls = C.executing_memory_class([_tensor_on(0)])
    assert C.apply(KERNEL, t, KEY, memory_class=mcls) is True
    C.served_evicted(KERNEL, KEY)
    assert t.cache[KEY].kwargs["BLOCK_N"] == 128
    assert C.served() == {"certified": 1, "swept": 0, "local": 0}


def test_an_evicted_seed_is_put_back_on_the_card_the_proof_does_not_cover(two_class_rig):
    """Launch on the 32 GB card: the 16 GB proof does not cover it, the machine's
    own earlier sweep (the seed) is put back — not a sweep again, not served
    certified. Seen RED with `reseed_evicted` made a no-op: the key stayed
    absent and the tuner would have re-swept a shape it had already swept."""
    import triton
    t = _Tuner()
    local = triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2)
    t.cache[KEY] = local
    C.note_local(1)
    C.override_seeded([(KERNEL, t)])
    mcls = C.executing_memory_class([_tensor_on(1)])
    assert C.apply(KERNEL, t, KEY, memory_class=mcls) is False, "proven on 16 GB, this card is 32 GB"
    assert C.reseed_evicted(KERNEL, t, KEY) is True
    assert t.cache[KEY] is local, "the machine's own config, not a new sweep"
    assert C.served() == {"certified": 0, "swept": 0, "local": 1}
    assert C.reseed_evicted(KERNEL, t, KEY) is False, "put back once"
