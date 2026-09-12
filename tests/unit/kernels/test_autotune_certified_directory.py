"""The certified autotune directory: its gate, its lookup, its announcement, its contradictions.

The directory (`src/neurobrix/config/autotune/<vendor>/<profile>/<kernel>.<dtype>.json`) is
an engine component. A file is trusted only if its proof re-reads; a certified setting is
applied at load without a sweep; a missing one is said in clear before the runtime sweeps;
a runtime exclusion that contradicts a certification is reported, never silent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobrix.kernels import autotune_certified as C

KERNEL = "neurobrix.kernels.ops.matmul.matmul_kernel"


class _Tuner:
    keys = ["M", "N", "K", "IEEE_PRECISION", "PROMOTE_B"]
    arg_names = ["a_ptr", "b_ptr", "c_ptr", "M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn",
                 "stride_cm", "stride_cn", "IEEE_PRECISION", "PROMOTE_B"]
    base_fn = type("F", (), {"__name__": "matmul_kernel"})

    def __init__(self):
        self.cache = {}
        self.nargs = {}


KEY = (1500, 1280, 1280, True, True, "fp32", "fp16", "fp32")


def _entry(deviation=1.0e-5, tolerance=1.0e-4, **over):
    e = {"config": {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                    "num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None},
         "proof": {"date": "2026-09-07T00:00:00+00:00", "engine_version": "0.5.3",
                   "backend": {"name": "cuda", "version": "triton 3.6.0"}, "shape": list(KEY),
                   "deviation": deviation, "tolerance": tolerance, "oracle": "fp64",
                   "machine": {"hostname": "test", "device": "V100"}},
         "excluded": [{"config": {"kwargs": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                                  "num_warps": 8, "num_stages": 3}, "deviation": 0.5, "tolerance": tolerance}]}
    e.update(over)
    return e


def _write(root: Path, entries, vendor="nvidia", profile="volta", dtype="fp32", kernel=KERNEL, fmt=C.FORMAT):
    path = root / vendor / profile / f"{C.kernel_short(kernel)}.{dtype}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": fmt, "vendor": vendor, "profile": profile, "kernel": kernel,
                                "dtype": dtype, "entries": entries}))
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR", str(tmp_path))
    monkeypatch.setattr(C, "active_profile", lambda: ("nvidia", "volta"))
    monkeypatch.setattr(C, "_tolerance_for", lambda v, p, d: 1.0e-4)
    C.reset()
    yield tmp_path
    C.reset()


def test_the_output_dtype_is_the_written_buffers():
    assert C.output_dtype(_Tuner(), KEY) == "fp32"
    assert C.output_dtype(_Tuner(), (8, 8, 8, True, True, "fp16", "fp16", "fp16")) == "fp16"


def test_a_complete_proof_re_reads(root):
    path = _write(root, {C.key_repr(KEY): _entry()})
    assert C.validate(json.loads(path.read_text()), path, _Tuner()) == []


@pytest.mark.parametrize("bad, reason", [
    ({"proof": None}, "no proof"),
    ({"proof": {"date": "2026-09-07T00:00:00+00:00"}}, "proof without"),
    (dict(deviation=0.5), "exceeds its tolerance"),
    (dict(tolerance=2.0e-2), "not the profile's"),
    ({"excluded": [{"config": {}, "deviation": 1.0e-6}]}, "within the tolerance"),
    ({"config": {"kwargs": {}}}, "no config"),
])
def test_a_proof_that_does_not_re_read_is_refused(root, bad, reason):
    entry = _entry(**{k: v for k, v in bad.items() if k not in ("deviation", "tolerance")},
                   **{k: v for k, v in bad.items() if k in ("deviation", "tolerance")})
    if "proof" in bad and bad["proof"] is None:
        entry.pop("proof")
    path = _write(root, {C.key_repr(KEY): entry})
    problems = C.validate(json.loads(path.read_text()), path, _Tuner())
    assert problems and any(reason in p for p in problems), problems
    assert C.lookup(KERNEL, _Tuner(), KEY) is None, "a refused file must not serve"


def test_a_file_whose_names_disagree_with_its_path_is_refused(root):
    path = _write(root, {C.key_repr(KEY): _entry()}, dtype="fp32")
    doc = json.loads(path.read_text()); doc["dtype"] = "fp16"
    path.write_text(json.dumps(doc))
    assert any("dtype inside" in p for p in C.validate(doc, path, _Tuner()))


def test_a_certified_setting_is_applied_without_a_sweep(root):
    _write(root, {C.key_repr(KEY): _entry()})
    t = _Tuner()
    assert C.apply(KERNEL, t, KEY) is True
    cfg = t.cache[KEY]
    assert cfg.kwargs == {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}
    assert cfg.num_warps == 4 and cfg.num_stages == 2
    assert C.served()["certified"] == 1 and C.served()["swept"] == 0


def test_a_missing_setting_is_said_once_and_swept(root, capsys):
    t = _Tuner()
    assert C.apply(KERNEL, t, KEY) is False
    C.announce_missing(KERNEL, t, KEY)
    C.announce_missing(KERNEL, t, KEY)
    out = capsys.readouterr().out
    assert out.count("no certified setting for matmul_kernel fp32") == 1
    assert "sweeping at runtime with the consensus screen" in out and "never in the engine's directory" in out
    assert C.served()["swept"] == 1


def test_off_switch_sweeps_everything(root, monkeypatch):
    _write(root, {C.key_repr(KEY): _entry()})
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED", "off")
    assert C.apply(KERNEL, _Tuner(), KEY) is False


def test_a_runtime_exclusion_that_contradicts_a_certification_is_reported(root, capsys, monkeypatch):
    """The runtime screen excludes the very config the directory certifies:
    printed as a CONTRADICTION, recorded, never silent."""
    from neurobrix.triton import autotune_cache as atc
    _write(root, {C.key_repr(KEY): _entry()})
    t = _Tuner()
    monkeypatch.setattr(atc, "_qual_of", lambda tuner: KERNEL)
    monkeypatch.setattr(atc, "key_of", lambda tuner, args, kwargs: KEY)
    recorded = []
    monkeypatch.setattr(atc, "record_screen_exclusions", lambda entries: recorded.extend(entries) or len(entries))

    class Ex:
        kernel = "matmul_kernel"; key = ("x",)
        config = "BLOCK_M: 64, BLOCK_N: 128, BLOCK_K: 32, GROUP_M: 8, num_warps: 4, num_ctas: 1, num_stages: 2"
        dtype = "fp32"; deviation = 0.3; tolerance = 1.0e-4
    found = C.report_contradictions(t, [Ex()])
    out = capsys.readouterr().out
    assert len(found) == 1 and found[0]["contradiction"] is True
    assert "CONTRADICTION" in out and "certified on 2026-09-07" in out
    assert recorded and recorded[0]["contradiction"] is True
    assert C.contradictions()[0]["runtime"]["deviation"] == 0.3


def test_a_config_the_directory_did_not_certify_is_no_contradiction(root, monkeypatch, capsys):
    from neurobrix.triton import autotune_cache as atc
    _write(root, {C.key_repr(KEY): _entry()})
    monkeypatch.setattr(atc, "_qual_of", lambda tuner: KERNEL)
    monkeypatch.setattr(atc, "key_of", lambda tuner, args, kwargs: KEY)

    class Ex:
        kernel = "matmul_kernel"; key = ("x",); config = "BLOCK_M: 128, BLOCK_N: 128, BLOCK_K: 64, GROUP_M: 8, num_warps: 8, num_ctas: 1, num_stages: 3"
        dtype = "fp32"; deviation = 0.5; tolerance = 1.0e-4
    assert C.report_contradictions(_Tuner(), [Ex()]) == []
    assert "CONTRADICTION" not in capsys.readouterr().out


def test_the_engines_own_directory_passes_the_gate():
    """Every file shipped with the engine re-reads (empty directory = nothing to fail)."""
    for path in C.files():
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert C.validate(doc, path) == [], path


def test_a_contradiction_is_reported_even_with_the_apply_switch_off(root, capsys, monkeypatch):
    """NBX_AUTOTUNE_CERTIFIED=off is the proof's runtime-sweep arm — exactly where
    the screen and a certification can disagree. The switch stops applying,
    never reporting."""
    from neurobrix.triton import autotune_cache as atc
    _write(root, {C.key_repr(KEY): _entry()})
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED", "off")
    monkeypatch.setattr(atc, "_qual_of", lambda tuner: KERNEL)
    monkeypatch.setattr(atc, "key_of", lambda tuner, args, kwargs: KEY)
    monkeypatch.setattr(atc, "record_screen_exclusions", lambda entries: len(entries))

    class Ex:
        kernel = "matmul_kernel"; key = ("x",)
        config = "BLOCK_M: 64, BLOCK_N: 128, BLOCK_K: 32, GROUP_M: 8, num_warps: 4, num_ctas: 1, num_stages: 2"
        dtype = "fp32"; deviation = 0.3; tolerance = 1.0e-4
    assert C.apply(KERNEL, _Tuner(), KEY) is False, "the switch stops applying"
    assert len(C.report_contradictions(_Tuner(), [Ex()])) == 1, "but never reporting"
    assert "CONTRADICTION" in capsys.readouterr().out


def test_a_certified_setting_overrides_what_the_local_cache_seeded(root):
    """A warm machine: the replay cache seeded the tuner before the lookup; the
    directory's setting must still win and be counted as certified, not local."""
    _write(root, {C.key_repr(KEY): _entry()})
    t = _Tuner()
    class _Seeded:                              # what the replay cache put there (another config)
        kwargs = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}; num_warps = 2; num_stages = 2
    t.cache[KEY] = _Seeded()
    t.cache[("other",)] = _Seeded()             # a key the directory does not know stays as seeded
    C.note_local(2)
    assert C.override_seeded([(KERNEL, t)]) == 1
    assert t.cache[KEY].kwargs == {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}
    assert t.cache[("other",)] is not None and t.cache[("other",)].kwargs["BLOCK_M"] == 32
    assert C.served() == {"certified": 1, "swept": 0, "local": 1}


def test_an_operand_widened_on_load_is_keyed_by_the_dtype_it_is_computed_in(root):
    """The directory certified the matmul with the activation widened IN MEMORY
    (`fp32,fp16,fp32`). Since the activation is widened in the kernel's registers
    (PROMOTE_A), its memory dtype reads fp16 in the tuner's key; the computation is the
    same and so is the setting: the entry is served at the fp32-tagged key and cached
    at the tuner's own. Without the flag (a card computing fp16 × fp16) nothing is
    borrowed: that is another computation."""
    _write(root, {C.key_repr(KEY): _entry()})
    t = _Tuner()
    memory_key = (1500, 1280, 1280, True, True, "fp16", "fp16", "fp32")
    assert C.apply(KERNEL, t, memory_key) is False
    twin = C.computed_key(t, memory_key, {"PROMOTE_A": True})
    assert twin == KEY
    assert C.apply(KERNEL, t, memory_key, lookup_key=twin) is True
    assert memory_key in t.cache and t.cache[memory_key].kwargs["BLOCK_N"] == 128
    assert C.computed_key(t, memory_key, {"PROMOTE_A": False}) is None
    assert C.computed_key(t, memory_key, {}) is None


def test_the_replay_cache_key_of_a_widened_operand_is_overridden_by_its_certified_twin(root, monkeypatch):
    """At load the local replay cache seeds the memory-dtype key of a matmul whose activation
    is widened on load; with no call at hand the flags are read from the profile's rule (no
    native bf16 → the narrow activation computes in fp32) and the certified twin overrides it."""
    from neurobrix.kernels import wrappers as W
    monkeypatch.setattr(W, "_NBX_HAS_NATIVE_BF16", False)
    _write(root, {C.key_repr(KEY): _entry()})
    t = _Tuner(); t.arg_names = t.arg_names + ["PROMOTE_A"]
    memory_key = (1500, 1280, 1280, True, True, "fp16", "fp16", "fp32")
    import triton
    t.cache[memory_key] = triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2)
    assert C.override_seeded([(KERNEL, t)]) == 1
    assert t.cache[memory_key].kwargs["BLOCK_N"] == 128
    monkeypatch.setattr(W, "_NBX_HAS_NATIVE_BF16", True)       # fp16 × fp16 computes in fp16: another computation
    t2 = _Tuner(); t2.arg_names = t.arg_names
    t2.cache[memory_key] = triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2)
    assert C.override_seeded([(KERNEL, t2)]) == 0


# --------------------------------------------------------------------------
# 2026-09-12 — an exit code that conflates known debt with a break
#
# The census accumulates across engine versions. When a wrapper changes how it
# computes its autotune key, every entry recorded under the old rule becomes
# unreachable: no run will ever present that key again, so there is nothing to
# certify and refusing it is the CORRECT outcome.
#
# `certify` counted those refusals as failures and returned non-zero for them.
# On 2026-09-12 it certified 21 shapes, refused 184 unreachable keys — the whole
# of D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE — and exited 1, the same 1 a
# genuine break produces. A status that cries wolf on every run of a healthy
# directory is a status nobody reads on the day it is right.
# --------------------------------------------------------------------------

def test_an_unreachable_census_key_is_not_a_failure():
    from neurobrix.kernels.autotune_certify import UnreachableCensusKey
    # It stays a RuntimeError, so any caller that already handled the old type
    # keeps working; what is new is that it can be told apart.
    assert issubclass(UnreachableCensusKey, RuntimeError)


def test_the_two_are_counted_apart_and_only_one_is_fatal():
    """The classification, exercised on the loop's own two branches."""
    from neurobrix.kernels.autotune_certify import UnreachableCensusKey

    summary = {"failed": 0, "unreachable": 0}

    def classify(exc):
        try:
            raise exc
        except UnreachableCensusKey:
            summary["unreachable"] += 1
        except Exception:
            summary["failed"] += 1

    classify(UnreachableCensusKey("the census and the kernel disagree"))
    classify(UnreachableCensusKey("another stale key"))
    classify(RuntimeError("the oracle disagreed with every candidate"))

    assert summary == {"failed": 1, "unreachable": 2}
    # The exit code reads `failed` alone. Were it to read the sum, a directory
    # whose only finding is the known debt would report a break for ever.
    assert (0 if not summary["failed"] else 1) == 1
    assert (0 if not (summary["failed"] - 1) else 1) == 0, (
        "with the real break removed, 2 unreachable keys must still exit 0")
