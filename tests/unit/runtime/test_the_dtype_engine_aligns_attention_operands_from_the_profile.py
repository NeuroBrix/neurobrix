"""Attention operands that disagree in dtype are aligned by ONE decision, the DtypeEngine's
`attention_operand_dtypes`, under the hardware profile's `precision.attention_operands` — the
Triton wrapper, the derived census and Prism's width estimate read it through
`launch_keys.sdpa_operand_dtypes`, the compiled and sequential engines through the compiled
DtypeEngine's twin. No caller holds a rule of its own, and no profile goes without the key.

Injections, each seen RED: `launch_keys.sdpa_operand_dtypes` keeping its own narrowest rule
(ignoring the profile) -> the widest-profile case fails; the compiled twin taking max for
"narrowest" -> the twin-equality case fails; `precision.attention_operands` removed from
volta.yml -> the every-profile case fails."""
import glob
import itertools
import os

import pytest
import yaml

from neurobrix.core.dtype import engine as CE
from neurobrix.triton import dtype as TE

_VENDORS = os.path.join(os.path.dirname(CE.__file__), "..", "..", "config", "vendors")
_FLOATS = ("float16", "bfloat16", "float32", "float64")


def _profile(name):
    with open(os.path.join(_VENDORS, name)) as f:
        return yaml.safe_load(f)


def test_every_vendor_profile_declares_the_alignment():
    files = sorted(glob.glob(os.path.join(_VENDORS, "*", "*.yml")))
    assert len(files) >= 27
    for f in files:
        with open(f) as fh:
            assert TE.attention_operand_alignment(yaml.safe_load(fh)) in TE.ATTENTION_OPERAND_ALIGNMENTS, f


def test_an_undeclared_or_unknown_alignment_is_refused_by_name():
    for prof in ({"architecture": "x", "precision": {}}, {"architecture": "x"},
                 {"architecture": "x", "precision": {"attention_operands": "fastest"}}):
        for eng in (TE, CE):
            with pytest.raises(ValueError, match="precision.attention_operands"):
                eng.attention_operand_alignment(prof)


def test_the_two_engines_take_the_same_decision():
    for q, k, v in itertools.product(_FLOATS, repeat=3):
        for r in (None,) + _FLOATS:
            for a in TE.ATTENTION_OPERAND_ALIGNMENTS:
                assert TE.attention_operand_dtypes(q, k, v, a, r) == CE.attention_operand_dtypes(q, k, v, a, r)
    for eng in (TE, CE):
        assert eng.attention_operand_dtypes("float16", "float16", "float32", "narrowest") == ("float16",) * 3 + (None,)
        assert eng.attention_operand_dtypes("float16", "float16", "float32", "widest") == ("float32",) * 3 + (None,)
        assert eng.attention_operand_dtypes("bfloat16", "float16", "float32", "narrowest")[0] == "bfloat16"
        with pytest.raises(ValueError, match="int64"):
            eng.attention_operand_dtypes("int64", "float16", "float16", "narrowest")


def test_the_launch_keys_read_the_profile_in_force(monkeypatch):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.ops import _configs
    from neurobrix.kernels.nbx_tensor import NBXDtype
    F16, F32 = NBXDtype.float16, NBXDtype.float32
    volta = _profile("nvidia/volta.yml")
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: volta)
    assert LK.sdpa_operand_dtypes(F16, F16, F32) == (F16, F16, F16, None)
    wide = dict(volta, precision=dict(volta["precision"], attention_operands="widest"))
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: wide)
    assert LK.sdpa_operand_dtypes(F16, F16, F32) == (F32, F32, F32, None)
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: {"architecture": "none"})
    with pytest.raises(ValueError, match="precision.attention_operands"):
        LK.sdpa_operand_dtypes(F16, F16, F32)
    assert LK.sdpa_operand_dtypes(F16, F16, F16) == (F16, F16, F16, None)   # agreement asks nothing


def test_the_compiled_engine_aligns_from_its_hardware_profile():
    import torch
    q, k, v = (torch.zeros(1, 2, 3, 4, dtype=d) for d in (torch.float16, torch.float16, torch.float32))
    eng = CE.DtypeEngine(torch.float16, hardware=("nvidia", "volta"))
    assert {t.dtype for t in eng.align_attention_operands(q, k, v)} == {torch.float16}
    with pytest.raises(RuntimeError, match="no hardware profile"):
        CE.DtypeEngine(torch.float16).align_attention_operands(q, k, v)
    same = CE.DtypeEngine(torch.float16).align_attention_operands(q, k, q)
    assert same[0] is q and same[2] is q
