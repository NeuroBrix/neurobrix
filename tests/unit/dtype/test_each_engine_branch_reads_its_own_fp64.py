"""Each engine branch reads from its vendor profile whether ITS kernels compute and store float64 /
complex128 — the device's fp64 (`precision.supports_fp64`) AND that branch's kernels'
(`precision.kernels_carry_fp64.compiled` / `.triton`) — and the two engines, given the same answer,
hold every dtype at the same width.

Before (merge-queue-22, 4dd80924): the DtypeEngine read the device's fp64 from the profile
(6fb2d574) while the Triton branch narrowed float64 -> float32 and complex128 -> complex64 by a
literal, in five places — the constant loader (`GraphExecutor._load_constant_triton`),
`constant_load_dtype`, `TritonSequence._parse_dtype` and the sequential dispatcher's two dtype
attributes — and Prism's width pass copied the literal (`_triton_remap`). On a Volta card the
compiled branch kept a float64 the Triton branch narrowed, and no profile said why. The why is the
Triton kernels' own (fp32-max: the elementwise and index kernels read at fp32 stride), so it is
declared per branch in the profile, not written in the code.

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_each_engine_branch_reads_its_own_fp64.py
"""
from __future__ import annotations

import pytest
import torch
import yaml

from neurobrix.core.config.loader import CONFIG_ROOT
from neurobrix.kernels.nbx_tensor import NBXDtype

PROFILES = sorted((CONFIG_ROOT / "vendors").glob("*/*.yml"))
BRANCHES = ("compiled", "triton")


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_vendor_profile_declares_the_fp64_of_each_branch_s_kernels(path):
    precision = (yaml.safe_load(path.read_text()) or {}).get("precision") or {}
    carry = precision.get("kernels_carry_fp64")
    assert isinstance(carry, dict) and set(carry) == set(BRANCHES) and all(
        isinstance(carry[b], bool) for b in BRANCHES), (
        f"{path.parent.name}/{path.stem}: precision.kernels_carry_fp64 must give a bool for "
        f"each of {BRANCHES} (found {carry!r})")


def _readers():
    from neurobrix.core.dtype.config import device_supports_fp64
    from neurobrix.triton.dtype import triton_has_fp64
    return (lambda v, a: device_supports_fp64("cuda:0", v, a)), triton_has_fp64


def _patched(monkeypatch, precision):
    from neurobrix.core.config import loader
    monkeypatch.setattr(loader, "get_vendor_config", lambda v, a: {"precision": precision})


@pytest.mark.parametrize("device,compiled,triton", [(d, c, t) for d in (True, False)
                                                    for c in (True, False) for t in (True, False)])
def test_each_branch_reads_the_device_and_its_own_kernels(monkeypatch, device, compiled, triton):
    _patched(monkeypatch, {"supports_fp64": device,
                           "kernels_carry_fp64": {"compiled": compiled, "triton": triton}})
    core, tri = _readers()
    assert core("v", "a") is (device and compiled)
    assert tri("v", "a") is (device and triton)


@pytest.mark.parametrize("precision,refused", [
    ({"supports_fp64": True}, (0, 1)),
    ({"supports_fp64": True, "kernels_carry_fp64": {"compiled": True}}, (1,)),
    ({"supports_fp64": True, "kernels_carry_fp64": {"triton": False}}, (0,)),
    ({"kernels_carry_fp64": {"compiled": True, "triton": True}}, (0, 1)),
], ids=["no-branch-key", "no-triton", "no-compiled", "no-device"])
def test_a_branch_whose_declaration_is_missing_is_refused(monkeypatch, precision, refused):
    _patched(monkeypatch, precision)
    for i in refused:                                   # 0: the compiled reader, 1: the Triton one
        with pytest.raises(ValueError, match="ZERO FALLBACK"):
            _readers()[i]("v", "a")


_WIDTHS = ("float64", "complex128", "float32", "complex64", "float16", "bfloat16", "int64")


@pytest.mark.parametrize("has_fp64", [True, False])
def test_both_engines_hold_every_dtype_at_the_same_width(has_fp64):
    from neurobrix.core.dtype.engine import DtypeEngine
    from neurobrix.triton.dtype import TritonDtypeEngine
    core = DtypeEngine(torch.float16, graph_dtype=torch.float16)
    core.device_has_fp64 = has_fp64
    tri = TritonDtypeEngine(NBXDtype.float16, has_fp64=has_fp64)
    for name in _WIDTHS:
        assert str(core.storage_dtype(getattr(torch, name))).replace("torch.", "") == \
            tri.storage_dtype(getattr(NBXDtype, name)).name, name


@pytest.mark.parametrize("has_fp64", [True, False])
def test_every_triton_site_asks_its_engine(has_fp64):
    from neurobrix.core.prism.runtime_widths import conservative_contract, runtime_dtypes
    from neurobrix.triton.dtype import constant_load_dtype
    from neurobrix.triton.sequence import TritonSequence
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    wide = {"float64": "float64" if has_fp64 else "float32",
            "complex128": "complex128" if has_fp64 else "complex64"}
    seq = TritonSequence({"ops": {}, "tensors": {}, "execution_order": []}, 0, NBXDtype.float16,
                         has_fp64=has_fp64)
    disp = TritonSequentialDispatcher(0, NBXDtype.float16, has_fp64=has_fp64)
    for traced, held in wide.items():
        assert constant_load_dtype(traced, "float16", has_fp64) == held
        assert seq._parse_dtype(f"torch.{traced}").name == held
        assert disp.resolve_attr({"type": "dtype", "value": f"torch.{traced}"}).name == held
        assert disp.resolve_attr({"type": "unknown", "value": f"torch.{traced}"}).name == held
    dag = {"ops": {"a::0": {"op_type": "aten::_to_copy", "input_tensor_ids": ["x"],
                            "output_tensor_ids": ["y"], "output_dtypes": ["torch.float64"],
                            "attributes": {"kwargs": {"dtype": {"type": "dtype",
                                                                "value": "torch.float64"}}}}},
           "tensors": {"x": {"dtype": "float32", "shape": [4], "is_input": True},
                       "y": {"dtype": "float64", "shape": [4]}},
           "execution_order": ["a::0"]}
    for engine in ("triton", "triton_sequential"):
        got = runtime_dtypes(dag, "float16", engine, has_native_bf16=False, contract=conservative_contract("test: no contract"),
                             has_fp64=has_fp64)
        assert got["y"] == wide["float64"], engine
