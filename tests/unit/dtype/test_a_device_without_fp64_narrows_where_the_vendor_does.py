"""A device without fp64 receives float64 / complex128 narrowed to float32 / complex64 by the
DtypeEngine, read from the vendor profile; a device with fp64 keeps them, byte for byte.

Measured 2026-10-08 on the M4 Pro (native, apple-on-mq20 75837c5c):
  Wan2.1-T2V-1.3B-Diffusers  `_load_constant_native` -> `tensor.to("mps")` of a float64 constant:
                             "Cannot convert a MPS Tensor to float64 dtype as the MPS framework
                             doesn't support float64" (`measure/Wan2.1-T2V-1.3B-Diffusers.native.log`)
  Flex.1-alpha               op `aten.arange::2`, traced `dtype=torch.float64`: the same refusal
                             (`measure/Flex.1-alpha.native.log`)
The vendor's diffusers runs both in float32 / complex64 on MPS (its RoPE picks
`torch.float32 if is_mps else torch.float64`). Every Apple profile declares
`precision.supports_fp64: false`; no Python read it. The fact is the vendor profile's, per
accelerator; the host's own is `config.DTYPE_SUPPORT["cpu"]`.

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_a_device_without_fp64_narrows_where_the_vendor_does.py
"""
from __future__ import annotations

import base64
import io

import pytest
import torch
import yaml

from neurobrix.core.config.loader import CONFIG_ROOT
from neurobrix.core.dtype.engine import DtypeEngine

PROFILES = sorted((CONFIG_ROOT / "vendors").glob("*/*.yml"))


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_vendor_profile_declares_whether_its_device_has_fp64(path):
    precision = (yaml.safe_load(path.read_text()) or {}).get("precision") or {}
    assert isinstance(precision.get("supports_fp64"), bool), (
        f"{path.parent.name}/{path.stem}: precision.supports_fp64 undeclared ({precision})")


@pytest.mark.parametrize("device,vendor,arch,expected", [
    ("mps:0", "apple", "apple_m4_pro", False),
    ("cpu", "apple", "apple_m4_pro", True),
    ("cuda:0", "nvidia", "volta", True),
])
def test_the_device_s_fp64_is_read_from_its_profile(device, vendor, arch, expected):
    from neurobrix.core.dtype.config import device_supports_fp64
    assert device_supports_fp64(device, vendor, arch) is expected


def _engine(has_fp64):
    return DtypeEngine(torch.float16, graph_dtype=torch.float16, device_has_fp64=has_fp64)


@pytest.mark.parametrize("has_fp64,real,cplx", [(False, torch.float32, torch.complex64),
                                                (True, torch.float64, torch.complex128)])
def test_a_wide_constant_is_narrowed_only_where_the_device_lacks_fp64(has_fp64, real, cplx):
    e = _engine(has_fp64)
    assert e.storage_dtype(torch.float64) == real
    assert e.storage_dtype(torch.complex128) == cplx
    assert e.storage_dtype(torch.float16) == torch.float16
    assert e.convert_constant(torch.arange(4, dtype=torch.float64)).dtype == real


@pytest.mark.parametrize("has_fp64,expected", [(False, torch.float32), (True, torch.float64)])
def test_a_traced_float64_creation_lands_in_the_device_s_widest(has_fp64, expected):
    fn = _engine(has_fp64).compile_op("aten::arange", torch.ops.aten.arange.start,
                                      {"output_dtypes": ["float64"]})
    assert fn(0, 8, dtype=torch.float64).dtype == expected


@pytest.mark.parametrize("has_fp64,expected", [(False, torch.float32), (True, torch.float64)])
def test_a_float64_to_copy_target_lands_in_the_device_s_widest(has_fp64, expected):
    fn = _engine(has_fp64).compile_op("aten::_to_copy", None, {"output_dtypes": ["float64"]})
    assert fn(torch.ones(3, dtype=torch.float32)).dtype == expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a device without fp64")
def test_a_float64_constant_loads_onto_mps():
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor.__new__(GraphExecutor)
    ex.device, ex._weights = "mps", {}
    ex._dtype_engine = DtypeEngine(torch.float32, graph_dtype=torch.float32, device_has_fp64=False)
    ex._placement_torch_dtype = lambda: torch.float32
    buf = io.BytesIO()
    torch.save(torch.linspace(0, 1, 5, dtype=torch.float64), buf)
    ex._load_constant_native(base64.b64encode(buf.getvalue()).decode(), "rope.freqs")
    t = ex._weights["rope.freqs"]
    assert t.device.type == "mps" and t.dtype == torch.float32, (t.device, t.dtype)
