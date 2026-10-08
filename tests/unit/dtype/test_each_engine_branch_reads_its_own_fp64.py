"""Each engine branch reads from its vendor profile whether ITS kernels compute float64 / complex128
(`precision.kernels_carry_fp64.<branch>`, a capability) and whether the branch HOLDS them at that
width (`precision.stores_fp64.<branch>`, a DtypeEngine policy), both with the device's
(`precision.supports_fp64`); the two engines, given the same storage answer, hold every dtype at the
same width; and nbx_tensor's backend capability is the profile's, declared by the engine.

Before (merge-queue-22, 4dd80924): the DtypeEngine read the device's fp64 from the profile
(6fb2d574) while the Triton branch narrowed float64 -> float32 and complex128 -> complex64 by a
literal, in five places — the constant loader (`GraphExecutor._load_constant_triton`),
`constant_load_dtype`, `TritonSequence._parse_dtype` and the sequential dispatcher's two dtype
attributes — and Prism's width pass copied the literal (`_triton_remap`). Beside them nbx_tensor
kept its own table (`_BACKEND_HAS_FP64 = {cuda: True, hip: True, metal: False}`, `.get(..., True)`)
for the integer-division widening and the cast door. Two facts, three owners: on a CUDA card the
kernels compute fp64 (integer division widens through it) while the branch narrows constants. Each
is now one profile key, read by one reader per branch (Dell review of 5b88648d, 2026-10-08 21:26).

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_each_engine_branch_reads_its_own_fp64.py
"""
from __future__ import annotations

import re
import types

import pytest
import torch
import yaml

from neurobrix.core.config.loader import CONFIG_ROOT
from neurobrix.kernels.nbx_tensor import NBXDtype

PROFILES = sorted((CONFIG_ROOT / "vendors").glob("*/*.yml"))
BRANCHES = ("compiled", "triton")
KEYS = ("kernels_carry_fp64", "stores_fp64")


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_vendor_profile_declares_each_branch_s_fp64_capability_and_storage(path):
    precision = (yaml.safe_load(path.read_text()) or {}).get("precision") or {}
    for key in KEYS:
        got = precision.get(key)
        assert isinstance(got, dict) and set(got) == set(BRANCHES) and all(
            isinstance(got[b], bool) for b in BRANCHES), (
            f"{path.parent.name}/{path.stem}: precision.{key} must give a bool for each of "
            f"{BRANCHES} (found {got!r})")


def _readers():
    from neurobrix.core.dtype.config import device_supports_fp64
    from neurobrix.triton.dtype import triton_has_fp64, triton_stores_fp64
    return ((lambda v, a: device_supports_fp64("cuda:0", v, a)), triton_stores_fp64,
            triton_has_fp64)


def _patched(monkeypatch, precision):
    from neurobrix.core.config import loader
    monkeypatch.setattr(loader, "get_vendor_config", lambda v, a: {"precision": precision})


_BOOLS = (True, False)


@pytest.mark.parametrize("device,cc,ct,sc,st", [(d, cc, ct, sc, st) for d in _BOOLS
                                                for cc in _BOOLS for ct in _BOOLS
                                                for sc in _BOOLS for st in _BOOLS])
def test_each_branch_reads_the_device_its_kernels_and_its_storage(monkeypatch, device, cc, ct, sc, st):
    _patched(monkeypatch, {"supports_fp64": device,
                           "kernels_carry_fp64": {"compiled": cc, "triton": ct},
                           "stores_fp64": {"compiled": sc, "triton": st}})
    core, tri_stores, tri_has = _readers()
    assert core("v", "a") is (device and cc and sc)
    assert tri_stores("v", "a") is (device and ct and st)
    assert tri_has("v", "a") is (device and ct)


_FULL = {"supports_fp64": True, "kernels_carry_fp64": {"compiled": True, "triton": True},
         "stores_fp64": {"compiled": True, "triton": False}}


def _without(key, branch=None):
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _FULL.items()}
    if branch is None:
        del p[key]
    else:
        del p[key][branch]
    return p


@pytest.mark.parametrize("precision,refused", [
    (_without("supports_fp64"), (0, 1, 2)),
    (_without("kernels_carry_fp64"), (0, 1, 2)),
    (_without("kernels_carry_fp64", "triton"), (1, 2)),
    (_without("kernels_carry_fp64", "compiled"), (0,)),
    (_without("stores_fp64"), (0, 1)),
    (_without("stores_fp64", "triton"), (1,)),
    (_without("stores_fp64", "compiled"), (0,)),
], ids=["no-device", "no-carry", "no-carry-triton", "no-carry-compiled", "no-stores",
        "no-stores-triton", "no-stores-compiled"])
def test_a_branch_whose_declaration_is_missing_is_refused(monkeypatch, precision, refused):
    _patched(monkeypatch, precision)
    for i in refused:                     # 0: compiled, 1: Triton storage, 2: Triton capability
        with pytest.raises(ValueError, match="ZERO FALLBACK"):
            _readers()[i]("v", "a")


@pytest.mark.parametrize("vendor,arch", [("nvidia", "volta"), ("nvidia", "ampere"),
                                         ("nvidia", "hopper"), ("amd", "cdna3")])
def test_on_the_rack_the_triton_kernels_compute_fp64_and_the_branch_narrows_it(vendor, arch):
    """Today's rack behaviour, kept: integer division widens through float64 (capability true)
    and constants are narrowed (storage false)."""
    _, tri_stores, tri_has = _readers()
    assert tri_has(vendor, arch) is True
    assert tri_stores(vendor, arch) is False


def test_on_apple_neither_branch_holds_fp64_and_the_triton_kernels_have_none():
    core, tri_stores, tri_has = _readers()
    assert (core("apple", "apple_silicon"), tri_stores("apple", "apple_silicon"), tri_has("apple", "apple_silicon")) \
        == (False, False, False)


def _profile(*devices):
    return types.SimpleNamespace(has_native_bf16=False, devices=[
        types.SimpleNamespace(brand=b, architecture=a) for b, a in devices])


def test_a_profile_whose_devices_disagree_is_refused_by_both_branches():
    from neurobrix.core.dtype.config import profile_device_supports_fp64
    from neurobrix.triton.dtype import profile_triton_has_fp64, profile_triton_stores_fp64
    mixed = _profile(("nvidia", "volta"), ("apple", "apple_silicon"))
    for reader in (profile_device_supports_fp64, profile_triton_has_fp64):
        with pytest.raises(ValueError, match="disagree"):
            reader(mixed)
    assert profile_triton_stores_fp64(mixed) is False       # they agree there: both narrow
    assert profile_device_supports_fp64(_profile(("nvidia", "volta"), ("nvidia", "ampere"))) is True


@pytest.mark.parametrize("devices,expected", [((("nvidia", "volta"),), True),
                                              ((("apple", "apple_silicon"),), False)])
def test_the_engine_declares_the_profile_s_capability_to_nbx_tensor(monkeypatch, devices, expected):
    from neurobrix.kernels import nbx_tensor as T
    from neurobrix.kernels import wrappers as W
    monkeypatch.setattr(T, "_BACKEND_HAS_FP64", None)
    monkeypatch.setattr(W, "_NBX_HW_PROFILE", None)
    monkeypatch.setattr(W, "_NBX_HAS_NATIVE_BF16", True)
    with pytest.raises(RuntimeError, match="never declared"):
        T.backend_has_fp64()
    W.set_hardware_profile(_profile(*devices))
    assert T.backend_has_fp64() is expected


def test_nbx_tensor_takes_its_fp64_capability_from_its_caller_only():
    import inspect
    from neurobrix.kernels import nbx_tensor as T
    with pytest.raises(TypeError):
        T.set_backend_has_fp64("yes")
    src = inspect.getsource(T)
    assert not re.search(r"_BACKEND_HAS_FP64\s*=\s*\{", src), "a per-backend fp64 table is back"
    assert not re.search(r"^\s*(from|import)\s+neurobrix\.(core|config|triton)\b", src, re.M), \
        "nbx_tensor imports the engine (library boundary)"


_WIDTHS = ("float64", "complex128", "float32", "complex64", "float16", "bfloat16", "int64")


@pytest.mark.parametrize("stores_fp64", [True, False])
def test_both_engines_hold_every_dtype_at_the_same_width(stores_fp64):
    from neurobrix.core.dtype.engine import DtypeEngine
    from neurobrix.triton.dtype import TritonDtypeEngine
    core = DtypeEngine(torch.float16, graph_dtype=torch.float16)
    core.device_has_fp64 = stores_fp64
    tri = TritonDtypeEngine(NBXDtype.float16, stores_fp64=stores_fp64)
    for name in _WIDTHS:
        assert str(core.storage_dtype(getattr(torch, name))).replace("torch.", "") == \
            tri.storage_dtype(getattr(NBXDtype, name)).name, name


@pytest.mark.parametrize("stores_fp64", [True, False])
def test_every_triton_site_asks_its_engine(stores_fp64):
    from neurobrix.core.prism.runtime_widths import conservative_contract, runtime_dtypes
    from neurobrix.triton.dtype import constant_load_dtype
    from neurobrix.triton.sequence import TritonSequence
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    wide = {"float64": "float64" if stores_fp64 else "float32",
            "complex128": "complex128" if stores_fp64 else "complex64"}
    seq = TritonSequence({"ops": {}, "tensors": {}, "execution_order": []}, 0, NBXDtype.float16,
                         stores_fp64=stores_fp64)
    disp = TritonSequentialDispatcher(0, NBXDtype.float16, stores_fp64=stores_fp64)
    for traced, held in wide.items():
        assert constant_load_dtype(traced, "float16", stores_fp64) == held
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
                             stores_fp64=stores_fp64)
        assert got["y"] == wide["float64"], engine
    got = runtime_dtypes(dag, "float16", "compiled", has_native_bf16=False,
                         contract=conservative_contract("test: no contract"), stores_fp64=stores_fp64)
    assert got["y"] == wide["float64"], "compiled"


def test_a_device_less_profile_answers_the_host_s_own_and_no_profile_is_refused():
    """`config/hardware/cpu-only-x86.yml` (`devices: []`) runs every component on the host: the
    compiled reader answers the host's own; the Triton readers keep their refusal (Triton has no
    host compute backend), and the solver routes that profile by name (next test). Red on c4fd00d6:
    the compiled reader refused it, and every component's plan with it (Dell re-review 22:13)."""
    from neurobrix.core.dtype.config import architecture_supports_dtype, profile_device_supports_fp64
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.triton.dtype import profile_triton_has_fp64, profile_triton_stores_fp64
    cpu_only = load_profile("cpu-only-x86")
    assert not cpu_only.devices
    assert profile_device_supports_fp64(cpu_only) is architecture_supports_dtype("cpu", "float64")
    for reader in (profile_triton_has_fp64, profile_triton_stores_fp64):
        with pytest.raises(ValueError, match="ZERO FALLBACK"):
            reader(cpu_only)
    with pytest.raises(ValueError, match="ZERO FALLBACK"):
        profile_device_supports_fp64(None)


_ONE_CAST = {"ops": {"a::0": {"op_type": "aten::_to_copy", "input_tensor_ids": ["x"],
                              "output_tensor_ids": ["y"], "output_dtypes": ["torch.float64"],
                              "attributes": {"kwargs": {"dtype": {"type": "dtype",
                                                                  "value": "torch.float64"}}}}},
             "tensors": {"x": {"dtype": "float32", "shape": [4], "is_input": True},
                         "y": {"dtype": "float64", "shape": [4]}},
             "execution_order": ["a::0"]}


class _Profiler:
    def build_symbol_map(self, input_config, placement_floor=False):
        return {}

    def _resolve_shape(self, tensor, symbol_map):
        return tuple(tensor["shape"])


@pytest.mark.parametrize("mode", ["compiled", "triton", "triton_sequential"])
def test_the_solver_prices_a_cpu_only_profile_in_every_mode(mode):
    """The solver's width pass (`PrismSolver._activation_widths`, run by `_compute_memory` before the
    CPU-only cascade) on the device-less profile: priced at the host's answer, never refused; and a
    missing profile is refused, not priced wide by a branch of its own."""
    from neurobrix.core.dtype.config import architecture_supports_dtype
    from neurobrix.core.prism.loader import load_profile
    from neurobrix.core.prism.solver import PrismSolver
    solver = PrismSolver()
    solver._mode = mode
    comp = types.SimpleNamespace(name="c", graph=_ONE_CAST)
    container = types.SimpleNamespace(cache_path=None)
    widths = solver._activation_widths(comp, container, _Profiler(), None, "float32",
                                       load_profile("cpu-only-x86"))
    assert widths["y"] == (8 if architecture_supports_dtype("cpu", "float64") else 4)
    with pytest.raises(ValueError, match="ZERO FALLBACK"):
        solver._activation_widths(comp, container, _Profiler(), None, "float32", None)


def test_a_compiled_serve_does_not_read_the_triton_surface():
    """`serving/engine.py` declares the Triton wrappers' profile (and with it the Triton branch's
    fp64 key, which refuses a device-less or mixed profile) only off the compiled mode, as
    `cli/commands/run.py` does."""
    import inspect
    from neurobrix.serving import engine
    src = inspect.getsource(engine)
    site = src.index("set_hardware_profile(hw_profile)")
    assert re.search(r'if self\.mode != "compiled":\s*\n\s*from neurobrix\.kernels\.wrappers import '
                     r'set_hardware_profile\s*\n\s*set_hardware_profile\(hw_profile\)', src), \
        "the serving engine declares the Triton surface in every mode"
    assert src.count("set_hardware_profile(") == 1 and site
