"""Every dtype width lives in ONE table, `config/dtypes.yml`, read by two mirror readers.

`neurobrix/core/dtype/itemsize.py` serves the PyTorch branch (Prism, the core executors, the
DtypeEngine); `neurobrix/triton/itemsize.py` serves the Triton branch (NBXTensor, the wrappers, the
launch keys, the certifier, the Triton loaders). The engines share no compute code, so each keeps
its own copy. This file holds them to four things:

  (a) the two readers carry the same functions and give the same answer on every input;
  (b) every value of the private tables they replaced is reproduced (written out below as the
      tables stood on the base, 50869807; complex is at its true width everywhere);
  (c) a dtype the table does not carry is refused BY NAME, naming the file to extend;
  (d) the Triton copy imports neither torch nor numpy (R33).

What this test does if the code were wrong: one width changed in either copy answers differently
from its twin -> (a) red; one width changed in the table -> (b) red; a silent `.get(dtype, 4)`
re-added -> (c) red.
"""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from neurobrix.core.dtype import itemsize as CORE
from neurobrix.triton import itemsize as TRITON

_SRC = Path(__file__).resolve().parents[3] / "src"


def _public(mod):
    return {n for n, f in inspect.getmembers(mod, inspect.isfunction)
            if f.__module__ == mod.__name__ and not n.startswith("_")}


def _answer(f, *args):
    try:
        return ("ok", f(*args))
    except Exception as e:                      # the refusal itself is part of the answer
        return (type(e).__name__, str(e))


# ───────────────────────────── (a) the two readers are one ─────────────────────────────

_UNKNOWN = ["float4", "fp16", "bf16", "half", "", "torch.float4", "float8_e4m3fnuz", "int4"]


def _names():
    from neurobrix.kernels.nbx_tensor import NBXDtype
    names = list(CORE.known_dtypes()) + _UNKNOWN + ["int4-g128-asym"]
    return names + ["torch." + n for n in CORE.known_dtypes()] + list(NBXDtype)


def test_the_two_readers_carry_the_same_functions():
    assert _public(CORE) == _public(TRITON), _public(CORE) ^ _public(TRITON)


def test_the_two_readers_read_the_same_table():
    assert CORE.itemsize_table() == TRITON.itemsize_table()
    assert CORE.known_dtypes() == TRITON.known_dtypes()


@pytest.mark.parametrize("fn", ["itemsize", "canonical_name"])
def test_the_two_readers_answer_the_same_per_dtype(fn):
    for d in _names():
        assert _answer(getattr(CORE, fn), d) == _answer(getattr(TRITON, fn), d), (fn, d)


def test_the_two_readers_store_the_same():
    for d in _names():
        for flag in (False, True):
            assert _answer(CORE.storage_dtype, d, flag) == _answer(TRITON.storage_dtype, d, flag), (d, flag)


def test_the_two_readers_represent_the_same():
    for d in _names():
        for numel in (0, 1, 127, 128, 129, 4096, 10 ** 9 + 7):
            for split in (False, True):
                assert (_answer(CORE.representation_bytes, d, numel, split)
                        == _answer(TRITON.representation_bytes, d, numel, split)), (d, numel, split)


def test_the_two_readers_decode_the_same_codes():
    for code in ["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8",
                 "U8", "BOOL", "C64", "f32", ""]:
        assert _answer(CORE.from_safetensors, code) == _answer(TRITON.from_safetensors, code), code
    for key in ["fp16", "bf16", "fp32", "fp64", "uint8", "uint16", "uint32", "uint64", "int1",
                "int8", "int16", "int32", "int64", "bool", "fp8", "float16", ""]:
        assert _answer(CORE.from_autotune_key, key) == _answer(TRITON.from_autotune_key, key), key


def test_a_torch_dtype_is_read_by_its_spelling():
    torch = pytest.importorskip("torch")
    for d in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.bool,
              torch.complex64, torch.complex128, torch.float8_e4m3fn, torch.uint8):
        assert CORE.itemsize(d) == d.itemsize == TRITON.itemsize(d), d


# ───────────────────── (b) every old table's value is reproduced ─────────────────────
# The private tables as they stood on the base (50869807), written out — not imported.

OLD_TABLES = {
    "core.dtype.config.BYTES_MAP": {
        "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4,
        "int16": 2, "int8": 1, "uint8": 1, "bool": 1, "complex64": 8, "complex128": 16},
    "config/system.yml dtype_bytes": {
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int8": 1, "int16": 2,
        "int32": 4, "int64": 8},
    "prism.solver._DTYPE_WIDTH / prism.layer_partition._DTYPE_WIDTH": {
        "float64": 8, "float32": 4, "bfloat16": 2, "float16": 2, "int64": 8, "int32": 4,
        "int16": 2, "int8": 1, "uint8": 1, "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1},
    "kernels.nbx_tensor._DTYPE_SIZES": {          # keyed by NBXDtype member name
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int8": 1, "int16": 2,
        "int32": 4, "int64": 8, "uint8": 1, "bool_": 1, "complex64": 8, "complex128": 16},
    "graph_executor._load_constant_triton / tools.constant_load_differential._DTYPE_BYTES": {
        "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4,
        "int8": 1, "uint8": 1, "bool": 1, "complex64": 8, "complex128": 16},
    "core.dtype.engine / triton.dtype._ATTENTION_OPERAND_BYTES": {
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8},
}
OLD_SAFETENSORS = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
                   "I8": 1, "U8": 1, "BOOL": 1}                                 # solver._ST_DTYPE_BYTES
OLD_AUTOTUNE = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8, "uint8": 1, "uint16": 2, "uint32": 4,
                "uint64": 8, "int1": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
                "bool": 1}                                                      # autotune_certify._itemsize
# solver._graph_constant_bytes priced the Triton loader's narrowing (float64 -> 4, complex128 -> 8):
# now `storage_dtype(dtype, stores_fp64=False)` read through the table.
OLD_TRITON_CONSTANT = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 4, "int32": 4,
                       "int64": 8, "int8": 1, "uint8": 1, "bool": 1, "complex64": 8, "complex128": 8}


@pytest.mark.parametrize("table", sorted(OLD_TABLES))
@pytest.mark.parametrize("reader", [CORE, TRITON], ids=["core", "triton"])
def test_every_old_table_value_is_reproduced(reader, table):
    from neurobrix.kernels.nbx_tensor import NBXDtype
    for name, width in OLD_TABLES[table].items():
        key = NBXDtype[name] if table.startswith("kernels.nbx_tensor") else name
        assert reader.itemsize(key) == width, (table, name)


@pytest.mark.parametrize("reader", [CORE, TRITON], ids=["core", "triton"])
def test_the_old_code_maps_are_reproduced(reader):
    for code, width in OLD_SAFETENSORS.items():
        assert reader.itemsize(reader.from_safetensors(code)) == width, code
    for key, width in OLD_AUTOTUNE.items():
        assert reader.itemsize(reader.from_autotune_key(key)) == width, key


@pytest.mark.parametrize("reader", [CORE, TRITON], ids=["core", "triton"])
def test_the_old_narrowed_constant_widths_are_reproduced(reader):
    for name, width in OLD_TRITON_CONSTANT.items():
        assert reader.itemsize(reader.storage_dtype(name, False)) == width, name
    assert reader.storage_dtype("float64", False) == "float32"
    assert reader.storage_dtype("complex128", False) == "complex64"
    for name in reader.known_dtypes():
        assert reader.storage_dtype(name, True) == name, name


def test_the_live_readers_answer_from_the_table():
    """The re-pointed readers, read live: NBXTensor's sizes and the engines' attention widths."""
    from neurobrix.kernels.nbx_tensor import NBXDtype, dtype_size
    for m in NBXDtype:
        assert dtype_size(m) == OLD_TABLES["kernels.nbx_tensor._DTYPE_SIZES"][m.name], m


def test_the_packed_encoding_is_its_bits_plus_its_group_scales():
    # int4-g128-asym: 4 bits each, one float16 scale and one float16 minimum per 128.
    assert CORE.representation_bytes("int4-g128-asym", 128) == 64 + 4
    assert CORE.representation_bytes("int4-g128-asym", 129) == 65 + 8
    assert CORE.representation_bytes("bfloat16", 10, split=True) == 40


# ───────────────────── (c) an unknown dtype is refused by name ─────────────────────

@pytest.mark.parametrize("reader", [CORE, TRITON], ids=["core", "triton"])
@pytest.mark.parametrize("name", _UNKNOWN)
def test_an_unknown_dtype_is_refused_by_name(reader, name):
    for call in (lambda: reader.itemsize(name), lambda: reader.storage_dtype(name, False),
                 lambda: reader.representation_bytes(name, 8)):
        with pytest.raises(ValueError) as e:
            call()
        assert repr(name) in str(e.value) and "config/dtypes.yml" in str(e.value), str(e.value)


@pytest.mark.parametrize("reader", [CORE, TRITON], ids=["core", "triton"])
def test_an_unknown_code_or_key_is_refused_by_name(reader):
    with pytest.raises(ValueError, match="'C64'.*config/dtypes.yml"):
        reader.from_safetensors("C64")
    with pytest.raises(ValueError, match="'fp8'.*config/dtypes.yml"):
        reader.from_autotune_key("fp8")
    with pytest.raises(TypeError, match="config/dtypes.yml"):
        reader.itemsize(4)
    with pytest.raises(ValueError, match="packed encoding"):
        reader.itemsize("int4-g128-asym")


# ───────────────────── (d) the Triton copy is torch-free and numpy-free ─────────────────────

def test_the_triton_reader_imports_no_torch_no_numpy():
    probe = ("import sys; import neurobrix.triton.itemsize as T; T.itemsize('float16'); "
             "T.representation_bytes('int4-g128-asym', 256); "
             "print('torch' in sys.modules, 'numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                         env={"PYTHONPATH": str(_SRC), "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout
    tree = ast.parse(Path(TRITON.__file__).read_text())
    imported = {(n.module or "") if isinstance(n, ast.ImportFrom) else a.name
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    assert imported <= {"__future__", "functools", "enum", "typing",
                        "neurobrix.core.config.loader"}, imported
