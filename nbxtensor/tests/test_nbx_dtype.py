"""nbxtensor.dtype.nbx_dtype: the enum's tables come from the one dtype table, and are the tables the tensor carried
when they were written by hand (tensor.py before M2) — the derivation changes no width, name or type string."""
from nbxtensor.dtype import nbx_dtype as N

D = N.NBXDtype
HAND_SIZES = {D.float16: 2, D.bfloat16: 2, D.float32: 4, D.float64: 8, D.int8: 1, D.int16: 2, D.int32: 4,
              D.int64: 8, D.uint8: 1, D.bool_: 1, D.complex64: 8, D.complex128: 16}
HAND_NAMES = {'float16': D.float16, 'fp16': D.float16, 'half': D.float16, 'bfloat16': D.bfloat16, 'bf16': D.bfloat16,
              'float32': D.float32, 'fp32': D.float32, 'float': D.float32, 'float64': D.float64, 'fp64': D.float64,
              'double': D.float64, 'int8': D.int8, 'int16': D.int16, 'int32': D.int32, 'int': D.int32,
              'int64': D.int64, 'long': D.int64, 'uint8': D.uint8, 'byte': D.uint8, 'bool': D.bool_,
              'complex64': D.complex64, 'complex128': D.complex128}
HAND_NUMPY = {'float16': D.float16, 'float32': D.float32, 'float64': D.float64, 'int8': D.int8, 'int16': D.int16,
              'int32': D.int32, 'int64': D.int64, 'uint8': D.uint8, 'bool': D.bool_, 'complex64': D.complex64,
              'complex128': D.complex128}
HAND_TYPESTR = {D.float16: '<f2', D.bfloat16: '<V2', D.float32: '<f4', D.float64: '<f8', D.int8: '<i1',
                D.int16: '<i2', D.int32: '<i4', D.int64: '<i8', D.uint8: '<u1', D.bool_: '|b1', D.complex64: '<c8',
                D.complex128: '<c16'}


def test_the_enum_values_are_frozen():
    assert [m.value for m in D] == list(range(12))
    assert [m.name for m in D][:4] == ["float16", "bfloat16", "float32", "float64"]


def test_the_derived_tables_are_the_hand_written_ones():
    assert N._DTYPE_SIZES == HAND_SIZES
    assert all(N._DTYPE_FROM_STR[k] == v for k, v in HAND_NAMES.items())
    assert N._NUMPY_DTYPE_MAP == HAND_NUMPY
    assert N._DTYPE_TYPESTR == HAND_TYPESTR
    assert N._FLOATING_DTYPES == {D.float16, D.bfloat16, D.float32, D.float64}
    assert N._COMPLEX_DTYPES == {D.complex64, D.complex128}


def test_parse_reads_names_and_torch_spellings():
    assert N.parse_dtype("torch.bfloat16") is D.bfloat16
    assert N.parse_dtype("half") is D.float16 and N.parse_dtype(D.int8) is D.int8
