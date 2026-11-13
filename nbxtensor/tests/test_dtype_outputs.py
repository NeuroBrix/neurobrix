"""The output dtype of an op, read from its built record or decided by the half cast-back rule (nbxtensor.dtype.outputs):
one definition every engine imports, torch-free."""
import itertools

import pytest

from nbxtensor.dtype.outputs import amp_fp32_output_dtype, built_output_dtype_name, built_output_is_complex

_GRAPHS = ("float16", "bfloat16", "float32", "float64")


@pytest.mark.parametrize("record, expected", [
    ({"output_dtypes": ["complex64"]}, True), ({"output_dtypes": ["complex32"]}, True),
    ({"output_dtypes": ["torch.complex128"]}, True), ({"output_dtypes": ["NBXDtype.complex64"]}, True),
    ({"output_dtypes": ["float32", "complex64"]}, True), ({"output_dtypes": ["float32"]}, False),
    ({"output_dtypes": []}, False), ({"input_dtypes": ["complex64"]}, False),
    ({"attributes": {"output_dtypes": ["complex64"]}}, False), (None, False), ("x", False)])
def test_a_complex_output_is_read_from_the_built_output_only(record, expected):
    assert built_output_is_complex(record) is expected


def test_the_built_output_name_is_the_first_outputs_plain_name():
    assert built_output_dtype_name({"output_dtypes": ["torch.float16", "float32"]}) == "float16"
    assert built_output_dtype_name({"output_dtypes": []}) is None
    assert built_output_dtype_name({"attributes": {"output_dtypes": ["float32"]}}) is None
    assert built_output_dtype_name(None) is None


def test_under_bfloat16_the_output_follows_the_graph():
    for safe, narrowed in itertools.product((False, True), repeat=2):
        assert amp_fp32_output_dtype("bfloat16", "bfloat16", safe, narrowed) == "bfloat16"
        for g in ("float16", "float32", "float64"):
            assert amp_fp32_output_dtype("bfloat16", g, safe, narrowed) == "float32"
    for g in (None, "", "int8"):
        with pytest.raises(ValueError, match="graph dtype"):
            amp_fp32_output_dtype("bfloat16", g, False, False)


def test_under_float16_only_the_contract_narrows():
    for g in _GRAPHS:
        assert amp_fp32_output_dtype("float16", g, False, False) == "float32"
        assert amp_fp32_output_dtype("float16", g, False, True) == "float16"
    for g in ("float16", "bfloat16"):                          # a half graph: the range contract narrows every such op
        assert amp_fp32_output_dtype("float16", g, True, False) == "float16"
    for g in ("float16", "bfloat16"):                          # a built fp32 island: only `narrowed` brings it down
        assert amp_fp32_output_dtype("float16", g, True, False, "float32") == "float32"
        assert amp_fp32_output_dtype("float16", g, False, True, "float32") == "float16"


def test_a_float32_model_under_float16_keeps_float32_unless_the_record_narrows():
    """`safe` is a range fact; a float32 model's precision is not: Kokoro-82M's harmonic source (a running phase of
    thousands of radians) stored in float16 became noise on the Triton engine (2026-11-13). On a graph wider than half,
    only `narrowed` (the record's measured set) brings an fp32-internal output down — as under bfloat16."""
    for g in ("float32", "float64"):
        for built in (None, "float32"):
            assert amp_fp32_output_dtype("float16", g, True, False, built) == "float32"
            assert amp_fp32_output_dtype("float16", g, True, True, built) == "float16"


@pytest.mark.parametrize("compute", ["float32", "float64", "int8"])
def test_a_compute_dtype_that_is_not_half_is_refused(compute):
    with pytest.raises(ValueError, match="not a half"):
        amp_fp32_output_dtype(compute, "bfloat16", False, False)
