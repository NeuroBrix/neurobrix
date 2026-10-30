"""The output dtype of an op, read from its traced record or decided by the half cast-back rule (nbxtensor.dtype.outputs):
one definition every engine imports, torch-free."""
import itertools

import pytest

from nbxtensor.dtype.outputs import amp_fp32_output_dtype, traced_output_dtype_name, traced_output_is_complex

_GRAPHS = ("float16", "bfloat16", "float32", "float64")


@pytest.mark.parametrize("record, expected", [
    ({"output_dtypes": ["complex64"]}, True), ({"output_dtypes": ["complex32"]}, True),
    ({"output_dtypes": ["torch.complex128"]}, True), ({"output_dtypes": ["NBXDtype.complex64"]}, True),
    ({"output_dtypes": ["float32", "complex64"]}, True), ({"output_dtypes": ["float32"]}, False),
    ({"output_dtypes": []}, False), ({"input_dtypes": ["complex64"]}, False),
    ({"attributes": {"output_dtypes": ["complex64"]}}, False), (None, False), ("x", False)])
def test_a_complex_output_is_read_from_the_traced_output_only(record, expected):
    assert traced_output_is_complex(record) is expected


def test_the_traced_output_name_is_the_first_outputs_plain_name():
    assert traced_output_dtype_name({"output_dtypes": ["torch.float16", "float32"]}) == "float16"
    assert traced_output_dtype_name({"output_dtypes": []}) is None
    assert traced_output_dtype_name({"attributes": {"output_dtypes": ["float32"]}}) is None
    assert traced_output_dtype_name(None) is None


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
        assert amp_fp32_output_dtype("float16", g, True, False) == "float16"
        assert amp_fp32_output_dtype("float16", g, False, True) == "float16"
    for g in ("float16", "bfloat16"):                          # a traced fp32 island: only `narrowed` brings it down
        assert amp_fp32_output_dtype("float16", g, True, False, "float32") == "float32"
        assert amp_fp32_output_dtype("float16", g, False, True, "float32") == "float16"
    assert amp_fp32_output_dtype("float16", "float32", True, False, "float32") == "float16"


@pytest.mark.parametrize("compute", ["float32", "float64", "int8"])
def test_a_compute_dtype_that_is_not_half_is_refused(compute):
    with pytest.raises(ValueError, match="not a half"):
        amp_fp32_output_dtype(compute, "bfloat16", False, False)
