"""Precision by policy (nbxtensor.dtype.policy, DESIGN rule 12): an explicit immutable object, plain data."""
import dataclasses

import pytest

from nbxtensor.dtype.policy import CLASSES, Policy, default_classes, get_policy


def test_each_class_computes_where_it_says():
    p = get_policy("params=float32,compute=float16,output=float32")
    assert p.compute_dtype("softmax", ["float16"]) == "float32"          # float32 class, whatever the compute dtype
    assert p.compute_dtype("mm", ["float32", "float32"]) == "float16"    # lower class: the compute dtype
    assert p.compute_dtype("addcmul", ["float16", "float32", "float16"]) == "float32"   # promote: the widest
    assert p.compute_dtype("add", ["float16", "float16"]) == "float16"   # no class: its inputs, promoted
    assert p.compute_dtype("add", ["int32", "int64"]) == "int64"         # no floating input: the policy is silent
    assert p.compute_dtype("full", []) is None                           # a creation op takes the dtype it is asked


def test_a_half_io_policy_lets_the_float32_kernels_take_half_inputs():
    p = get_policy("compute=float16,params=float16,output=float16")
    assert p.compute_dtype("layer_norm", ["float16"]) == "float32"
    assert p.replace(float32_ops_half_io=True).compute_dtype("layer_norm", ["float16"]) == "float16"
    assert p.replace(float32_ops_half_io=True).compute_dtype("exp", ["float16"]) == "float32"   # not a half-io op


def test_a_fill_sentinel_is_clamped_to_the_target_dtype():
    p = Policy()
    assert p.fill_value("masked_fill", -3.3895313892515355e+38, "float16") == -65504.0
    assert p.fill_value("full", 1e30, "bfloat16") == 1e30
    assert p.fill_value("add", -3.4e38, "float16") == -3.4e38           # not a fill op: untouched
    assert p.fill_value("fill", 7, "float16") == 7                       # not a float scalar: untouched


def test_the_policy_is_plain_immutable_data():
    p = get_policy("float16")
    assert (p.param, p.compute, p.output) == ("float16", "float16", "float16")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.compute = "float32"
    d = p.as_dict()
    assert set(d["classes"]) == set(CLASSES) and "softmax" in d["classes"]["float32"]
    assert get_policy("p=bfloat16,c=float16,o=float32").param == "bfloat16"   # the dtype table's names


def test_a_policy_refuses_what_it_cannot_mean():
    with pytest.raises(ValueError, match="not a floating dtype"):
        get_policy("compute=int8,params=float32,output=float32")
    with pytest.raises(ValueError, match="names no field"):
        get_policy("accumulate=float32")
    with pytest.raises(KeyError, match="not a class"):
        Policy().ops("fp16")


def test_the_shipped_classes_are_disjoint_where_they_compete():
    c = default_classes()
    assert not (c["float32"] & c["lower"]) and not (c["float32"] & c["promote"]) and not (c["lower"] & c["promote"])
