"""The library's default precision policy (nbxtensor/dtype/policy.yml) carries the DtypeEngine's op classes exactly,
until the engine reads them from the library (one fact, one place: the copy below is the one being retired)."""
from neurobrix.core.dtype import engine as E
from nbxtensor.dtype.policy import default_classes


def test_the_shipped_op_classes_are_the_engines():
    c = default_classes()
    assert c["float32"] == E.AMP_FP32_OPS
    assert c["lower"] == E.AMP_FP16_OPS
    assert c["promote"] == E.AMP_PROMOTE_OPS
    assert c["float32_half_io"] == E._FP32_OPS_HALF_IO
    assert c["scalar_fill"] == E.AMP_SCALAR_FILL_OPS
    assert c["creation_fill"] == E.AMP_CREATION_FILL_OPS
