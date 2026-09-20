"""A finalizer must not raise on a tensor that never ran `__init__`.

`NBXTensor.__del__` read `self._owns_data` directly. With `__slots__` and no
`__init__`, that attribute does not exist and the finalizer raises
`AttributeError`. Python does not propagate an exception from `__del__` — it
prints it as an unraisable exception and ABANDONS THE REST OF THE METHOD. The
rest of the method is the `free_cuda` call, so on a tensor that did own device
memory the buffer would never be returned to the allocator.

`NBXTensor.__new__(NBXTensor)` is not a contrivance: it is what `copy`,
`pickle`, and `test_prefill_determinism`'s routing-contract test do, and it is
what any `__init__` that raises part-way leaves behind.
"""
import sys

import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor


def test_del_on_a_tensor_that_never_ran_init_does_not_raise():
    seen = []
    hook = getattr(sys, "unraisablehook", None)
    assert hook is not None
    sys.unraisablehook = lambda u: seen.append(u)
    try:
        t = NBXTensor.__new__(NBXTensor)
        del t
    finally:
        sys.unraisablehook = hook
    assert not seen, (
        "the finalizer raised on a half-built tensor: "
        + "; ".join(f"{u.exc_type.__name__}: {u.exc_value}" for u in seen))


def test_a_half_built_tensor_owns_nothing_so_nothing_is_leaked():
    """The reason returning early is correct and not a papered-over free.

    `__init__` is pure assignment — it allocates no device memory — so a tensor
    that did not finish it cannot own any. Ownership is recorded by `__init__`
    from the `owns_data` argument, never established inside the tensor itself.
    """
    t = NBXTensor.__new__(NBXTensor)
    for slot in ("_owns_data", "_data_ptr"):
        assert not hasattr(t, slot), f"{slot} is set before __init__ runs"
    del t
