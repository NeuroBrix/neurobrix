"""`running_generator()` must name the backend the CERTIFIER stamped.

The generator gate refuses a certified setting whose proof was made under a
different compiler. That is right, and it depends entirely on the reader and the
writer agreeing about the compiler's name.

They did not. `running_generator()` read `nbx_tensor.BACKEND_NAME`, which **does
not exist** — `grep -rn BACKEND_NAME src/` finds exactly one hit, the read itself.
So it fell back to its `"cuda"` default, `proof_backend` omits the name when it is
cuda, and on Apple the gate compared the directory's `triton 3.8.0 mps` against a
running label of `triton 3.8.0`. Measured 2026-09-19: **all 945 certified Apple
entries were refused**, every shape swept at runtime, and the log said the
settings "were certified under triton 3.8.0 mps and this engine runs
triton 3.8.0".

The name the certifier writes is the TRITON TARGET's backend — `mps` under
triton-ext, `metal` under the archived fork, `cuda` on the rack — and NOT the
engine's own name for the backend, which is `metal` here either way.
`autotune_certify._current_backend()` documents that divergence at length; this
test pins the generator label to the target, which is the half the stamp records.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triton")

from neurobrix.kernels.autotune_certified import (  # noqa: E402
    proof_backend, running_generator,
)


def _target_or_skip():
    try:
        from neurobrix.kernels.launcher import target
        t = target()
    except Exception:
        pytest.skip("no Triton target the launcher can resolve")
    name = getattr(t, "backend", None)
    if not name:
        pytest.skip("the Triton target does not name a backend")
    return name


def test_the_running_label_names_the_triton_target_backend():
    name = _target_or_skip()
    gen = running_generator()
    assert gen is not None, "a resolvable Triton target must produce a label"
    if name == "cuda":
        assert gen.endswith("cuda") or " " not in gen.split("triton ")[-1].strip(), gen
    else:
        assert gen.endswith(f" {name}"), (
            f"the running label is {gen!r} but this Triton target is {name!r}. "
            f"A gate that refuses on this label will refuse every entry the "
            f"certifier stamped on this machine.")


def test_the_running_label_round_trips_through_proof_backend():
    """Reader and writer must agree, which is the whole contract."""
    name = _target_or_skip()
    import triton
    written = proof_backend({"backend": {"triton": str(triton.__version__),
                                         "name": name}})
    assert running_generator() == written, (
        f"running_generator() is {running_generator()!r} but a proof stamped on "
        f"this machine reads {written!r}")
