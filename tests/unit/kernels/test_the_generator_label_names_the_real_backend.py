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
        # The name is PRESENT in the label; an out-of-tree backend appends its
        # own source hash after it, so it is no longer terminal.
        assert f" {name}" in gen, (
            f"the running label is {gen!r} but this Triton target is {name!r}. "
            f"A gate that refuses on this label will refuse every entry the "
            f"certifier stamped on this machine.")


def test_the_running_label_round_trips_through_proof_backend():
    """Reader and writer must agree, which is the whole contract — and the
    version they agree on is the DISTRIBUTION's, which carries the pin. The
    identity is now three parts (version, name, and an out-of-tree backend's
    hash), so the round-trip is built from the door, not a hand-made {triton,
    name} that would omit the hash and disagree with the reader."""
    name = _target_or_skip()
    from neurobrix.kernels.autotune_certified import generator_identity
    import importlib.metadata as md
    try:
        ver = md.version("triton")
    except md.PackageNotFoundError:
        import triton
        ver = str(triton.__version__)
    written = proof_backend({"backend": generator_identity()})
    assert running_generator() == written, (
        f"running_generator() is {running_generator()!r} but a proof stamped on "
        f"this machine reads {written!r}")
    assert ver in written, (ver, written)   # the pin-bearing version is shared


def test_the_label_sees_a_pin_move_when_the_metadata_does():
    """5a495ee2 -> 4a15f415 left `__version__` at a bare 3.8.0 on BOTH pins,
    so a label built from it served 964 entries to a compiler that had moved
    114 commits — measured 2026-09-20, and the gate never knew. The
    distribution version (`3.8.0+git<pin>`) is the identity that moves."""
    _target_or_skip()
    import importlib.metadata as md
    import triton
    try:
        ver = md.version("triton")
    except md.PackageNotFoundError:
        pytest.skip("no distribution metadata for triton on this install")
    if ver == str(triton.__version__):
        pytest.skip("this install's metadata adds nothing over __version__ — "
                    "the label cannot be finer than its sources")
    gen = running_generator()
    assert ver in gen, (
        f"the distribution says {ver!r} but the label reads {gen!r}: a pin "
        f"move is invisible to the gate again")


def test_the_writer_stamps_the_same_identity_the_reader_expects():
    """One door: `certify._backend()` writes what `running_generator()` reads.
    Two hands and two spellings is how ec938641's gate refused all 945."""
    _target_or_skip()
    from neurobrix.kernels.autotune_certify import _backend
    written = proof_backend({"backend": _backend()})
    assert written == running_generator(), (
        f"the certifier stamps {written!r} but the gate expects "
        f"{running_generator()!r}: every fresh proof would be refused on the "
        f"machine that just made it")
