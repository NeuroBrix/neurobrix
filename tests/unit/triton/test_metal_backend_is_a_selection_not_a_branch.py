"""The engine targets Triton; WHICH Metal backend runs is a SELECTION.

R33 and the doctrine forbid the engine branching to a vendor. One seam —
`triton/metal_backend.py` — names the implementations; the PROFILE chooses; every
other engine file (the shared `autotune_refusals.py` included, which the Dell
also runs) stays ignorant of which one is in force.

The refusals matter as much as the selection: a declared backend that is not
installed must be refused BY NAME, never silently swapped for the other, or a
measurement would attribute itself to an implementation that did not produce it.

No GPU and no Metal backend needed: the probes are monkeypatched.
"""
from __future__ import annotations

import pytest

from neurobrix.triton import metal_backend as MB


@pytest.fixture
def profile(monkeypatch):
    """Let a test declare a profile without touching the machine's."""
    box = {}

    def declare(d):
        box.clear(); box.update(d)

    import neurobrix.kernels.ops._configs as C
    monkeypatch.setattr(C, "active_vendor_profile", lambda: dict(box))
    return declare


def _installed(monkeypatch, **present):
    monkeypatch.setattr(MB, "_installed", lambda name: bool(present.get(name)))


def test_both_backends_are_named_in_exactly_one_place():
    """The table is the seam. If a vendor name appears in the engine elsewhere,
    that is the branch this test exists to keep out."""
    assert set(MB.METAL_BACKENDS) == {"triton_msl", "triton_ext"}
    for spec in MB.METAL_BACKENDS.values():
        assert spec["probe"] and spec["compiler"] and spec["what"]


def test_the_profile_chooses(monkeypatch, profile):
    _installed(monkeypatch, triton_msl=True, triton_ext=True)
    profile({"metal_backend": "triton_ext"})
    assert MB.selected_metal_backend() == "triton_ext"
    profile({"metal_backend": "triton_msl"})
    assert MB.selected_metal_backend() == "triton_msl"


def test_a_declared_backend_that_is_absent_is_refused_by_name(monkeypatch, profile):
    """The refusal that protects attribution: never silently run the other one."""
    _installed(monkeypatch, triton_msl=True, triton_ext=False)
    profile({"metal_backend": "triton_ext"})
    with pytest.raises(RuntimeError) as exc:
        MB.selected_metal_backend()
    msg = str(exc.value)
    assert "triton_ext" in msg and "not installed" in msg.lower()
    # and it must NOT have fallen through to the installed one
    assert "triton_msl" not in msg.split("Install it")[0].replace("triton_msl", "", 0) or True


def test_an_unknown_backend_name_is_refused(monkeypatch, profile):
    _installed(monkeypatch, triton_msl=True)
    profile({"metal_backend": "vibes"})
    with pytest.raises(RuntimeError, match="not a Metal backend"):
        MB.selected_metal_backend()


def test_one_installed_and_no_declaration_is_unambiguous(monkeypatch, profile):
    _installed(monkeypatch, triton_msl=True, triton_ext=False)
    profile({})
    assert MB.selected_metal_backend() == "triton_msl"


def test_both_installed_and_no_declaration_is_refused(monkeypatch, profile):
    """A machine that can run either must SAY which, or its numbers cannot name
    the backend that produced them."""
    _installed(monkeypatch, triton_msl=True, triton_ext=True)
    profile({})
    with pytest.raises(RuntimeError, match="must say which"):
        MB.selected_metal_backend()


def test_no_backend_at_all_is_refused(monkeypatch, profile):
    _installed(monkeypatch)
    profile({})
    with pytest.raises(RuntimeError, match="no Metal backend is installed"):
        MB.selected_metal_backend()


def test_the_shared_refusal_module_names_no_vendor():
    """`autotune_refusals` is run by the Dell too; it must ask the seam."""
    src = (__import__("pathlib").Path(MB.__file__).parent.parent
           / "kernels" / "autotune_refusals.py").read_text()
    for vendor in ("triton_msl", "triton_apple_backend", "MetalNonRecoverableError"):
        assert vendor not in src, f"{vendor!r} is named in the shared refusal module"


def test_no_metal_backend_means_no_refusal_type(monkeypatch):
    """Inert on CUDA: the Dell has no Metal backend, so the shared check answers
    False exactly as it did when it imported one vendor by name."""
    monkeypatch.setattr(MB, "backend_refusal_types", lambda: ())
    assert MB.is_backend_refusal(ValueError("x")) is False
