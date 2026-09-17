"""The engine targets Triton; WHICH Metal backend runs is a SELECTION.

The bledden triton-msl fork was archived on 2026-09-17: it stays in this
repository's history and in the atelier clones as reference, and it is no longer
selectable. One row is still a seam — adding a backend is adding a row — and the
refusals below matter more with one backend than with two, not less.

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

    # THREE doors, and the environment is the first of them. An operator
    # override (`NEUROBRIX_METAL_BACKEND`) sits ABOVE the profile by design, so
    # a test that declares a profile while that variable is set is not testing
    # what it thinks: measured 2026-09-17, this file passed run standalone and
    # failed inside the suite runner, which exports the override for every
    # chunk. Clear it, and the profile is what chooses.
    monkeypatch.delenv("NEUROBRIX_METAL_BACKEND", raising=False)

    import neurobrix.kernels.ops._configs as C
    # BOTH remaining doors: the seam reads its declaration through
    # `vendor_profile_for_arch` (see the cycle it had to break), and everything
    # else in the engine reads `active_vendor_profile`.
    monkeypatch.setattr(C, "active_vendor_profile", lambda: dict(box))
    monkeypatch.setattr(C, "vendor_profile_for_arch", lambda arch: dict(box))
    return declare


def _installed(monkeypatch, **present):
    monkeypatch.setattr(MB, "_installed", lambda name: bool(present.get(name)))


def test_the_only_backend_is_named_in_exactly_one_place():
    """The table is the seam. If a vendor name appears in the engine elsewhere,
    that is the branch this test exists to keep out. One row is still a seam:
    adding a backend is adding a row, which is the whole point of the table."""
    assert set(MB.METAL_BACKENDS) == {"triton_ext"}


def test_the_profile_chooses(monkeypatch, profile):
    profile({"metal_backend": "triton_ext"})
    _installed(monkeypatch, triton_ext=True)
    assert MB.selected_metal_backend() == "triton_ext"


def test_a_declared_backend_that_is_absent_is_refused_by_name(monkeypatch, profile):
    """The refusal that protects attribution: never silently run something else."""
    profile({"metal_backend": "triton_ext"})
    _installed(monkeypatch)
    with pytest.raises(RuntimeError) as e:
        MB.selected_metal_backend()
    assert "triton_ext" in str(e.value) and "NOT installed" in str(e.value)


def test_the_archived_fork_is_refused_by_name_not_silently_swapped(monkeypatch, profile):
    """The bledden triton-msl fork was archived on 2026-09-17: it stays in this
    repository's history and in the atelier clones as reference, and it is not
    selectable.

    A profile that still declares it — an old checkout, a machine someone has
    not updated — must be REFUSED BY NAME. Silently running triton-ext instead
    would attribute its numbers to an implementation that did not produce them,
    which is the exact failure this seam exists to prevent, and it would be
    worse here than for a typo because the name used to be valid."""
    profile({"metal_backend": "triton_msl"})
    _installed(monkeypatch, triton_ext=True)
    with pytest.raises(RuntimeError) as e:
        MB.selected_metal_backend()
    assert "triton_msl" in str(e.value)
    assert "triton_ext" not in str(e.value).split("declares")[0]


def test_an_unknown_backend_name_is_refused(monkeypatch, profile):
    profile({"metal_backend": "not_a_backend"})
    _installed(monkeypatch, triton_ext=True)
    with pytest.raises(RuntimeError) as e:
        MB.selected_metal_backend()
    assert "not_a_backend" in str(e.value)


def test_one_installed_and_no_declaration_is_unambiguous(monkeypatch, profile):
    profile({})
    _installed(monkeypatch, triton_ext=True)
    assert MB.selected_metal_backend() == "triton_ext"


def test_no_backend_at_all_is_refused(monkeypatch, profile):
    profile({})
    _installed(monkeypatch)
    with pytest.raises(RuntimeError):
        MB.selected_metal_backend()


def test_the_shared_refusal_module_names_no_vendor():
    """`autotune_refusals` is run by the Dell too; it must ask the seam."""
    import inspect

    from neurobrix.kernels import autotune_refusals as R
    src = inspect.getsource(R)
    assert "triton_msl" not in src and "triton_apple_backend" not in src


def test_no_metal_backend_means_no_refusal_type(monkeypatch):
    """Inert on CUDA: the Dell has no Metal backend, so the shared check answers
    False rather than raising."""
    monkeypatch.setattr(MB, "backend_refusal_types", lambda: ())
    assert MB.is_backend_refusal(ValueError("x")) is False


def test_every_declared_backend_names_a_driver_or_is_refused(monkeypatch, profile):
    """The one row carries a driver; if a future row does not, the launcher
    refuses rather than launching through an ABI that is not its own."""
    for name, row in MB.METAL_BACKENDS.items():
        assert row.get("nbx_driver"), f"{name} names no driver"

    profile({"metal_backend": "triton_ext"})
    _installed(monkeypatch, triton_ext=True)
    monkeypatch.setitem(MB.METAL_BACKENDS["triton_ext"], "nbx_driver", None)
    with pytest.raises(RuntimeError) as e:
        MB.nbx_driver_module()
    assert "triton_ext" in str(e.value)


def test_the_fallback_marker_table_still_has_a_row_for_the_backend():
    """Empty markers are a STATEMENT about triton-ext, not a missing row.

    Searched 2026-09-17 in the installed `triton_apple_backend`: its Python, its
    Objective-C and the strings of its compiled dylib carry no "fall back",
    "falling back" or "on cpu" message. It refuses rather than computing
    elsewhere. The archived fork did fall back, which is why the table exists —
    so the row must remain present and empty, not disappear, or a backend that
    CAN fall back would be added without anyone noticing the door is there."""
    assert "triton_ext" in MB._FALLBACK_MARKERS
    assert MB.backend_fallback_markers() == ()   # it asks the SELECTED backend


def test_the_declaration_is_READ_never_inferred(monkeypatch):
    """The bug this pins, measured 2026-09-17.

    `selected_metal_backend` used to read its declaration through
    `active_vendor_profile()`, which resolves the profile through
    `launcher.target()` — whose backend NAME comes from this very function.
    Profile needs target, target needs backend, backend needs profile. On a cold
    process the inner call came back empty, `declared` became None, and the
    function fell through to "the single installed backend": it INFERRED what
    the profile was there to tell it, and agreed with the declaration only by
    luck, while exactly one backend happened to be installed. The symptom that
    exposed it was a refusal misreporting its own reason — "the profile declares
    none" about a profile that declares one.

    NOTE ON THE INSTRUMENT. The `profile` fixture patches BOTH doors, so a test
    using it passes on the broken code too — it hands the declaration through the
    very call that could not reach it. This test therefore makes
    `active_vendor_profile` RAISE, which is what the cycle amounted to, and
    supplies the declaration only through the door the seam must use. Verified
    red on the pre-fix seam, green after.
    """
    import neurobrix.kernels.ops._configs as C

    def _cycle():
        raise RuntimeError("profile needs target, target needs backend")

    monkeypatch.setattr(C, "active_vendor_profile", _cycle)
    monkeypatch.setattr(C, "vendor_profile_for_arch",
                        lambda arch: {"metal_backend": "triton_ext"})
    _installed(monkeypatch)                 # declared, and NOT installed

    with pytest.raises(MB.BackendSelectionRefused) as e:
        MB.selected_metal_backend()
    said = str(e.value)
    assert "triton_ext" in said and "NOT installed" in said
    assert "declares none" not in said, (
        "the seam could not read the declaration and inferred instead: " + said)


def test_the_seam_does_not_reach_the_profile_through_the_target(monkeypatch):
    """Structural half of the same contract, so the cycle cannot be reintroduced
    by a refactor that happens to keep this file green."""
    import ast
    import inspect
    import textwrap

    # Read the AST, not the text. The first version of this test grepped the
    # source and matched the NAME INSIDE THE COMMENT that explains why the call
    # must not be there — a test failing on its own documentation.
    tree = ast.parse(textwrap.dedent(inspect.getsource(MB.selected_metal_backend)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.attr for a in ast.walk(tree) if isinstance(a, ast.Attribute)}
    for imported in ast.walk(tree):
        if isinstance(imported, ast.ImportFrom):
            names |= {a.name for a in imported.names}

    assert "active_vendor_profile" not in names, (
        "selected_metal_backend must not resolve the profile through the target: "
        "the target's backend name comes from selected_metal_backend")
    assert "vendor_profile_for_arch" in names


def test_the_selection_refusal_has_a_name_of_its_own():
    """It must be tellable from 'this machine has no Triton target' BY TYPE.

    `ops/_configs.arch_smem_budget` wraps its target resolution in a broad
    `except`, and returning None there is right for a machine with no target.
    For a DELIBERATE refusal it was catastrophic: the whole hardware profile
    resolved empty and was cached empty, so smem budget, screen bytes and the
    declared backend went silently unread. Distinguishing them by message text
    would make every rewording a silent behaviour change."""
    assert issubclass(MB.BackendSelectionRefused, RuntimeError)
    import inspect
    src = inspect.getsource(MB.selected_metal_backend)
    assert "raise RuntimeError(" not in src, (
        "a selection refusal raised as a bare RuntimeError cannot be told from "
        "any other RuntimeError by the callers that must not swallow it")
