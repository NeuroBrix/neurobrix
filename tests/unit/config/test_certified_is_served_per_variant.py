"""A certified entry is served only to the variant it was proven on.

The certified autotune directory is `config/autotune/<vendor>/<profile>/`,
and `<profile>` is the name of the profile FILE that matched. Cutting the
Apple profiles one file per reported variant therefore split the certified
directory the same way, with no new code: this machine resolves
`apple/apple_m4_pro/`, and an M1 resolves `apple/apple_m1/`.

That is the property worth pinning, because it is the one that would rot
quietly. If a variant file ever stopped being the exact match — a renamed
key, a `compute_capability` that no longer equals the reported target — the
engine would fall back to the family profile and start serving `apple_silicon`
entries to every Mac, which is precisely the averaging this cut removed. The
numbers would still be plausible.
"""
import pytest

from neurobrix.kernels import autotune_certified as AC
from neurobrix.kernels.ops import _configs as C


def _device_name():
    try:
        import Metal
    except Exception:
        return None
    dev = Metal.MTLCreateSystemDefaultDevice()
    return dev.name() if dev is not None else None


@pytest.mark.skipif(_device_name() is None, reason="not an Apple device")
def test_this_machine_certifies_under_its_own_variant():
    C.arch_smem_budget()                       # resolve the active profile
    ident = AC.active_profile()
    assert ident is not None, (
        "no active profile: the certified directory would have no name and "
        "nothing could be served at all")
    vendor, profile = ident
    variant = _device_name().lower().replace(" ", "_")     # apple_m4_pro
    assert vendor == "apple", f"vendor resolved to {vendor!r}"
    assert profile == variant, (
        f"this device reports {_device_name()!r}, so its certified directory "
        f"must be {vendor}/{variant}/; it resolved to {vendor}/{profile}/. "
        f"Serving another variant's entries is the averaging the per-variant "
        f"cut exists to remove, and the numbers would still look plausible.")


@pytest.mark.skipif(_device_name() is None, reason="not an Apple device")
def test_the_active_profile_is_the_exact_one_not_the_family():
    C.arch_smem_budget()
    assert C._ACTIVE_PROFILE.get("_exact") is True, (
        "the family profile is serving this machine. It will run — an unknown "
        "chip must — but a variant that HAS its own file should never reach "
        "the fallback, and the fallback announcement in the log is the only "
        "thing that would have told you.")
