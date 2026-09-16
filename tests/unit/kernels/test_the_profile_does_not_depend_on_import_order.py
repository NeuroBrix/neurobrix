"""The hardware profile must not depend on WHEN it is asked for.

Measured on M4 Pro, 2026-09-16: asking `arch_smem_budget()` before anything else
imported yaml made `_ACTIVE_PROFILE` cache EMPTY. Resolving the target imports
the Metal backend's compiler, which is re-entrant with Triton's backend
discovery and leaves partially initialised modules behind; the `import yaml`
that followed then died inside the libyaml C extension with

    AttributeError: partially initialized module 'yaml' has no attribute 'error'

An AttributeError is not an ImportError, so it escaped the guard and the profile
came back empty — indistinguishable from "this machine has no profile". The
Apple profile's `metal_backend`, `autotune_screen_max_bytes` and smem budget went
unread, and `selected_metal_backend()` INFERRED the backend from what happened to
be installed instead of reading the declaration. An inferred backend cannot name
the implementation that produced a measurement, which is the whole point of the
seam.

A subprocess is the instrument, because the defect IS the import order: inside an
already-warm interpreter yaml is long since imported and the bug cannot appear.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

PROBE = r"""
import sys
# nothing imports yaml first: this is the cold order the engine really takes
from neurobrix.kernels.ops._configs import arch_smem_budget, active_vendor_profile
arch_smem_budget()
p = active_vendor_profile()
print("KEYS", len(p))
print("VENDOR", p.get("_vendor", "<none>"))
"""


def _run():
    return subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                          text=True, env={"PATH": "/usr/bin:/bin",
                                          "PYTHONPATH": str(SRC),
                                          "TOOLCHAINS": "Metal",
                                          "HOME": str(Path.home())}, timeout=300)


def _has_profile():
    """Does this machine have a vendor profile at all? (Warm interpreter.)"""
    import yaml                                    # noqa: F401  (warm it first)
    from neurobrix.kernels.ops._configs import arch_smem_budget, active_vendor_profile
    arch_smem_budget()
    return bool(active_vendor_profile())


@pytest.mark.skipif(not _has_profile(),
                    reason="no vendor profile matches this machine at all")
def test_a_cold_interpreter_resolves_the_same_profile_as_a_warm_one():
    r = _run()
    assert r.returncode == 0, f"probe crashed:\n{r.stderr[-2000:]}"
    keys = int(next(l for l in r.stdout.splitlines() if l.startswith("KEYS")).split()[1])
    vendor = next(l for l in r.stdout.splitlines() if l.startswith("VENDOR")).split(maxsplit=1)[1]
    assert keys > 0, (
        "the profile resolved EMPTY when asked before anything imported yaml; "
        "a cached empty profile is indistinguishable from 'no profile' and makes "
        f"the backend selection an inference. stderr:\n{r.stderr[-2000:]}")
    assert vendor != "<none>"
