"""Fixtures shared by the kernels cells."""

# The shared rig door, so every measuring cell in this directory can ask for a
# free rig by naming `a_free_rig` — see `_rig.py` for why it is a fixture and not
# a module-level mark.
from ._rig import a_free_rig, rig_reason  # noqa: F401
