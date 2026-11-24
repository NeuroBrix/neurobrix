"""A census cell that binds a compile target leaves the process's target as it found it.

What this test would do if the fixture were gone: the first cell binds the V100 target through the engine's
own `_bind_target`, the second finds it still bound (a kernel test after it would compile for NVIDIA on any
host) and fails. The two cells are ordered: pytest runs a file's cells in definition order.
"""
from __future__ import annotations

from nbxtensor.backend import launcher
from neurobrix.kernels import census as _census

_BEFORE = []


def test_binding_a_v100_target_takes_effect_inside_the_cell():
    """The bind is real inside the cell: the launcher answers the declared V100 target (cc 70)."""
    _BEFORE.append(launcher._TARGET)
    _census._bind_target("c4140-4xv100-16GB-nvlink", None)
    assert launcher.target().backend == "cuda" and launcher.target().arch == 70


def test_the_next_cell_finds_the_target_as_the_host_had_it():
    """After the cell above, the process target is what it was before any census cell bound one."""
    assert launcher._TARGET is _BEFORE[0]
