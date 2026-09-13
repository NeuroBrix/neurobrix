"""Component tiling must survive on every GPU, not only on CUDA.

`PrismSolver.solve` decides component-level spatial tiling during placement
and then filters the decisions down to those whose component "actually landed
on a GPU" -- that is what the comment above the filter says. The filter used
to read `startswith("cuda")`, so on an Apple (`mps:0`), AMD (`hip:0`) or Intel
(`xpu:0`) device every tiling decision the solver had just made was silently
discarded, and the component was planned untiled at its full activation size.

The replacement is `not startswith("cpu")`. On NVIDIA hardware the two are
INDISTINGUISHABLE -- a component's device string there is either `cuda:N` or
`cpu`, and both predicates agree on both -- so this is inert for CUDA and only
restores the intended behaviour elsewhere. That equivalence is what the first
test pins, because "it doesn't change CUDA" is a claim that has to be checkable
rather than asserted.
"""

from __future__ import annotations

import pytest

# Every device string form a plan's ComponentAllocation.device can take.
CUDA_MACHINE_DEVICES = ["cuda:0", "cuda:1", "cuda", "cpu"]
OTHER_GPU_DEVICES = ["mps:0", "mps", "hip:0", "xpu:0"]


def _old(dev: str) -> bool:
    return str(dev).startswith("cuda")


def _new(dev: str) -> bool:
    return not str(dev).startswith("cpu")


@pytest.mark.parametrize("dev", CUDA_MACHINE_DEVICES)
def test_change_is_inert_on_every_device_string_a_cuda_machine_produces(dev):
    assert _old(dev) == _new(dev), (
        f"the vendorless filter changes behaviour for {dev!r} on a CUDA "
        f"machine: old={_old(dev)} new={_new(dev)}"
    )


@pytest.mark.parametrize("dev", OTHER_GPU_DEVICES)
def test_change_restores_tiling_on_the_gpus_that_were_excluded(dev):
    assert not _old(dev), f"{dev} was expected to be excluded by the old filter"
    assert _new(dev), f"{dev} is a GPU and must keep its tiling decision"


def test_cpu_placement_still_drops_its_tiling_decision():
    # The filter's purpose: a component that ended up off the GPU must not
    # carry a stale tiling flag from a rejected strategy attempt.
    assert not _new("cpu")


def test_the_solver_uses_the_vendorless_predicate():
    """The predicate above is only meaningful if the solver actually uses it."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "core" / "prism" / "solver.py"
    text = src.read_text()
    idx = text.find("plan.component_tiling = {")
    assert idx != -1, "the component_tiling filter moved; re-point this test"
    window = text[idx:idx + 400]
    assert 'startswith("cpu")' in window, (
        "the component_tiling filter no longer uses the vendorless predicate:\n"
        + window[:300]
    )
    assert 'startswith("cuda")' not in window, (
        "the component_tiling filter branches on cuda again — it drops every "
        "tiling decision on mps/hip/xpu:\n" + window[:300]
    )
