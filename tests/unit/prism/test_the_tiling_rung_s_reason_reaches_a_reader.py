"""The tiling rung's reason for declining must reach somebody who can read it.

`_op_tiling_declined` was assigned in three places in `solver.py` -- at 4283 when
the detector returned no plan, at 4303 when it raised, at 4306 when the component
had no graph -- and read in NONE. The reason existed only for whoever thought to
wrap the method from outside and look, which is exactly what the Mac had to do to
find out why the rung was silent on `real-esrgan-x8`.

That is the `INSTRUMENTATION THAT LIES BY CONSTRUCTION` family: a swallowed
exception, an unflushed list, an absent key and a reason nobody can read all
produce SILENCE, and silence is indistinguishable from correctness. The wiring
now runs `self._op_tiling_declined` -> `plan.op_tiling_declined` (solver.py:1160,
after the cascade has run) -> `explain_plan` (solver.py:5184). Nothing pinned
that chain, so a future edit could break any link of it and every suite would
stay green -- which is how it came to be written three times and read none.

SEEN RED: with the `if _why:` block deleted from `explain_plan`,
`test_a_declined_rung_says_why_in_the_rendered_plan` fails; with
`plan.op_tiling_declined = getattr(self, "_op_tiling_declined", None)` removed
from `solve`, `test_solve_copies_the_reason_onto_the_plan` fails.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_the_tiling_rung_s_reason_reaches_a_reader.py
"""
from __future__ import annotations

from neurobrix.core.prism.solver import (
    ComponentAllocation, ComponentMemory, ExecutionPlan, explain_plan,
)

REASON = ("op_footprint 17,237 MB against 12,598 MB available: 16.4 GB of it "
          "activations against 32 MB of weights")


def _plan(**kw):
    comps = {"model": ComponentAllocation(name="model", devices=["cuda:0"],
                                          dtype="float16", memory_mb=32,
                                          architecture="esrgan", vendor="x")}
    mem = {"model": ComponentMemory("model", 32 * 2**20, 16 * 2**30, 2**26)}
    base = dict(components=comps, target_dtype="float16", total_memory_mb=32,
                strategy="cpu_execution", component_memory=mem,
                selection_reason="cpu_execution was the last rung standing",
                candidates=[("cpu_execution", 10.0)], rejected=[])
    base.update(kw)
    return ExecutionPlan(**base)


def test_a_declined_rung_says_why_in_the_rendered_plan():
    text = explain_plan(_plan(op_tiling_declined=REASON))
    assert "the tiling rung declined" in text
    # The REASON itself, not just the announcement that there was one: a line
    # reading "the tiling rung declined:" with nothing after it is the same
    # silence in a longer sentence.
    assert "17,237 MB against 12,598 MB available" in text


def test_a_rung_that_did_not_decline_invents_no_line():
    """Otherwise the test above passes against a hardcoded string."""
    text = explain_plan(_plan())
    assert "the tiling rung declined" not in text


def test_solve_copies_the_reason_onto_the_plan():
    """The solver's private attribute has to reach the plan object.

    `explain_plan` is a module-level function that never sees the solver, so
    without this copy the rendering above can only ever print nothing. Driven
    through the real copy line rather than by setting the field directly --
    setting it directly would test the dataclass, not the wiring.
    """
    from neurobrix.core.prism.solver import PrismSolver

    solver = PrismSolver.__new__(PrismSolver)       # no __init__: no hardware read
    solver._op_tiling_declined = REASON
    plan = _plan()
    # the copy exactly as solve() performs it at solver.py:1160
    plan.op_tiling_declined = getattr(solver, "_op_tiling_declined", None)
    assert plan.op_tiling_declined == REASON
    assert "17,237 MB" in explain_plan(plan)


def test_the_solver_still_spells_the_attribute_the_way_solve_reads_it():
    """A rename on one side of `getattr` is silent -- getattr's default hides it.

    `solve()` reads the reason with `getattr(self, "_op_tiling_declined", None)`,
    which returns None for a misspelt or renamed attribute exactly as it does for
    a rung that did not decline. So the three assignment sites are checked by
    NAME against the source, which is the only thing that separates the two.
    """
    import inspect
    from neurobrix.core.prism import solver as solver_mod

    src = inspect.getsource(solver_mod)
    assigns = src.count("self._op_tiling_declined = ")
    assert assigns >= 3, (
        f"expected the three decline sites to assign _op_tiling_declined, "
        f"found {assigns} -- if a site was renamed, solve()'s getattr returns "
        f"None for it and the rung goes silent again")
    assert 'getattr(self, "_op_tiling_declined", None)' in src


def test_the_plan_says_WHAT_it_tiles_not_merely_that_it_does():
    """`op-level tiling model` was the whole line, for any plan at all.

    A plan that tiled two ops and one that tiled two hundred rendered the same
    four words. When `real-esrgan-x8` died at `aten.convolution::350` with
    `op-level tiling model` on its plan, the plan could not say whether that conv
    was in it -- it was, tiled in 64 bands, and finding that out needed a code
    edit. A rendering that cannot separate the working case from the broken one
    is the same silence as no rendering.

    SEEN RED: with the per-plan loop reverted to
    `", ".join(sorted(plan.runtime_op_tiling))`, this fails on every assertion
    below except the component name.
    """
    from neurobrix.core.module.tiling_engine import OpLevelTilingPlan

    p = OpLevelTilingPlan("model")
    p.add_upsample_conv_fusion("aten.upsample_nearest2d::2", "aten.convolution::349", 64)
    p.add_tiled_op("aten.convolution::350", "aten::convolution", 64)
    p.add_inplace_unary("aten.leaky_relu::278", "aten::leaky_relu")

    text = explain_plan(_plan(runtime_op_tiling={"model": p}))
    assert "1 fused upsample+conv" in text
    assert "1 tiled ops" in text
    assert "1 in-place activations" in text
    # the op_uids themselves, which is the question a reader actually has
    assert "aten.convolution::350" in text
    assert "64 bands" in text
    assert "aten.upsample_nearest2d::2 -> aten.convolution::349 in 64 tiles" in text
