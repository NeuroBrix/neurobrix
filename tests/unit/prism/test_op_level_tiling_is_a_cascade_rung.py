"""Op-level tiling is a rung of the cascade, not a decoration applied after it.

`runtime_op_tiling` has existed for a long time, and it was computed in `solve`
AFTER a strategy had been chosen, against the allocations that strategy produced.
So a component the cascade had already sent to the host was examined with
`device_caps.get("cpu") == 0` and skipped. Tiling could decorate a component that
was already on the accelerator; it could never be the REASON one stayed there.

Measured on both machines the same day, `real-esrgan-x8` at 1024x1024: **17237 MB
planned, 16.4 GB of it activations against 32 MB of weights**, and both printing
`tiling none planned`. Apple's cascade exhausted and `_fail_error` refused; CUDA's
did not refuse at all — `cpu_streaming` accepted and won by score. Neither machine
ever asked whether the work could be cut.

These cells pin the rung's PLACE and its RESTRAINT. What it does when it fires is
proven by a run, not here.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from neurobrix.core.prism.solver import PrismSolver
from neurobrix.core.strategies import STRATEGY_REGISTRY

_MB = 1024 * 1024


def _cascade_lists():
    """Every ("name", self._try_name) tuple, in the order the source lists them."""
    tree = ast.parse(inspect.getsource(PrismSolver))
    lists = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        names = []
        for elt in node.elts:
            if (isinstance(elt, ast.Tuple) and len(elt.elts) == 2
                    and isinstance(elt.elts[0], ast.Constant)
                    and isinstance(elt.elts[1], ast.Attribute)
                    and elt.elts[1].attr.startswith("_try_")):
                names.append(elt.elts[0].value)
        if names:
            lists.append(names)
    return lists


def test_it_is_in_every_cascade_and_above_the_rungs_that_give_up_more():
    """Three lists — no-GPU, single-GPU, multi-GPU — and it belongs in all of them.

    A rung present in two of three is the defect this repository keeps meeting from
    other directions: the machine it was forgotten on is the one that needed it.
    """
    lists = _cascade_lists()
    assert len(lists) >= 3, f"expected the three cascades, found {len(lists)}"
    for names in lists:
        assert "op_level_tiling" in names, names
        i = names.index("op_level_tiling")
        # Above the rungs that cut MORE than the overflowing ops.
        for lower in ("layer_streaming", "cpu_execution", "cpu_streaming"):
            if lower in names:
                assert i < names.index(lower), (
                    f"op_level_tiling must be tried before {lower}: {names}")
        # Below every rung that keeps the component whole on the accelerator.
        for higher in ("single_gpu", "lazy_sequential", "zero3"):
            if higher in names:
                assert i > names.index(higher), (
                    f"op_level_tiling must be tried after {higher}: {names}")


def test_the_score_places_it_between_zero3_and_layer_streaming():
    """The ordering IS the inertia — there is no gate, it simply loses.

    Read from the AST, not from lines: a line-based scan for `"zero3":` matches the
    PROSE around the table as readily as the entry, which is the failure this
    repository has now met often enough to stop writing.
    """
    tree = ast.parse(inspect.getsource(PrismSolver))
    scores = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if "cpu_streaming" not in keys or "zero3" not in keys:
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                    and isinstance(v.value, (int, float))):
                scores[k.value] = v.value
        break
    for name in ("zero3", "op_level_tiling", "layer_streaming", "cpu_streaming"):
        assert name in scores, f"{name} has no score: {sorted(scores)}"
    assert (scores["zero3"] > scores["op_level_tiling"]
            > scores["layer_streaming"] > scores["cpu_streaming"]), scores


def test_the_name_is_buildable():
    """A name Prism can emit and the registry cannot build is a crash at selection.

    `cpu_streaming` was exactly that on 2026-09-03. The placement here is
    single-GPU; what differs is that the overflowing ops are cut, and that rides in
    `runtime_op_tiling`.
    """
    assert "op_level_tiling" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["op_level_tiling"] is STRATEGY_REGISTRY["single_gpu"]


def _solver(oom_reserve_mb=512):
    s = PrismSolver.__new__(PrismSolver)
    s.oom_reserve_mb = oom_reserve_mb
    return s


class _Dev:
    def __init__(self, mb, name="cuda:0"):
        self.capacity_mb = mb
        self.device_string = name


class _Mem:
    def __init__(self, w_mb, a_mb):
        self.weight_bytes = int(w_mb * _MB)
        self.activation_bytes = int(a_mb * _MB)
        self.overhead_bytes = 0

    @property
    def weight_mb(self):
        return self.weight_bytes / _MB

    @property
    def activation_mb(self):
        return self.activation_bytes / _MB


def test_it_declines_when_nothing_overflows():
    """A rung that fires when it is not needed would take work off a card that
    was holding it perfectly well."""
    s = _solver()
    out = s._try_op_level_tiling([("m", _Mem(100, 200))], {}, [_Dev(16384)], {}, None, None)
    assert out is None


def test_it_declines_when_the_weights_alone_exceed_the_card():
    """The coarse case: nothing to reshape because nothing fits."""
    s = _solver()
    out = s._try_op_level_tiling([("m", _Mem(20000, 500))], {}, [_Dev(16384)], {}, None, None)
    assert out is None


def test_it_declines_when_the_weights_FIT_but_are_not_what_overflows():
    """The case the coarse check does NOT catch, and the one the guard is for.

    Weights 8000 MB on a 15872 MB budget: they fit, so the total-weights check
    passes. Activations 9000 MB then overflow the 7872 MB left. But 9000 against
    8000 is not activation-DOMINANCE — this is a model whose weights are most of
    its footprint, and cutting its ops would shave a third off a problem that is
    two thirds weights. It belongs on a sharding or offload rung.

    The first version of this cell used 20000 MB of weights, which the coarse
    check catches first, so the dominance guard was never reached and removing the
    guard left the cell GREEN. Measured: with `dominated_by_activations` deleted,
    seven of seven still passed.
    """
    s = _solver()
    s._neural_components = [object()]
    s._input_config = None
    s._target_dtype_str = "float16"
    s._detect_op_level_tiling_pairs = lambda *a, **k: {"m": object()}
    out = s._try_op_level_tiling([("m", _Mem(8000, 9000))], {}, [_Dev(16384)], {}, None, None)
    assert out is None, "a weights-heavy model was sent down the tiling rung"


def test_it_declines_when_the_detector_can_tile_nothing():
    """It does not invent a plan shape.

    The rung asks `_detect_op_level_tiling_pairs` — which knows which ops the
    runtime can actually intercept — rather than promising a reshaping of its own
    devising. A rung that claimed a tiling the runtime cannot execute would be
    worse than the refusal it replaced.
    """
    s = _solver()
    s._neural_components = []          # nothing for the detector to work on
    s._input_config = None
    s._target_dtype_str = "float16"
    out = s._try_op_level_tiling([("m", _Mem(32, 16000))], {}, [_Dev(16384)], {}, None, None)
    assert out is None


def test_the_apple_shape_reaches_the_detector_rather_than_being_screened_out():
    """32 MB of weights against 16.4 GB of activations must not be pre-refused.

    This is the shape both machines met and neither tiled. The rung must carry it
    as far as the detector — whether the detector can then tile it is the
    detector's answer, and the cell above covers the case where it cannot.
    """
    from neurobrix.core.prism import plan_advice as pa
    assert pa.dominated_by_activations(32 * _MB, int(16.4 * 1024 * _MB))

    s = _solver()
    seen = {}

    def _detector(container, components, allocations, profile, input_config, dtype):
        seen["allocations"] = allocations
        return {}

    s._detect_op_level_tiling_pairs = _detector
    s._neural_components = [object()]
    s._input_config = None
    s._target_dtype_str = "float16"
    s._try_op_level_tiling([("m", _Mem(32, 16400))], {}, [_Dev(16384)], {}, None, None)

    assert "allocations" in seen, "the rung screened the Apple shape out before asking"
    # And it asks about a GPU placement — the question the post-hoc call cannot ask,
    # because by then the component has already been sent to the host.
    assert seen["allocations"]["m"][0] == "cuda:0", seen["allocations"]
