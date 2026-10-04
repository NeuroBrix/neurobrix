"""A random op inside a graph draws from the run's generator, on the ATen branch as on Triton.

A diffusers pipeline threads ONE seeded generator through a run: an image encoder's posterior
sample (`retrieve_latents(vae.encode(x), generator)`), then the initial noise, then every
stochastic scheduler step. The ATen branch drew the initial and scheduler noise from the run's
generator (`VariableResolver.sampling_generator`) but a graph's own `aten::randn_like` from
torch's GLOBAL generator, seeded with the same seed: the two streams re-emit one sequence, so a
traced posterior draw equalled the first frame of the initial noise, element for element. The
Triton branch never had the defect (`kernels/rng_stream` serves every draw). The executor now
arms `core/runtime/graph/run_generator` with the resolver's generator for the ATen modes, and
both ATen dispatch sites (compiled `CompiledOpResolver`, sequential `NativeATenDispatcher`)
draw from it.

What would this file do if the code were wrong?
- a graph draw still taken from the global stream -> the first test (its draw equals the global
  stream's, not the run's) and the third (the posterior draw re-emits the initial noise);
- a draw that does not ADVANCE the run's generator (a copy, a re-seeded clone) -> the second test
  (the next draw from the generator repeats the graph's);
- the sequential site left on the global stream -> its parametrized rows;
- the executor not arming it, or leaving it armed after the request -> the fourth test.
"""
import pytest
import torch

from neurobrix.core.runtime.graph import run_generator
from neurobrix.core.runtime.graph.compiled_ops import CompiledOpResolver
from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher

SEED = 7
POSTERIOR = (1, 4, 1, 4, 8)            # [B, C, F=1, H, W]: an encoded first frame
# numel a multiple of 16: torch's CPU normal fill draws in blocks of 16, so a prefix of a larger
# draw re-emits a smaller one exactly (measured on the CPU; CUDA not measured here).
INIT = (1, 3, 4, 4, 8)                 # [B, F, C, H, W]: the initial noise, frame 0 first


@pytest.fixture(autouse=True)
def _disarmed():
    run_generator.arm(None)
    yield
    run_generator.arm(None)


def _compiled_randn_like(ref):
    fn = CompiledOpResolver(torch.device("cpu"), torch.float32)._make_pinned_rand("randn_like")
    return fn(ref)


def _sequential_randn_like(ref):
    d = NativeATenDispatcher(device="cpu", compute_dtype=torch.float32)
    return d.dispatch("aten::randn_like", [ref], {"kwargs": {}})


SITES = {"compiled": _compiled_randn_like, "sequential": _sequential_randn_like}


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_graph_draw_comes_from_the_run_generator(site):
    gen = torch.Generator().manual_seed(SEED)
    run_generator.arm(lambda: gen)
    torch.manual_seed(SEED + 1)                                    # the global stream elsewhere
    got = SITES[site](torch.zeros(POSTERIOR))
    ref = torch.randn(POSTERIOR, generator=torch.Generator().manual_seed(SEED))
    assert torch.equal(got, ref)


@pytest.mark.parametrize("site", sorted(SITES))
def test_one_stream_in_consumption_order(site):
    """The graph's draw advances the run's generator: the initial noise drawn next from it is the
    vendor pipeline's draw 2, not a repeat of draw 1."""
    gen = torch.Generator().manual_seed(SEED)
    run_generator.arm(lambda: gen)
    posterior_eps = SITES[site](torch.zeros(POSTERIOR))
    init = torch.randn(INIT, generator=gen)                       # what the resolver draws next
    vendor = torch.Generator().manual_seed(SEED)
    assert torch.equal(posterior_eps, torch.randn(POSTERIOR, generator=vendor))
    assert torch.equal(init, torch.randn(INIT, generator=vendor))


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_posterior_draw_does_not_re_emit_the_initial_noise(site):
    """The defect, measured: on two streams from one seed the posterior's noise IS frame 0 of the
    initial noise; on the run's one stream it is not."""
    first_frame = lambda init: init[:, 0].reshape(-1)              # noqa: E731
    # the old wiring: graph draw on the global stream, initial noise on a generator, same seed
    torch.manual_seed(SEED)
    old_eps = SITES[site](torch.zeros(POSTERIOR)).reshape(-1)
    old_init = torch.randn(INIT, generator=torch.Generator().manual_seed(SEED))
    assert torch.equal(old_eps, first_frame(old_init))             # the class this file closes
    # the run's one generator (the global stream seeded alike, as the CLI does)
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    run_generator.arm(lambda: gen)
    eps = SITES[site](torch.zeros(POSTERIOR)).reshape(-1)
    init = torch.randn(INIT, generator=gen)
    assert not torch.equal(eps, first_frame(init))


def test_a_run_without_a_generator_keeps_the_global_stream():
    torch.manual_seed(SEED)
    got = _compiled_randn_like(torch.zeros(POSTERIOR))
    torch.manual_seed(SEED)
    assert torch.equal(got, torch.randn(POSTERIOR))


def test_an_unknown_op_is_refused_by_name():
    with pytest.raises(ValueError, match="normal"):
        run_generator.draw("normal", [2], torch.float32, "cpu")


# ─────────────────────────────── the executor arms it per request ───────────────────────────────

def test_the_executor_arms_the_resolver_generator_for_the_request_and_disarms_after():
    from neurobrix.core.runtime.executor import RuntimeExecutor
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver

    seen = {}

    class _Handler:
        def execute(self):
            seen["gen"] = run_generator.generator()
            seen["resolver_gen"] = rt.variable_resolver.sampling_generator()
            return {}

    rt = RuntimeExecutor.__new__(RuntimeExecutor)
    rt.mode = "compiled"
    rt.pkg = rt.plan = None
    rt.executors, rt.modules = {}, {}
    rt.strategy = object()
    rt._connections_index, rt._loop_id, rt._nbx_path_str = {}, None, ""
    rt._persistent_mode, rt._binned_request = False, None
    rt.setup = lambda: None
    rt._prepare_defaults = lambda inputs: {"seed": SEED}

    def _init_resolver(inputs, merged):
        rt.variable_resolver = VariableResolver({}, merged, {}, {}, {}, device="cpu", mode="compiled")
    rt._init_variable_resolver = _init_resolver
    rt._set_runtime_resolution_on_executors = lambda merged: None
    rt._init_helpers = lambda: None
    rt._detect_flow_type = lambda: "static_graph"
    rt._get_primary_device = lambda: "cpu"
    rt._create_flow_handler = lambda flow_type, ctx: _Handler()

    rt.execute({})
    assert seen["gen"] is not None and seen["gen"] is seen["resolver_gen"]
    assert run_generator.generator() is None                       # disarmed after the request
