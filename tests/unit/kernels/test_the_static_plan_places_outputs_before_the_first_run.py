"""S4b (the supervisor 2026-10-28 09:12): the Triton engine's memory plan BEFORE the first run, from the graph alone — no
recording run. Each op's outputs are known before it runs (element counts at the bound symbols, the op that last reads
them); an allocation an op makes whose bytes are one of them times a float width takes a slice of one workspace at the
lowest offset free of every slice still HELD — held until its pointer is freed (a view keeps its base alive), so the plan
places but never overlaps two live tensors; what has no candidate or no room goes to the pool. And under the plan the
device never grows while the pool parks freed blocks: they go back to the driver first. Measured on the M4 Pro, HAT-L x4
at the judged request: Triton peak 10 688 -> 9 194 MB against the native 10 113 MB, output bit-equal. No card."""
import pytest

from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA

MB = 1 << 20


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setattr(DA, "_plan_workspaces", {})
    monkeypatch.setattr(DA, "_ws_held", {})
    monkeypatch.setattr(DA, "_ws_retiring", set())
    base = 1 << 40
    cands = {0: [[100 * MB // 2, 2, False]], 1: [[50 * MB // 4, 3, False], [10 * MB, 1, False]],
             2: [[100 * MB // 2, 4, False]]}
    DA.plan_static_begin(0, base, 160 * MB, cands)
    yield base
    DA.plan_static_stop()


def _at(op):
    DA.plan_at_op(op)


def test_outputs_take_slices_that_never_overlap_while_held(plan):
    _at(0)
    a = DA._static_place(100 * MB)                  # op 0's output, bf16
    _at(1)
    b = DA._static_place(50 * MB)                   # op 1's first output, fp32
    assert a == plan and b == plan + 100 * MB
    _at(2)
    assert DA._static_place(100 * MB) is None       # a held + b held: no 100 MB gap in 160 MB -> the pool serves it
    DA.free_cuda(a)                                 # a slice is free once its pointer is freed, never before
    _at(3)
    DA._static_cands[3] = [[25 * MB, 4, False]]
    assert DA._static_place(100 * MB) == plan       # back in a's place
    assert DA._static_hits == 3 and DA._static_miss == 1


def test_an_allocation_no_output_of_the_op_explains_goes_to_the_pool(plan):
    _at(1)
    assert DA._static_place(7 * MB + 3) is None     # not an output's element count times a width
    assert DA._static_place(10 * MB) == plan        # the op's 10 M-element uint8/bool output
    assert DA._static_place(10 * MB) is None        # taken once: a second one is the pool's


def test_under_the_plan_the_device_does_not_grow_over_parked_blocks(monkeypatch, plan):
    flushed = []
    monkeypatch.setattr(DA, "_pool_enabled", True)
    monkeypatch.setattr(DA, "_maybe_init_pool", staticmethod(lambda: None))
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 0))
    monkeypatch.setattr(DA, "_pool_take", staticmethod(lambda dev, n: None))
    monkeypatch.setattr(DA, "_pool_flush", staticmethod(lambda dev=None: flushed.append(dev) or 0))

    class Refuse:
        def __getattr__(self, name):
            return lambda *a: 1                       # every driver call refused: the order of calls is the test
    import neurobrix.kernels.nbx_tensor as NT
    monkeypatch.setattr(NT, "_gpu_runtime", lambda: Refuse())
    _at(9)                                            # no output planned at op 9: the request reaches the driver
    with pytest.raises(Exception):
        DA.malloc_cuda(64 * MB, 0)
    assert flushed and flushed[0] == 0, flushed       # the parked blocks went back before the driver was asked


def test_a_workspace_given_back_with_a_slice_alive_waits_for_that_slice(monkeypatch, plan):
    """A run output that is a view of a placed intermediate outlives the run: its workspace retires, and goes with it.
    Freed with its sequence, mochi's second transformer piece read a pointer no allocation held (2026-10-28 09:47)."""
    _at(0)
    a = DA._static_place(100 * MB)
    assert DA.plan_retire(plan) is False             # a is alive: the workspace stays
    assert plan in DA._plan_workspaces
    monkeypatch.setattr(DA, "_pool_enabled", False)
    calls = []
    orig = DA.free_cuda

    def spy(ptr):
        calls.append(ptr)
        if ptr == plan and plan not in DA._plan_workspaces:
            return                                     # the workspace's own free: the driver call is not the test
        return orig(ptr)
    monkeypatch.setattr(DA, "free_cuda", staticmethod(spy))
    DA.free_cuda(a)                                    # its last slice: the workspace goes now
    assert plan in calls and plan not in DA._plan_workspaces, calls
