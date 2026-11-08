"""A run that meets a device out-of-memory at a component re-plans once and re-enters its flow there (Dell + Mac,
2026-11-08 18:51, release 0.6.3; core/runtime/replan.py, RuntimeExecutor._execute_component).

Why: mochi-1-preview native on the M4 Pro, first run on a fresh home: the VAE, after the streamed transformer, met an MPS
out-of-memory; the run recorded its growth (10 033 MB) and exited 1, and only a second run, planned with the record, was
green. Now the first run records, releases, re-solves and calls the VAE again with the transformer's latents kept.

No card: a three-component flow over fake executors and fake plans; the executor's own boundary, signature and refusals.
What each cell would do if the code were wrong: a run that dies at the first OOM, re-runs a finished component, loses its
output, loops on re-plans, or re-enters with the same plan (red); a non-OOM error or an unarmed run re-planned (red); an
untyped MPS message not recognised, or any RuntimeError taken for an OOM (red); the cache released while the failed
call's frames still hold its tensors (red)."""
import gc
import types
import weakref

import pytest

from neurobrix.core.runtime.executor import RuntimeExecutor


class _Held:
    """A tensor the failed call held in its frame."""


def _plan(strategy, vae_stream=None):
    alloc = lambda: types.SimpleNamespace(devices=["mps:0"], dtype="bfloat16", strategy=strategy)
    return types.SimpleNamespace(strategy=strategy, loading_mode="lazy",
                                 components={c: alloc() for c in ("text_encoder", "transformer", "vae")},
                                 component_stream={"vae": vae_stream} if vae_stream else {})


class _OOM(RuntimeError):
    pass


def _run(plans, fails, armed=True, error=None, kv=(), loaded=()):
    """The flow text_encoder -> transformer -> vae over a fake executor; `fails` = {component: calls that raise}; `kv`:
    components whose executor holds a flow's interceptors; `loaded`: components still loaded when the vae runs."""
    ex = RuntimeExecutor.__new__(RuntimeExecutor)
    ex.mode, ex.plan, ex._strategy_name, ex.strategy = "compiled", plans[0], plans[0].strategy, "S0"
    ex.executors = {c: types.SimpleNamespace(name=f"exec-{c}-0", _op_uid_interceptors={"sdpa::3": "kv"} if c in kv else {},
                                             _weights_loaded=c in loaded) for c in ("text_encoder", "transformer", "vae")}
    ex._stream_runtime = {"vae": "pieces borrowing exec-vae-0"}
    ex._completed, ex._replanned = set(), set()
    ex._flow_ctx = types.SimpleNamespace(plan=ex.plan, strategy=ex.strategy)
    ex._input_synthesizer = types.SimpleNamespace(plan=ex.plan)
    log, outputs, calls, refs = [], {}, {}, []
    solves = iter(plans[1:])

    def call(comp, phase, ts):
        calls[comp] = calls.get(comp, 0) + 1
        if calls[comp] <= fails.get(comp, 0):
            held = _Held()
            refs.append(weakref.ref(held))
            try:
                raise (error or _OOM)("MPS backend out of memory (MPS allocated: 11.97 GiB, other allocations: 4.71 GiB)")
            except Exception as inner:   # the sequences' wrapping, `from` the allocator's own
                raise RuntimeError(f"Failed at op aten.conv3d::7 (aten::conv3d): {inner}") from inner
        outputs[f"{comp}.output_0"] = f"{comp} under {ex.plan.strategy}"
        return outputs[f"{comp}.output_0"]

    def release(device):
        gc.collect()
        log.append(("release", [r() is None for r in refs]))

    def solve():
        log.append("solve")
        nxt = next(solves)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    ex._execute_component_call = call
    ex._unload_component_weights = lambda c: log.append(("unload", c))
    ex._ensure_weights_loaded = lambda c: log.append(("load", c))
    ex._build_executor = lambda c: log.append(("build", c)) or types.SimpleNamespace(name=f"exec-{c}-1")
    ex._init_strategy = lambda: setattr(ex, "strategy", "S1")
    ex._apply_plan_tiling = lambda components=None: log.append(("tiling", sorted(components)))
    ex._get_primary_device = lambda: "mps:0"
    ex.replan = None
    if armed:
        from neurobrix.core.runtime.replan import Replan, in_chain
        ex.replan = Replan(is_device_oom=lambda e: in_chain(e, lambda x: isinstance(x, _OOM)),
                           record_growth=lambda: log.append("record") or 10033.0, solve=solve)
    import neurobrix.core.runtime.executor as E
    kept, E._release_device_memory = E._release_device_memory, lambda mode, device: release(device)
    try:
        for comp in ("text_encoder", "transformer", "vae"):
            ex._execute_component(comp, "forward", None)
    finally:
        E._release_device_memory = kept
    return ex, log, outputs, calls


def test_the_failed_component_is_re_planned_and_called_again_with_the_finished_outputs_kept(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="neurobrix.core.runtime.executor")
    ex, log, outputs, calls = _run([_plan("lazy_sequential"), _plan("lazy_sequential", vae_stream={"chunks": 4})],
                                   {"vae": 1})
    assert calls == {"text_encoder": 1, "transformer": 1, "vae": 2}       # nothing finished runs again
    assert outputs["transformer.output_0"] == "transformer under lazy_sequential"   # the latents kept
    assert ex.plan.component_stream == {"vae": {"chunks": 4}}               # the new plan in force
    assert ex._flow_ctx.plan is ex.plan and ex._input_synthesizer.plan is ex.plan and ex._flow_ctx.strategy == "S1"
    assert ex.executors["vae"].name == "exec-vae-1" and ex.executors["transformer"].name == "exec-transformer-0"
    assert ex._stream_runtime == {}                    # the old vae's stream pieces are not reused
    # recorded first, every component unloaded and the cache released before Prism is asked, then the vae re-placed
    assert log[0] == "record" and log.index("solve") > log.index(("release", [True]))
    assert [e for e in log if isinstance(e, tuple) and e[0] in ("build", "tiling")] == [("build", "vae"), ("tiling", ["vae"])]
    # one WARNING record: shown on the terminal whatever the level (nbxtensor.terminal: WARNING always shown)
    said = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "re-planned after a device out-of-memory" in r.getMessage()]
    assert len(said) == 1 and "vae ran out of device memory" in said[0] and "10033 MB" in said[0]
    assert "re-enters at vae" in said[0] and "text_encoder, transformer kept" in said[0]


def test_the_failed_call_s_frames_are_gone_before_the_cache_is_released():
    _, log, _, _ = _run([_plan("lazy_sequential"), _plan("lazy_sequential", vae_stream={"chunks": 4})], {"vae": 1})
    assert ("release", [True]) in log and ("release", [False]) not in log


def test_the_same_plan_for_the_components_left_is_a_named_refusal():
    with pytest.raises(RuntimeError, match=r"same plan \(lazy_sequential\) for the components left \(vae\)") as e:
        _run([_plan("lazy_sequential"), _plan("lazy_sequential")], {"vae": 1})
    assert "10033 MB) is recorded" in str(e.value)


def test_prism_s_refusal_is_the_run_s_named_refusal():
    refusal = RuntimeError("This model cannot run on this machine: 31 000 MB required, 24 000 MB available")
    with pytest.raises(RuntimeError, match=r"Prism refuses:\nThis model cannot run on this machine"):
        _run([_plan("lazy_sequential"), refusal], {"vae": 1})


def test_a_component_is_re_planned_once_never_in_a_loop():
    with pytest.raises(RuntimeError, match=r"vae ran out of device memory again after its one re-plan"):
        _run([_plan("lazy_sequential"), _plan("lazy_sequential", vae_stream={"chunks": 4}),
              _plan("lazy_sequential", vae_stream={"chunks": 8})], {"vae": 2})


def test_an_error_that_is_no_device_oom_or_an_unarmed_run_fails_as_before():
    with pytest.raises(RuntimeError, match=r"^Failed at op aten.conv3d::7") as e:
        _run([_plan("lazy_sequential"), _plan("x")], {"vae": 1}, error=ValueError)
    assert isinstance(e.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match=r"^Failed at op aten.conv3d::7"):
        _run([_plan("lazy_sequential"), _plan("x")], {"vae": 1}, armed=False)


def test_each_engine_knows_its_device_oom_from_its_own_facts():
    """compiled: torch's typed OutOfMemoryError, or the message the vendor file declares (Apple: MPS raises it untyped);
    triton: the allocator's DeviceOOMError — anywhere in the chain the sequences raise."""
    import torch
    from nbxtensor.memory.allocator import DeviceOOMError
    from neurobrix.cli.commands.run import _arm_replan

    def armed(mode, brand, arch, dev):
        d = types.SimpleNamespace(brand=brand, architecture=arch, get_device_string=lambda: dev)
        ex = types.SimpleNamespace()
        _arm_replan(ex, mode, None, types.SimpleNamespace(devices=[d]), None, None, None, None)
        return ex.replan.is_device_oom

    def wrapped(inner):
        try:
            raise inner
        except Exception as e:
            try:
                raise RuntimeError(f"Failed at op aten.conv3d::7 (aten::conv3d): {e}") from e
            except RuntimeError as outer:
                return outer

    mps = armed("compiled", "apple", "apple_m4_pro", "mps:0")
    assert mps(wrapped(RuntimeError("MPS backend out of memory (MPS allocated: 11.97 GiB, other allocations: 4.71 GiB)")))
    assert not mps(wrapped(RuntimeError("shapes cannot be multiplied")))
    cuda = armed("compiled", "nvidia", "volta", "cuda:0")
    assert cuda(wrapped(torch.OutOfMemoryError("CUDA out of memory.")))
    assert not cuda(wrapped(RuntimeError("MPS backend out of memory")))      # a message another vendor declares
    tri = armed("triton", "nvidia", "volta", "cuda:0")
    assert tri(wrapped(DeviceOOMError("GPU malloc failed (error 2) for 8589934592 bytes")))
    assert not tri(wrapped(torch.OutOfMemoryError("CUDA out of memory.")))


def test_what_the_flow_had_loaded_is_loaded_again_under_the_new_plan():
    _, log, _, _ = _run([_plan("lazy_sequential"), _plan("lazy_sequential", vae_stream={"chunks": 4})], {"vae": 1},
                        loaded=("transformer",))
    assert ("load", "transformer") in log and log.index(("load", "transformer")) > log.index(("build", "vae"))


def test_a_component_holding_a_flow_s_state_is_never_run_again_or_rebuilt():
    """A decoder's KV cache lives in interceptors on its executor: a call that failed part way has written some layers,
    and a rebuilt executor has none — re-running either computes wrong tokens; the run refuses by name instead."""
    with pytest.raises(RuntimeError, match=r"vae holds state between its calls \(_op_uid_interceptors\)") as e:
        _run([_plan("lazy_sequential"), _plan("lazy_sequential", vae_stream={"chunks": 4})], {"vae": 1}, kv=("vae",))
    assert "10033 MB) is recorded" in str(e.value)
    moved = _plan("lazy_sequential", vae_stream={"chunks": 4})
    moved.components["transformer"].devices = ["cpu"]
    with pytest.raises(RuntimeError, match=r"Prism moves transformer, which holds state between its calls"):
        _run([_plan("lazy_sequential"), moved], {"vae": 1}, kv=("transformer",))


def test_the_triton_recovery_imports_no_torch():
    """R33: the Triton engine's recovery (its classifier, the executor's boundary) loads no torch."""
    import subprocess
    import sys
    code = ("import sys, types; sys.modules['torch'] = None\n"
            "from neurobrix.cli.commands.run import _arm_replan\n"
            "from neurobrix.core.runtime.executor import RuntimeExecutor\n"
            "from nbxtensor.memory.allocator import DeviceOOMError\n"
            "ex = types.SimpleNamespace()\n"
            "_arm_replan(ex, 'triton', None, types.SimpleNamespace(devices=[]), None, None, None, None)\n"
            "assert ex.replan.is_device_oom(DeviceOOMError('GPU malloc failed'))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
