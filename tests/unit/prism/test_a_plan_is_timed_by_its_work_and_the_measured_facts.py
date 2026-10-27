"""A candidate plan's estimated time (core/prism/time_estimate.py): each component's uses at its device's roofline,
plus the weights each strategy moves over the host link, from MEASURED facts — a missing fact refused by name.

What these tests would do if the code were wrong: a roofline taking the sum instead of the max fails the first case;
zero3 moving weights once instead of every use ranks it with single_gpu and fails the second; a missing fact
answered by a default passes where the third expects a refusal.
"""
import pytest

from neurobrix.core.prism import time_estimate as T
from neurobrix.core.prism.work import Work

#: the engine's loader measured at the copy rate here, so a load costs what the host link moves (12 GB/s)
LOADER = {"BF16": {"device": {"cold": 12.0, "warm": 12.0}, "host": {"cold": 12.0, "warm": 12.0}}}
#: every component these plans name, stored in bf16 (the key the loader rate is read for)
ST = {c: "bfloat16" for c in ("transformer", "lm", "model", "c")}
GPU = T.DeviceFacts(matmul_flops={"float16": 100e12}, memory_bps=900e9, h2d_bps=12e9,
                    load_rates={"compiled": LOADER, "triton": LOADER})
HOST = T.HostFacts(matmul_flops=1e12, memory_bps=100e9, storage_bps=2e9)


def test_one_use_is_the_roofline_of_its_flops_and_bytes():
    w = Work(flops=int(100e12), weight_bytes=int(9e9), activation_bytes=0)       # 1 s of compute, 0.01 s of bytes
    assert T.compute_seconds(w, GPU, "float16") == pytest.approx(1.0)
    w = Work(flops=int(1e12), weight_bytes=int(450e9), activation_bytes=int(450e9))
    assert T.compute_seconds(w, GPU, "float16") == pytest.approx(1.0)              # memory-bound


def test_streaming_weights_every_use_costs_what_the_host_link_moves():
    works = {"transformer": Work(flops=int(1e12), weight_bytes=int(12e9), activation_bytes=0)}
    uses = {"transformer": 20}
    whole = T.plan_seconds("single_gpu", {"transformer": "cuda:0"}, works, uses, {"cuda:0": GPU}, HOST, "float16", stored=ST)
    zero3 = T.plan_seconds("zero3", {"transformer": "zero3:cuda:0"}, works, uses, {"cuda:0": GPU}, HOST, "float16", stored=ST)
    use = T.compute_seconds(works["transformer"], GPU, "float16")                  # 12 GB at 900 GB/s: memory-bound
    assert use == pytest.approx(12e9 / 900e9)
    assert whole == pytest.approx(20 * use + 1.0)                                  # 20 uses + one load
    assert zero3 == pytest.approx(20 * use + 1.0 + 20 * 1.0)                       # land in host memory once, cross every use
    host = T.plan_seconds("cpu_execution", {"transformer": "cpu"}, works, uses, {}, HOST, "float16", stored=ST)
    assert host == pytest.approx(20 * 1.0) and whole < zero3                        # 1e12 flops at 1 TFLOP/s: compute-bound
    heavy = {"transformer": Work(flops=int(1e9), weight_bytes=int(200e9), activation_bytes=0)}
    assert T.plan_seconds("cpu_execution", {"transformer": "cpu"}, heavy, {"transformer": 1}, {}, HOST,
                          "float16", stored=ST) == pytest.approx(2.0)                          # 200 GB at 100 GB/s: memory-bound


def test_a_generating_component_loads_its_weights_once_and_runs_its_prompt_pass_once():
    token = {"lm": Work(flops=int(1e9), weight_bytes=int(12e9), activation_bytes=0)}       # one token: memory-bound
    first = {"lm": Work(flops=int(100e12), weight_bytes=int(12e9), activation_bytes=0)}    # the prompt: 1 s of compute
    t = T.plan_seconds("single_gpu", {"lm": "cuda:0"}, token, {"lm": 32}, {"cuda:0": GPU}, HOST, "float16", prompt=first, stored=ST)
    assert t == pytest.approx(32 * 12e9 / 900e9 + 1.0 + 1.0)                          # 32 tokens + prompt + ONE load
    z = T.plan_seconds("zero3", {"lm": "zero3:cuda:0"}, token, {"lm": 32}, {"cuda:0": GPU}, HOST, "float16", prompt=first, stored=ST)
    assert z == pytest.approx(32 * 12e9 / 900e9 + 1.0 + 1.0 + 33 * 1.0)               # zero3: one landing, a crossing per pass


def test_a_fact_the_profile_does_not_carry_is_refused():
    with pytest.raises(T.MissingFact, match="no `measured:`"):
        T.device_facts(None, "cuda:0")
    with pytest.raises(T.MissingFact, match="bfloat16"):
        T.compute_seconds(Work(1, 1, 1), GPU, "bfloat16")
    with pytest.raises(T.MissingFact, match="host_memory_gbps"):
        T.host_facts({"host_matmul_gflops": 700})
    with pytest.raises(T.MissingFact, match="not measured"):
        T.plan_seconds("single_gpu", {"c": "cuda:1"}, {"c": Work(1, 1, 1)}, {}, {"cuda:0": GPU}, HOST, "float16", stored=ST)


def test_a_request_runs_a_loop_per_step_a_language_model_per_token_and_the_rest_once():
    diffusion = {"type": "iterative_process", "pre_loop": ["text_encoder"], "loop": {"components": ["transformer"]},
                 "post_loop": ["vae"]}
    assert T.uses_per_request(diffusion, 20, None) == {"text_encoder": ("once", 1), "vae": ("once", 1),
                                                       "transformer": ("loop", 20)}
    llm = {"order": ["model.visual", "model.language_model", "lm_head"],
           "generation": {"lm_component": "model.language_model", "head_component": "lm_head"}}
    assert T.uses_per_request(llm, None, 8) == {"model.visual": ("once", 1), "model.language_model": ("token", 8),
                                                "lm_head": ("token", 8)}
    asr = {"order": ["model.encoder", "model.decoder"],
           "stages": [{"component": "model.encoder", "execution": "forward"},
                      {"component": "model.decoder", "execution": "autoregressive"}]}
    assert T.uses_per_request(asr, None, 5)["model.decoder"] == ("token", 5)
    with pytest.raises(T.MissingFact, match="step count"):
        T.uses_per_request(diffusion, None, None)


def test_a_load_is_priced_at_the_engine_s_own_measured_loader_rate():
    """A load is the engine's LOADER — read, decode, convert, land — at its measured rate (`weight_load_gbps`,
    tools/profile_facts.py), cold the first time and warm after; never a bare copy rate. The rule-13 proof (V100 rack,
    2026-10-27): Wan2.2-I2V-A14B loaded in 123-173 s, Qwen3-30B-A3B in 294-447 s, against estimates at the H2D rate.
    What this test would do if the code were wrong: a load at h2d_gbps gives 1 s for both engines — RED; a profile
    without the fact estimated anyway — RED."""
    facts = T.device_facts({"matmul_tflops": {"float16": 42.3}, "memory_gbps": 789, "h2d_gbps": 8.0,
                            "weight_load_gbps": {"triton": {"BF16": {"device": {"cold": 2.0, "warm": 4.0}}},
                                                 "compiled": {"BF16": {"device": {"cold": 1.0, "warm": 2.0}}}}}, "cuda:0")
    w = {"model": Work(flops=0.0, weight_bytes=8e9, activation_bytes=0.0)}
    args = ("single_gpu", {"model": "cuda:0"}, w, {"model": 1}, {"cuda:0": facts}, None, "float16")
    # the same compute either way; the load 8 GB at 2 GB/s (triton, cold) against 8 GB at 1 GB/s (compiled, cold)
    assert abs(T.plan_seconds(*args, engine="compiled", stored=ST) - T.plan_seconds(*args, engine="triton", stored=ST) - 4.0) < 1e-6
    # a component loaded in two phases: once cold, once warm
    two = T.plan_seconds("lazy_sequential", {"model": "cuda:0"}, w, {"model": 1}, {"cuda:0": facts}, None, "float16",
                         phases={"model": 2}, engine="triton", stored=ST)
    assert abs(two - T.plan_seconds(*args, engine="triton", stored=ST) - 8e9 / 4e9) < 1e-6
    bare = T.device_facts({"matmul_tflops": {"float16": 42.3}, "memory_gbps": 789, "h2d_gbps": 8.0}, "cuda:0")
    with pytest.raises(T.MissingFact, match="weight_load_gbps.triton.BF16.device.cold"):
        T.plan_seconds("single_gpu", {"model": "cuda:0"}, w, {"model": 1}, {"cuda:0": bare}, None, "float16",
                       engine="triton", stored=ST)


def test_a_reloaded_piece_pays_the_loader_at_every_use_after_the_first():
    """layer_streaming reloads a piece from the container at every use: the first at the loader's cold rate, the next
    ones at the measured reload rate when the profile states one (`weight_reload_gbps`), else the loader's warm rate."""
    facts = T.device_facts({"matmul_tflops": {"float16": 42.3}, "memory_gbps": 789, "h2d_gbps": 8.0,
                            "weight_load_gbps": {"triton": {"BF16": {"device": {"cold": 2.0, "warm": 4.0}}}}}, "cuda:0")
    w = {"lm": Work(flops=0.0, weight_bytes=8e9, activation_bytes=0.0)}
    t = T.plan_seconds("layer_streaming", {"lm": "cuda:0"}, w, {"lm": 10}, {"cuda:0": facts}, None, "float16",
                       engine="triton", stored=ST)
    use = 10 * 8e9 / 789e9                                                            # ten memory-bound uses
    assert abs(t - use - (8e9 / 2e9 + 9 * 8e9 / 4e9)) < 1e-6
    host = T.HostFacts(matmul_flops=1e12, memory_bps=100e9, reload_bps=1e9)
    t = T.plan_seconds("layer_streaming", {"lm": "cuda:0"}, w, {"lm": 10}, {"cuda:0": facts}, host, "float16",
                       engine="triton", stored=ST)
    assert abs(t - use - (8e9 / 2e9 + 9 * 8e9 / 1e9)) < 1e-6


def test_a_lazy_plan_s_host_held_component_lands_in_host_memory_then_streams():
    """A lazy plan states the component it streams from host memory as `zero3:<card>` (Wan2.2's ratchet, Qwen3-30B's
    expert store): it lands in page-locked host memory at the loader's HOST rate, then — a ratchet — its blocks cross
    the pinned link at every pass; an expert store crosses only its routed experts (priced per use). The rule-13
    tables (2026-10-27): priced as a device load, Qwen3-30B's lazy plan was estimated 124 s for 432-494 s measured.
    What this test would do if the code were wrong: the same seconds as a plain device load — RED."""
    facts = T.device_facts({"matmul_tflops": {"float16": 42.3}, "memory_gbps": 789, "h2d_gbps": 8.0,
                            "weight_load_gbps": {"triton": {"BF16": {"device": {"cold": 2.0, "warm": 2.0},
                                                                     "host": {"cold": 0.5, "warm": 0.5}}}}}, "cuda:0")
    w = {"model": Work(flops=0.0, weight_bytes=8e9, activation_bytes=0.0)}
    args = (w, {"model": 2}, {"cuda:0": facts}, None, "float16")
    use = 2 * 8e9 / 789e9
    plain = T.plan_seconds("lazy_sequential", {"model": ("cuda:0", {})}, *args, engine="triton", stored=ST)
    held = T.plan_seconds("lazy_sequential", {"model": ("zero3:cuda:0", {})}, *args, engine="triton", stored=ST)
    assert abs(plain - use - 8e9 / 2e9) < 1e-6
    assert abs(held - use - (8e9 / 0.5e9 + 2 * 8e9 / 8e9)) < 1e-6
    store = {"model": Work(flops=0.0, weight_bytes=8e9, activation_bytes=0.0, expert_bytes_per_use=int(1e9))}
    routed = T.plan_seconds("lazy_sequential", {"model": ("zero3:cuda:0", {})}, store, {"model": 2}, {"cuda:0": facts},
                            None, "float16", engine="triton", stored=ST)
    assert abs(routed - use - (8e9 / 0.5e9 + 2 * 1e9 / 8e9)) < 1e-6
