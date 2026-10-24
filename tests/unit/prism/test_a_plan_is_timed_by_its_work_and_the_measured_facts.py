"""A candidate plan's estimated time (core/prism/time_estimate.py): each component's uses at its device's roofline,
plus the weights each strategy moves over the host link, from MEASURED facts — a missing fact refused by name.

What these tests would do if the code were wrong: a roofline taking the sum instead of the max fails the first case;
zero3 moving weights once instead of every use ranks it with single_gpu and fails the second; a missing fact
answered by a default passes where the third expects a refusal.
"""
import pytest

from neurobrix.core.prism import time_estimate as T
from neurobrix.core.prism.work import Work

GPU = T.DeviceFacts(matmul_flops={"float16": 100e12}, memory_bps=900e9, h2d_bps=12e9)
HOST = T.HostFacts(matmul_flops=1e12, storage_bps=2e9)


def test_one_use_is_the_roofline_of_its_flops_and_bytes():
    w = Work(flops=int(100e12), weight_bytes=int(9e9), activation_bytes=0)       # 1 s of compute, 0.01 s of bytes
    assert T.compute_seconds(w, GPU, "float16") == pytest.approx(1.0)
    w = Work(flops=int(1e12), weight_bytes=int(450e9), activation_bytes=int(450e9))
    assert T.compute_seconds(w, GPU, "float16") == pytest.approx(1.0)              # memory-bound


def test_streaming_weights_every_use_costs_what_the_host_link_moves():
    works = {"transformer": Work(flops=int(1e12), weight_bytes=int(12e9), activation_bytes=0)}
    uses = {"transformer": 20}
    whole = T.plan_seconds("single_gpu", {"transformer": "cuda:0"}, works, uses, {"cuda:0": GPU}, HOST, "float16")
    zero3 = T.plan_seconds("zero3", {"transformer": "zero3:cuda:0"}, works, uses, {"cuda:0": GPU}, HOST, "float16")
    use = T.compute_seconds(works["transformer"], GPU, "float16")                  # 12 GB at 900 GB/s: memory-bound
    assert use == pytest.approx(12e9 / 900e9)
    assert whole == pytest.approx(20 * use + 1.0)                                  # 20 uses + one load
    assert zero3 == pytest.approx(20 * use + 20 * 1.0)                             # the weights cross every use
    host = T.plan_seconds("cpu_execution", {"transformer": "cpu"}, works, uses, {}, HOST, "float16")
    assert host == pytest.approx(20 * 1.0) and whole < zero3


def test_a_fact_the_profile_does_not_carry_is_refused():
    with pytest.raises(T.MissingFact, match="no `measured:`"):
        T.device_facts(None, "cuda:0")
    with pytest.raises(T.MissingFact, match="bfloat16"):
        T.compute_seconds(Work(1, 1, 1), GPU, "bfloat16")
    with pytest.raises(T.MissingFact, match="not measured"):
        T.plan_seconds("single_gpu", {"c": "cuda:1"}, {"c": Work(1, 1, 1)}, {}, {"cuda:0": GPU}, HOST, "float16")


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
