"""Two prices nobody held (the Dell's reading, 2026-11-10 17:06, Qwen3-Coder-30B-A3B served on two V100-32GB: 3 562 MB
of "activations" for a head whose weight is 593 MB — more than one resident expert block):

1. the head's `aten::t` of its weight was priced as a weight-sized tensor on the compiled engine, kept "by measurement
   owed" for the dtype engine's fp32 copy — but the flow runs the TEXT generation head with AMP off, where nothing is
   copied: the view is free there as on the Triton engines (`FlowBindings.head_runs_without_amp`);
2. the daemon planned every served family at `batch_size=2`, `dtype="float16"`, literals from its first commit, so a
   served LLM's head was scaled x2 for a batch nobody sends. It plans by `run`'s rule (`request_batch_and_dtype`).
What this would do if the code were wrong: card memory reserved for nothing, fewer resident expert blocks, or a served
upscaler planned for two images of a dtype its container does not declare."""
import ast
from pathlib import Path

import pytest

from neurobrix.core.prism.flow_bindings import FlowBindings

ROOT = Path(__file__).resolve().parents[3]


def _bindings(tmp_path, gen):
    for name in ("language_model", "lm_head", "gen_head"):
        (tmp_path / "components" / name).mkdir(parents=True, exist_ok=True)
    return FlowBindings({"flow": {"generation": gen}, "components": {}}, tmp_path, "m")


def test_only_the_text_generation_head_runs_without_amp(tmp_path):
    fb = _bindings(tmp_path, {"type": "autoregressive_text", "lm_component": "language_model",
                              "head_component": "lm_head"})
    assert fb.head_runs_without_amp("lm_head") is True
    assert fb.head_runs_without_amp("language_model") is False
    image = _bindings(tmp_path, {"type": "autoregressive_image", "lm_component": "language_model",
                                 "head_component": "gen_head"})
    assert image.head_runs_without_amp("gen_head") is False            # the flow leaves its AMP on
    assert FlowBindings({"flow": {}}, tmp_path, "m").head_runs_without_amp("lm_head") is False


def test_the_flow_still_turns_the_text_head_s_amp_off_where_the_price_says_so():
    """The price reads a line of the flow: if the flow stops turning AMP off, this fails before a plan is wrong."""
    src = (ROOT / "src/neurobrix/core/flow/autoregressive.py").read_text()
    at = src.index('if gen_type == "autoregressive_text":')
    nxt = src.index("return TextStrategy(lm_head_executor, session)", at)
    assert "lm_head_executor._dtype_engine.amp_enabled = False" in src[at:nxt]


@pytest.mark.parametrize("defaults, family, manifest, want", [
    ({}, {}, {"dtype": "bfloat16"}, (1, None, "bfloat16")),                       # a served LLM: one prompt, no guidance
    ({"guidance_scale": 4.5, "dtype": "float16"}, {}, {}, (2, 2, "float16")),      # guidance in force doubles
    ({"guidance_scale": 1.0}, {}, {"dtype": "float32"}, (1, None, "float32")),     # guidance stated, not in force
    ({"batch_size": 1}, {"guidance_scale": 7.5}, {"dtype": "float16"}, (2, 2, "float16")),
])
def test_the_batch_and_the_dtype_of_a_plan_follow_the_container_not_a_literal(defaults, family, manifest, want):
    from neurobrix.core.prism.request_config import request_batch_and_dtype
    assert request_batch_and_dtype(None, defaults, family, manifest, 1) == want


def test_a_container_that_declares_no_dtype_is_refused_not_given_one():
    from neurobrix.core.prism.request_config import request_batch_and_dtype
    from neurobrix.core.runtime_values import MissingRuntimeValue
    with pytest.raises(MissingRuntimeValue):
        request_batch_and_dtype(None, {}, {}, {}, 1)


def test_a_serving_session_gets_the_same_three_as_plan_fields():
    from neurobrix.core.prism.request_config import served_plan_fields
    assert served_plan_fields(None, {}, {"dtype": "bfloat16"}) == {"batch_size": 1, "guidance_passes": None,
                                                                   "dtype": "bfloat16"}


def test_the_compiled_engine_reads_the_text_head_s_transpose_in_place_and_no_other(tmp_path):
    from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig
    fb = _bindings(tmp_path, {"type": "autoregressive_text", "lm_component": "language_model",
                              "head_component": "lm_head"})
    cfg = InputConfig(batch_size=1, dtype="float16", flow=fb)
    free = ActivationProfiler.weight_transposes_free
    assert free("compiled", cfg, "lm_head") and not free("compiled", cfg, "language_model")
    assert free("triton", cfg, "language_model") and not free("compiled", InputConfig(batch_size=1), "lm_head")


def test_the_daemon_states_no_batch_and_no_dtype_of_its_own():
    tree = ast.parse((ROOT / "src/neurobrix/serving/engine.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "InputConfig":
            literal = {k.arg for k in node.keywords if k.arg in ("batch_size", "dtype")
                       and isinstance(k.value, ast.Constant)}
            assert not literal, f"the daemon's InputConfig states {sorted(literal)} as literals"
