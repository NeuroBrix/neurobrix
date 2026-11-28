"""Unit tests for list-batched NextDiT denoiser I/O, flow-matching step conventions,
VAE shift_factor, FlowEuler linspace_inclusive sigmas, and diffusion chat-template tokenization.

All tests run 100 % on CPU without GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from neurobrix.core.components.base import ComponentConfig
from neurobrix.core.components.handlers.vae_handler import VAEComponentHandler
from neurobrix.core.flow.base import FlowContext
from neurobrix.core.flow.iterative_process import IterativeProcessHandler
from neurobrix.core.module.scheduler.flow.flow_euler import FlowEulerScheduler
from neurobrix.core.module.text.processor import TextProcessor
from neurobrix.core.runtime.resolution.input_resolver import InputResolver
from neurobrix.core.runtime.resolution.list_batched_denoiser import (
    apply_model_timestep_convention,
    apply_velocity_sign,
    _repeat_timestep,
    build_list_batched_passes,
    flow_step_convention,
    list_batched_spec,
    resolve_state_variable_alias,
    trim_caption_embedding,
    unpack_list_batched_output,
)


def _make_nextdit_dag(slots: int = 2) -> dict:
    tensors = {}
    in_ids = []
    for i in range(slots):
        tensors[f"input::x.{i}"] = {"input_name": f"x.{i}", "shape": [16, 1, 16, 16]}
        in_ids.append(f"input::x.{i}")
    for i in range(slots):
        tensors[f"input::cap_feats.{i}"] = {"input_name": f"cap_feats.{i}", "shape": [10 - 2 * i, 2560]}
        in_ids.append(f"input::cap_feats.{i}")
    tensors["input::t"] = {"input_name": "t", "shape": [slots]}
    in_ids.append("input::t")
    out_ids = []
    for i in range(slots):
        oid = f"aten.clone::{140 + i}::out_0"
        tensors[oid] = {"output_name": f"output_{i}", "shape": [16, 1, 16, 16]}
        out_ids.append(oid)
    return {"tensors": tensors, "input_tensor_ids": in_ids, "output_tensor_ids": out_ids}


def _make_ctx(dag: dict, loop_extra: dict | None = None, comp_attrs: dict | None = None):
    loop_def = {
        "driver": "scheduler",
        "components": ["transformer"],
        "state_variable": "global.x",
        "state_input": "x",
        "index_variable": "global.t",
    }
    if loop_extra:
        loop_def.update(loop_extra)
    topo = {
        "components": {
            "transformer": {
                "type": "neural",
                "interface": {"inputs": ["t"], "outputs": ["output_0"]},
                "shapes": {"t": [2]},
                "attributes": comp_attrs or {},
            }
        },
        "flow": {"pre_loop": ["text_encoder"], "loop": loop_def, "post_loop": ["vae"]},
        "extracted_values": {},
    }
    pkg = SimpleNamespace(topology=topo, manifest={"family": "image"}, cache_path=None, variables={})
    exec_obj = SimpleNamespace(_dag=dag)
    return SimpleNamespace(pkg=pkg, executors={"transformer": exec_obj})


class _FakeNBXTensor:
    """Minimal pure-numpy tensor mimicking NBXTensor methods (no torch)."""

    nbx_dtype = "bfloat16"

    def __init__(self, arr: np.ndarray):
        self._arr = np.ascontiguousarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    def numpy(self):
        return self._arr

    def narrow(self, dim: int, start: int, length: int):
        sl = [slice(None)] * self._arr.ndim
        sl[dim] = slice(start, start + length)
        return _FakeNBXTensor(self._arr[tuple(sl)])

    def squeeze(self, dim: int):
        return _FakeNBXTensor(np.squeeze(self._arr, axis=dim))

    def unsqueeze(self, dim: int):
        return _FakeNBXTensor(np.expand_dims(self._arr, axis=dim))

    def contiguous(self):
        return _FakeNBXTensor(self._arr.copy())

    def __neg__(self):
        return _FakeNBXTensor(-self._arr)

    @classmethod
    def from_numpy(cls, arr, dtype=None, device=None):
        return cls(np.asarray(arr))


# ---------------------------------------------------------------------------
# 1. list_batched_spec detection
# ---------------------------------------------------------------------------

def test_standard_batched_denoiser_returns_none():
    dag = {
        "tensors": {
            "input::hidden_states": {"input_name": "hidden_states", "shape": [1, 16, 64, 64]},
            "input::encoder_hidden_states": {"input_name": "encoder_hidden_states", "shape": [1, 77, 2048]},
            "input::timestep": {"input_name": "timestep", "shape": [1]},
        },
        "input_tensor_ids": ["input::hidden_states", "input::encoder_hidden_states", "input::timestep"],
        "output_tensor_ids": [],
    }
    ctx = _make_ctx(dag, loop_extra={"state_input": "hidden_states"})
    assert list_batched_spec(ctx, "transformer") is None


@pytest.mark.parametrize("slots", [1, 2])
def test_list_batched_spec_detects_nextdit_slots(slots):
    ctx = _make_ctx(_make_nextdit_dag(slots))
    spec = list_batched_spec(ctx, "transformer")
    assert spec is not None
    assert spec["state_prefix"] == "x"
    assert spec["cond_prefix"] == "cap_feats"
    assert spec["slot_count"] == slots
    assert spec["frame_axis"] == 1
    assert spec["timestep_input"] == "t"
    assert spec["output_names"] == [f"output_{i}" for i in range(slots)]


def test_list_batched_spec_refuses_non_contiguous_slots():
    dag = {
        "tensors": {
            "input::x.0": {"input_name": "x.0", "shape": [16, 1, 16, 16]},
            "input::x.2": {"input_name": "x.2", "shape": [16, 1, 16, 16]},
        },
        "input_tensor_ids": ["input::x.0", "input::x.2"],
        "output_tensor_ids": [],
    }
    ctx = _make_ctx(dag)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        list_batched_spec(ctx, "transformer")


def test_list_batched_spec_on_real_zimage_container_if_present():
    container = Path("/mnt/nvme_ai_work/built_models_1127/Z-Image-Turbo/Z-Image-Turbo")
    if not (container / "topology.json").exists():
        pytest.skip("Z-Image-Turbo container not present on this host")
    topo = json.loads((container / "topology.json").read_text())
    pkg = SimpleNamespace(topology=topo, manifest={"family": "image"}, cache_path=container, variables={})
    ctx = SimpleNamespace(pkg=pkg, executors={})
    spec = list_batched_spec(ctx, "transformer")
    assert spec is not None
    assert spec["state_prefix"] == "x"
    assert spec["cond_prefix"] == "cap_feats"
    assert spec["slot_count"] == 2
    assert spec["frame_axis"] == 1
    assert spec["timestep_input"] == "t"
    assert spec["output_names"] == ["output_0", "output_1"]


# ---------------------------------------------------------------------------
# 2. Flow step convention (timestep inversion, velocity sign, CFG formula)
# ---------------------------------------------------------------------------

def test_flow_step_convention_nextdit_vs_standard():
    sched = FlowEulerScheduler({"num_train_timesteps": 1000, "shift": 3.0})
    ctx = _make_ctx(_make_nextdit_dag(2))
    spec = list_batched_spec(ctx, "transformer")

    conv = flow_step_convention(ctx, "transformer", sched, list_spec=spec)
    assert conv == {
        "invert_model_timestep": True,
        "velocity_sign": -1.0,
        "cfg_formula": "pos_plus_scale_diff",
    }

    # Standard denoiser (no list_spec): untouched
    std_conv = flow_step_convention(ctx, "transformer", sched, list_spec=None)
    assert std_conv == {
        "invert_model_timestep": False,
        "velocity_sign": 1.0,
        "cfg_formula": "standard",
    }

    # Explicit overrides in topology take precedence
    ctx_override = _make_ctx(
        _make_nextdit_dag(2),
        loop_extra={"invert_model_timestep": False, "velocity_sign": 1.0, "cfg_formula": "standard"},
    )
    over = flow_step_convention(ctx_override, "transformer", sched, list_spec=spec)
    assert over == {
        "invert_model_timestep": False,
        "velocity_sign": 1.0,
        "cfg_formula": "standard",
    }


def test_apply_timestep_and_velocity_conventions():
    conv = {"invert_model_timestep": True, "velocity_sign": -1.0, "cfg_formula": "pos_plus_scale_diff"}
    sigma = torch.tensor([0.75], dtype=torch.float32)
    t_model = apply_model_timestep_convention(sigma, conv)
    assert torch.allclose(t_model, torch.tensor([0.25]))

    v = torch.tensor([1.0, -2.5])
    assert torch.equal(apply_velocity_sign(v, conv), torch.tensor([-1.0, 2.5]))

    nbx_v = _FakeNBXTensor(np.array([1.0, -2.5], dtype=np.float32))
    np.testing.assert_array_equal(apply_velocity_sign(nbx_v, conv).numpy(), np.array([-1.0, 2.5], dtype=np.float32))


# ---------------------------------------------------------------------------
# 3. Caption trimming (PyTorch + NBXTensor narrow path)
# ---------------------------------------------------------------------------

def test_trim_caption_embedding_torch_and_nbx_match():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((1, 12, 16)).astype(np.float32)
    mask_arr = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    t_out = trim_caption_embedding(torch.from_numpy(arr), torch.from_numpy(mask_arr))
    n_out = trim_caption_embedding(_FakeNBXTensor(arr), _FakeNBXTensor(mask_arr))

    assert len(t_out) == 1 and len(n_out) == 1
    assert tuple(t_out[0].shape) == (5, 16)
    assert tuple(n_out[0].shape) == (5, 16)
    np.testing.assert_allclose(t_out[0].numpy(), n_out[0].numpy())
    np.testing.assert_allclose(t_out[0].numpy(), arr[0, :5])


# ---------------------------------------------------------------------------
# 4. Packing & unpacking (1-slot and 2-slot, CFG on and off)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("do_cfg", [False, True])
def test_build_and_unpack_list_batched_all_modes(slots, do_cfg):
    spec = {
        "state_prefix": "x",
        "cond_prefix": "cap_feats",
        "slot_count": slots,
        "frame_axis": 1,
        "timestep_input": "t",
        "output_names": [f"output_{i}" for i in range(slots)],
    }
    state = torch.randn(1, 4, 8, 8)
    ts = torch.tensor([0.25])
    pos_caps = [torch.randn(6, 16)]
    neg_caps = [torch.randn(4, 16)] if do_cfg else None

    passes = build_list_batched_passes(state, ts, pos_caps, neg_caps, spec, do_cfg)
    expected_passes = 2 if (do_cfg and slots == 1) else 1
    assert len(passes) == expected_passes
    for staged, pass_t in passes:
        assert tuple(pass_t.shape) == (slots,)
        for i in range(slots):
            assert tuple(staged[f"x.{i}"].shape) == (4, 1, 8, 8)
            assert f"cap_feats.{i}" in staged

    pos_pred = torch.full((4, 1, 8, 8), 3.0)
    neg_pred = torch.full((4, 1, 8, 8), 1.0)
    if not do_cfg:
        raw_outs = [{"output_0": pos_pred, "output_1": pos_pred} if slots == 2 else {"output_0": pos_pred}]
        out = unpack_list_batched_output(raw_outs, spec, False, 0.0, "pos_plus_scale_diff")
        assert tuple(out.shape) == (1, 4, 8, 8)
        assert torch.allclose(out, torch.full((1, 4, 8, 8), 3.0))
    else:
        raw_outs = (
            [{"output_0": pos_pred, "output_1": neg_pred}]
            if slots == 2
            else [{"output_0": pos_pred}, {"output_0": neg_pred}]
        )
        out = unpack_list_batched_output(raw_outs, spec, True, 5.0, "pos_plus_scale_diff")
        # pos + 5 * (pos - neg) = 3 + 5 * 2 = 13
        assert tuple(out.shape) == (1, 4, 8, 8)
        assert torch.allclose(out, torch.full((1, 4, 8, 8), 13.0))


# ---------------------------------------------------------------------------
# 5. End-to-end IterativeProcessHandler loop parity with ZImagePipeline math
# ---------------------------------------------------------------------------

def test_iterative_process_handler_nextdit_loop_matches_zimage_math():
    dag = _make_nextdit_dag(slots=2)
    sched_cfg = {
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "sigma_schedule": {"family": "linspace_inclusive"},
    }
    driver = FlowEulerScheduler(sched_cfg)

    topology = {
        "components": {
            "text_encoder": {"type": "neural", "interface": {"inputs": ["input_ids"], "outputs": ["last_hidden_state"]}},
            "transformer": {
                "type": "neural",
                "interface": {"inputs": ["t"], "outputs": ["output_0"]},
                "shapes": {"t": [2]},
            },
        },
        "connections": [{"from": "global.t", "to": "transformer.t"}],
        "flow": {
            "type": "iterative_process",
            "pre_loop": ["text_encoder"],
            "loop": {
                "driver": "scheduler",
                "components": ["transformer"],
                "state_variable": "global.x",
                "state_input": "x",
                "index_variable": "global.t",
            },
            "post_loop": [],
        },
        "extracted_values": {},
    }

    init_latent = torch.ones(1, 16, 4, 4, dtype=torch.float32)
    pos_emb = torch.arange(1 * 6 * 8, dtype=torch.float32).reshape(1, 6, 8)
    pos_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int64)

    class _MockVR:
        mode = "compiled"
        def __init__(self):
            self.resolved = {
                "global.latents": init_latent.clone(),
                "global.num_inference_steps": 4,
                "global.attention_mask": pos_mask,
                "text_encoder.last_hidden_state": pos_emb,
            }
            self.loop_state = {}
        def get(self, k, default=None):
            if k in self.resolved:
                return self.resolved[k]
            raise KeyError(k)
        def set(self, k, v):
            self.resolved[k] = v
        def sampling_generator(self):
            return None
        def resolve_all(self):
            return self.resolved

    vr = _MockVR()
    pkg = SimpleNamespace(
        topology=topology,
        manifest={"family": "image", "model_name": "test-nextdit"},
        defaults={},
        cache_path=None,
        variables={"global.latents": {"type": "tensor", "role": "state_tensor"}},
    )
    exec_obj = SimpleNamespace(_dag=dag)
    plan = SimpleNamespace(components={"text_encoder": SimpleNamespace(dtype=torch.float32)})
    ctx = FlowContext(
        pkg=pkg,
        plan=plan,
        strategy=None,
        modules={"scheduler": driver},
        executors={"transformer": exec_obj},
        variable_resolver=vr,
        connections_index={"transformer": {"t": ["global.t"]}},
        loop_id="primary_loop",
        primary_device="cpu",
        nbx_path_str="",
    )

    ir = InputResolver(vr, {"transformer": {"t": ["global.t"]}}, topology, ctx.loop_id, ctx.executors)
    seen_calls = []

    def _mock_exec(comp_name, phase, loop_timestep=None):
        if comp_name != "transformer":
            return {}
        if loop_timestep is not None:
            vr.loop_state[ctx.loop_id] = loop_timestep
        inp = ir.resolve_component_inputs(comp_name)
        seen_calls.append({k: v.clone() for k, v in inp.items()})
        # Deterministic velocity: returns +2.0 * x.0 (so negated velocity = -2.0 * x)
        return {"output_0": inp["x.0"] * 2.0, "output_1": inp["x.1"] * 2.0}

    handler = IterativeProcessHandler(
        ctx=ctx,
        execute_component_fn=_mock_exec,
        extract_primary_output_fn=lambda c, o: o["output_0"] if isinstance(o, dict) else o,
        unload_weights_fn=lambda *a, **kw: None,
    )
    handler._execute_main_loop("scheduler", ["transformer"], do_cfg=False, guidance_scale=0.0)

    assert len(seen_calls) == 4
    # Reference ZImagePipeline math: sigmas from linspace_inclusive(4) shifted by 3.0
    ref_sched = FlowEulerScheduler(sched_cfg)
    ref_sched.set_timesteps(4)
    ref_state = init_latent.clone()
    for step_idx, sigma in enumerate(ref_sched.timesteps):
        call = seen_calls[step_idx]
        # Model received t_model = 1.0 - sigma with shape [2]
        assert torch.allclose(call["t"], torch.full((2,), 1.0 - float(sigma.item())))
        # Model received x.0 [16, 1, 4, 4] and trimmed cap_feats.0 [3, 8]
        assert tuple(call["x.0"].shape) == (16, 1, 4, 4)
        assert tuple(call["cap_feats.0"].shape) == (3, 8)
        assert torch.equal(call["cap_feats.0"], pos_emb[0, :3])
        # Negated velocity fed to scheduler.step
        noise_pred = -(ref_state * 2.0)
        ref_state = ref_sched.step(noise_pred, sigma, ref_state)["prev_sample"]

    assert torch.allclose(vr.get("global.x"), ref_state, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. FlowEuler linspace_inclusive schedule matches diffusers Z-Image-Turbo
# ---------------------------------------------------------------------------

def test_flow_euler_linspace_inclusive_matches_zimage_turbo_sigmas():
    cfg = {
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "sigma_schedule": {"family": "linspace_inclusive"},
    }
    s = FlowEulerScheduler(cfg)
    s.set_timesteps(9)
    # Exact diffusers FlowMatchEulerDiscreteScheduler(shift=3.0, sigma_min=0.0).set_timesteps(9)
    raw = np.linspace(1.0, 0.0, 9)
    expected = 3.0 * raw / (1.0 + 2.0 * raw)
    np.testing.assert_allclose(s.sigmas.numpy(), expected, atol=1e-6)
    assert float(s.sigmas[-1].item()) == 0.0


# ---------------------------------------------------------------------------
# 7. VAE shift_factor
# ---------------------------------------------------------------------------

def test_vae_handler_applies_scaling_and_shift_factor():
    cfg = ComponentConfig(
        class_name="AutoencoderKL",
        component_type="vae",
        scaling_factor=0.3611,
        shift_factor=0.1159,
    )
    handler = VAEComponentHandler(cfg, cache_path=Path("/tmp"))
    z = torch.ones(1, 16, 4, 4, dtype=torch.float32) * 2.0
    out = handler.transform_inputs({"z": z}, phase="post_loop")["z"]
    expected = (z / 0.3611) + 0.1159
    assert torch.allclose(out, expected, atol=1e-6)

    # Absent shift_factor: only scaling_factor applied
    cfg_no_shift = ComponentConfig(class_name="AutoencoderKL", component_type="vae", scaling_factor=0.5)
    out2 = VAEComponentHandler(cfg_no_shift, cache_path=Path("/tmp")).transform_inputs({"z": z}, phase="post_loop")["z"]
    assert torch.allclose(out2, z / 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. TextProcessor chat_template for diffusion
# ---------------------------------------------------------------------------

def test_text_processor_diffusion_applies_chat_template_when_declared():
    tok = MagicMock()
    tok.has_chat_template.return_value = True
    tok.apply_chat_template.return_value = "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
    tok.return_value = {"input_ids": [[1, 2, 3, 0]], "attention_mask": [[1, 1, 1, 0]]}

    topo = {
        "components": {"text_encoder": {"shapes": {"input_ids": [1, 4]}}},
        "extracted_values": {
            "tokenizer": {
                "max_sequence_length": 4,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        },
    }
    vr = SimpleNamespace(mode="compiled", resolved={})
    tp = TextProcessor(tok, defaults={}, topology=topo, variable_resolver=vr)
    ids, mask = tp.tokenize_for_diffusion("a cat", device="cpu")
    tok.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "a cat"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert tuple(ids.shape) == (1, 4)
    assert tuple(mask.shape) == (1, 4)


def test_text_processor_diffusion_handles_add_generation_prompt_in_chat_template_kwargs():
    calls = []

    class _StrictChatTokenizer:
        def has_chat_template(self):
            return True

        def apply_chat_template(
            self,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        ):
            calls.append((messages, tokenize, add_generation_prompt, enable_thinking))
            return f"templated:{messages[0]['content']}"

        def __call__(self, prompt=None, text=None, max_length=None, padding=None, truncation=None):
            return {"input_ids": [[1, 2, 3, 0]], "attention_mask": [[1, 1, 1, 0]]}

    topo = {
        "components": {"text_encoder": {"shapes": {"input_ids": [1, 4]}}},
        "extracted_values": {
            "tokenizer": {
                "max_sequence_length": 4,
                "chat_template_kwargs": {
                    "add_generation_prompt": True,
                    "enable_thinking": True,
                    "tokenize": True,
                },
            }
        },
    }
    vr = SimpleNamespace(mode="compiled", resolved={})
    tp = TextProcessor(_StrictChatTokenizer(), defaults={}, topology=topo, variable_resolver=vr)

    ids, mask = tp.tokenize_for_diffusion("a cat", device="cpu")
    neg_ids, neg_mask = tp.tokenize_negative(device="cpu", negative_prompt="")

    assert calls == [
        ([{"role": "user", "content": "a cat"}], False, True, True),
        ([{"role": "user", "content": ""}], False, True, True),
    ]
    assert tuple(ids.shape) == (1, 4)
    assert tuple(mask.shape) == (1, 4)
    assert tuple(neg_ids.shape) == (1, 4)
    assert tuple(neg_mask.shape) == (1, 4)


# ---------------------------------------------------------------------------
# 9. R33: list_batched_denoiser imports cleanly without torch
# ---------------------------------------------------------------------------

def test_list_batched_denoiser_is_torch_free_at_import(monkeypatch):
    mod_name = "neurobrix.core.runtime.resolution.list_batched_denoiser"
    saved = sys.modules.pop(mod_name, None)
    monkeypatch.setitem(sys.modules, "torch", None)
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "list_batched_spec")
    finally:
        if saved is not None:
            sys.modules[mod_name] = saved
def test_list_batched_denoiser_ast_has_zero_torch_imports():
    import ast
    mod_path = Path(__file__).resolve().parents[3] / "src/neurobrix/core/runtime/resolution/list_batched_denoiser.py"
    tree = ast.parse(mod_path.read_text(), filename=str(mod_path))
    torch_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    torch_imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "torch" or mod.startswith("torch."):
                torch_imports.append((node.lineno, mod))
    assert torch_imports == [], f"Forbidden torch imports in list_batched_denoiser.py: {torch_imports}"


@pytest.mark.parametrize("count", [1, 2])
def test_repeat_timestep_scalar_on_torch_and_nbx(monkeypatch, count):
    # PyTorch state (including bfloat16 state -> float32 timestep on same device)
    torch_state_f32 = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
    t_f32 = _repeat_timestep(0.25, count, state=torch_state_f32)
    assert isinstance(t_f32, torch.Tensor)
    assert t_f32.dtype == torch.float32
    assert tuple(t_f32.shape) == (count,)
    assert torch.allclose(t_f32, torch.full((count,), 0.25, dtype=torch.float32))

    torch_state_bf16 = torch.zeros(1, 4, 4, 4, dtype=torch.bfloat16)
    t_bf16 = _repeat_timestep(1, count, state=torch_state_bf16)
    assert t_bf16.dtype == torch.float32
    assert tuple(t_bf16.shape) == (count,)
    assert torch.allclose(t_bf16, torch.full((count,), 1.0, dtype=torch.float32))

    # Fake NBXTensor state (pure numpy, no GPU)
    nbx_state = _FakeNBXTensor(np.zeros((1, 4, 4, 4), dtype=np.float32))
    t_nbx = _repeat_timestep(0.25, count, state=nbx_state)
    assert isinstance(t_nbx, _FakeNBXTensor)
    assert tuple(t_nbx.shape) == (count,)
    assert t_nbx.numpy().dtype == np.float32
    np.testing.assert_allclose(t_nbx.numpy(), np.full((count,), 0.25, dtype=np.float32))

    # NBXTensor-shaped object without class from_numpy delegates to nbxtensor.tensor.NBXTensor.from_numpy
    import nbxtensor.tensor as nbx_mod
    calls = []
    def _fake_from_numpy(arr, dtype=None, device=None):
        calls.append((arr.copy(), arr.dtype, device))
        return _FakeNBXTensor(arr)
    monkeypatch.setattr(nbx_mod.NBXTensor, "from_numpy", staticmethod(_fake_from_numpy))
    nbx_like = SimpleNamespace(nbx_dtype="bfloat16", _device="cuda", _device_idx=1, shape=(1, 4, 4, 4))
    t_mock = _repeat_timestep(0.75, count, state=nbx_like)
    assert tuple(t_mock.shape) == (count,)
    assert len(calls) == 1
    assert calls[0][1] == np.float32 and calls[0][2] == 1
    np.testing.assert_allclose(calls[0][0], np.full((count,), 0.75, dtype=np.float32))


def test_trim_caption_embedding_real_nbxtensor_with_detach_and_cross_masks():
    """Regression: real NBXTensor implements .detach() and has no .bool();
    _is_nbx_tensor must recognize it and handle NBX/numpy/torch masks."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor

    rng = np.random.default_rng(7)
    arr = rng.standard_normal((1, 8, 16)).astype(np.float32)
    mask_arr = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int64)

    nbx_emb = NBXTensor.from_raw(arr.ctypes.data, arr.shape, NBXDtype.float32, device="cpu", base=arr)
    nbx_mask = NBXTensor.from_raw(mask_arr.ctypes.data, mask_arr.shape, NBXDtype.int64, device="cpu", base=mask_arr)
    assert hasattr(nbx_emb, "detach") and not hasattr(nbx_mask, "bool")

    # 1) NBXTensor emb + NBXTensor mask (the exact live Z-Image-Turbo Triton path)
    out_nbx = trim_caption_embedding(nbx_emb, nbx_mask)
    assert len(out_nbx) == 1
    assert isinstance(out_nbx[0], NBXTensor)
    assert tuple(out_nbx[0].shape) == (4, 16)
    np.testing.assert_allclose(out_nbx[0].numpy(), arr[0, :4])

    # 2) NBXTensor emb + numpy mask + torch mask
    out_np = trim_caption_embedding(nbx_emb, mask_arr)
    out_pt = trim_caption_embedding(nbx_emb, torch.from_numpy(mask_arr))
    np.testing.assert_allclose(out_np[0].numpy(), arr[0, :4])
    np.testing.assert_allclose(out_pt[0].numpy(), arr[0, :4])

    # 3) torch.Tensor emb + NBXTensor mask (must not call m.bool())
    out_cross = trim_caption_embedding(torch.from_numpy(arr), nbx_mask)
    assert isinstance(out_cross[0], torch.Tensor)
    np.testing.assert_allclose(out_cross[0].numpy(), arr[0, :4])

    # 4) _FakeNBXTensor carrying .detach() (guards against hasattr(t, "detach") trap)
    class _FakeWithDetach(_FakeNBXTensor):
        def detach(self):
            return self

    out_fake = trim_caption_embedding(_FakeWithDetach(arr), _FakeWithDetach(mask_arr))
    assert tuple(out_fake[0].shape) == (4, 16)
    np.testing.assert_allclose(out_fake[0].numpy(), arr[0, :4])


def test_parse_dtype_preserves_complex64_and_complex128():
    """Complex dtypes (used by NextDiT / Z-Image 3D RoPE) must never fall back to float32."""
    import torch
    from neurobrix.core.dtype.config import get_torch_dtype, is_floating_name, parse_dtype

    assert parse_dtype("torch.complex64") == torch.complex64
    assert parse_dtype("complex64", compute_dtype=torch.float16) == torch.complex64
    assert parse_dtype("torch.complex128", compute_dtype=torch.bfloat16) == torch.complex128
    assert get_torch_dtype("complex64") == torch.complex64
    assert get_torch_dtype("complex128") == torch.complex128
    assert is_floating_name("complex64") is False
    assert is_floating_name("complex128") is False


def test_extract_caption_embeddings_uses_topology_declared_output_key():
    """When topology.connections wires text_encoder.output_35 -> transformer.cap_feats.0,
    extract_caption_embeddings must select output_35 rather than last_hidden_state."""
    import torch
    from types import SimpleNamespace
    from neurobrix.core.runtime.resolution.list_batched_denoiser import (
        extract_caption_embeddings,
        resolve_caption_source_key,
        resolve_encoder_hidden_key,
    )

    out_35 = torch.full((1, 4, 8), 35.0)
    last_hs = torch.full((1, 4, 8), 99.0)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    vr = SimpleNamespace(resolved={
        "text_encoder.output_35": out_35,
        "text_encoder.last_hidden_state": last_hs,
        "global.attention_mask": mask,
    })
    topo = {
        "flow": {"pre_loop": ["text_encoder"], "loop": {"components": ["transformer"]}},
        "connections": [
            {"from": "text_encoder.output_35", "to": "transformer.cap_feats.0"},
            {"from": "text_encoder.output_35", "to": "transformer.cap_feats.1"},
        ],
    }
    ctx = SimpleNamespace(pkg=SimpleNamespace(topology=topo), variable_resolver=vr)

    assert resolve_caption_source_key(ctx, "text_encoder", "transformer", "cap_feats") == "text_encoder.output_35"
    assert resolve_encoder_hidden_key(ctx, "text_encoder") == "text_encoder.output_35"

    pos_caps, neg_caps = extract_caption_embeddings(ctx, do_cfg=False, comp_name="transformer", cond_prefix="cap_feats")
    assert neg_caps is None
    assert len(pos_caps) == 1
    assert tuple(pos_caps[0].shape) == (2, 8)
    assert torch.all(pos_caps[0] == 35.0)
