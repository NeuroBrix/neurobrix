"""Capture conv::36 output via runtime path (no interceptor replacement).

Purpose: see what tiled_conv2d_spatial produces in seq vs tri mode at 4Kpx.
We DO NOT replace the interceptor — instead we monkey-patch dispatchers
to log the result of conv::36 specifically.
"""
import sys
import os
import json
import torch
import inspect
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NBX_DISABLE_AUTOTUNE", "1")

from neurobrix.nbx import NBXContainer
from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
from neurobrix.core.prism.autodetect import get_or_create_default_profile
from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.core.runtime.executor import RuntimeExecutor
from neurobrix.cli.commands.run import find_model
from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch

DUMP_PATH = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/vae_isolation_input.pt"
MODEL = "Sana_1600M_4Kpx_BF16"

# Targets: capture output of THESE op_uids when their interceptor returns
TARGETS = ["aten.convolution::36", "aten.convolution::41", "aten.convolution::46", "aten.convolution::48", "aten.upsample_nearest2d::2"]

CAPTURED = {"sequential": {}, "triton_sequential": {}}


def to_torch_cpu(t):
    if isinstance(t, NBXTensor):
        return nbx_to_torch(t).detach().cpu().contiguous()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().contiguous()
    return None


def stats(t, label):
    if t is None:
        return None
    if not t.is_floating_point():
        return f"{label}: non-float dtype={t.dtype} shape={tuple(t.shape)}"
    return {
        "max_abs": float(t.abs().float().max()),
        "mean_abs": float(t.abs().float().mean()),
        "dtype": str(t.dtype),
        "shape": tuple(t.shape),
    }


def build_executor(mode):
    nbx_path = find_model(MODEL)
    container = NBXContainer.load(str(nbx_path))
    cache_path = container._cache_path
    defaults = json.load((cache_path / "runtime" / "defaults.json").open())
    height = defaults.get("height", 1024)
    width = defaults.get("width", 1024)
    vae_scale = defaults.get("vae_scale_factor", 8)
    hardware_id = get_or_create_default_profile()
    hw_profile = load_profile(hardware_id)
    input_config = InputConfig(batch_size=2, height=height, width=width, dtype="float16", vae_scale=vae_scale)
    plan = PrismSolver().solve_smart(container, hw_profile, input_config)
    pkg = NBXRuntimeLoader().load(str(nbx_path))
    executor = RuntimeExecutor(pkg, plan, mode=mode)
    executor.setup()
    md = executor._prepare_defaults({"global.prompt": "a red apple", "global.num_inference_steps": 12})
    executor._init_variable_resolver({"global.prompt": "a red apple", "global.num_inference_steps": 12}, md)
    executor._set_runtime_resolution_on_executors(md)
    executor._init_helpers()
    executor._ensure_weights_loaded("vae")
    return executor


def wrap_existing_interceptor(vae_exec, mode):
    """Wrap the runtime's existing interceptors to capture output."""
    if not hasattr(vae_exec, "_op_uid_interceptors"):
        return 0
    existing = vae_exec._op_uid_interceptors
    n_wrapped = 0
    for uid in TARGETS:
        if uid in existing:
            orig = existing[uid]
            def make_wrapped(_orig, _uid):
                def wrapped(*args, **kwargs):
                    result = _orig(*args, **kwargs)
                    out_cpu = to_torch_cpu(result if not isinstance(result, (list, tuple)) else result[0])
                    CAPTURED[mode][_uid] = stats(out_cpu, _uid)
                    CAPTURED[mode][_uid + "_data"] = out_cpu
                    return result
                return wrapped
            existing[uid] = make_wrapped(orig, uid)
            n_wrapped += 1
        else:
            # Op not intercepted — let it run via dispatch + wrap dispatch
            pass
    return n_wrapped


# For ops that aren't intercepted, monkey-patch dispatch to log them.
from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher
from neurobrix.triton.sequential import TritonSequentialDispatcher

_orig_seq_disp = NativeATenDispatcher.dispatch
_orig_tri_disp = TritonSequentialDispatcher.dispatch


def _get_caller_op_uid():
    frame = inspect.currentframe()
    if frame is None:
        return None
    for _ in range(6):
        frame = frame.f_back
        if frame is None:
            return None
        op_uid = frame.f_locals.get("op_uid")
        if isinstance(op_uid, str):
            return op_uid
    return None


def _logged_seq(self, op_type, inputs, attributes):
    result = _orig_seq_disp(self, op_type, inputs, attributes)
    op_uid = _get_caller_op_uid()
    if op_uid in TARGETS:
        out_cpu = to_torch_cpu(result if not isinstance(result, (list, tuple)) else result[0])
        CAPTURED["sequential"][op_uid] = stats(out_cpu, op_uid)
        CAPTURED["sequential"][op_uid + "_data"] = out_cpu
    return result


def _logged_tri(self, op_type, inputs, attributes):
    result = _orig_tri_disp(self, op_type, inputs, attributes)
    op_uid = _get_caller_op_uid()
    if op_uid in TARGETS:
        out_cpu = to_torch_cpu(result if not isinstance(result, (list, tuple)) else result[0])
        CAPTURED["triton_sequential"][op_uid] = stats(out_cpu, op_uid)
        CAPTURED["triton_sequential"][op_uid + "_data"] = out_cpu
    return result


def run_mode(mode):
    print(f"\n=== {mode} ===")
    executor = build_executor(mode)
    vae_exec = executor.executors["vae"]
    n_wrapped = wrap_existing_interceptor(vae_exec, mode)
    print(f"  wrapped {n_wrapped} runtime interceptors")
    saved = torch.load(DUMP_PATH, weights_only=False)
    target_dev = "cuda:2"
    for v in saved.values():
        if v.get("kind") == "torch":
            target_dev = v["device"]
            break
    comp_inputs = {}
    for k, p in saved.items():
        if p.get("kind") == "torch":
            comp_inputs[k] = p["tensor"].to(target_dev)
        else:
            comp_inputs[k] = p.get("value")
    vae_tiling = executor._component_tiling.get("vae")
    try:
        if vae_tiling is not None:
            spatial_input = next((v for v in comp_inputs.values()
                                  if hasattr(v, 'dim') and v.dim() == 4), None)
            if spatial_input is not None and vae_tiling.should_tile(spatial_input):
                input_name = next(iter(comp_inputs.keys()))
                def execute_tile(tile):
                    return vae_exec.run({input_name: tile})
                vae_tiling.tiled_execute(spatial_input, execute_tile)
            else:
                vae_exec.run(comp_inputs)
        else:
            vae_exec.run(comp_inputs)
    except Exception as e:
        print(f"  stopped: {type(e).__name__}: {str(e)[:150]}")
    for uid in TARGETS:
        s = CAPTURED[mode].get(uid)
        if s and isinstance(s, dict):
            print(f"  {uid}: max_abs={s['max_abs']:.4g} mean={s['mean_abs']:.4g} dtype={s['dtype']} shape={s['shape']}")


NativeATenDispatcher.dispatch = _logged_seq
TritonSequentialDispatcher.dispatch = _logged_tri

run_mode("sequential")
import gc; gc.collect(); torch.cuda.empty_cache()
run_mode("triton_sequential")
gc.collect(); torch.cuda.empty_cache()

NativeATenDispatcher.dispatch = _orig_seq_disp
TritonSequentialDispatcher.dispatch = _orig_tri_disp


def diff_stats(label, seq_t, tri_t):
    if seq_t is None or tri_t is None:
        print(f"\n{label}: missing"); return
    if seq_t.shape != tri_t.shape:
        print(f"\n{label}: shape mismatch"); return
    diff = (seq_t.float() - tri_t.float()).abs()
    seq_max = float(seq_t.abs().float().max())
    tri_max = float(tri_t.abs().float().max())
    rel = abs(seq_max - tri_max) / max(seq_max, tri_max, 1e-9)
    max_d = float(diff.max())
    frac = float((diff > 1e-3).float().mean())
    print(f"\n{label}: shape={tuple(seq_t.shape)} seq dtype={seq_t.dtype} tri dtype={tri_t.dtype}")
    print(f"  seq_max={seq_max:.4g} tri_max={tri_max:.4g} rel={rel:.4f} max_d={max_d:.4g} frac>1e-3={frac:.4f}")


print("\n" + "=" * 70)
print("CROSS-MODE DIFFS")
print("=" * 70)
for uid in TARGETS:
    diff_stats(uid, CAPTURED["sequential"].get(uid + "_data"), CAPTURED["triton_sequential"].get(uid + "_data"))
