"""POINT 4 ÉTAPE A — capture relu::15 input + output in both modes.

If inputs match between seq and tri but outputs diverge → relu wrapper
is the bug (kernel-level).
If inputs differ → relu::15 is innocent, walk back upstream to find
the true root.
"""
import sys
import os
import json
import torch
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

OUT_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05"
DUMP_PATH = OUT_DIR / "vae_isolation_input.pt"
MODEL = "Sana_1600M_4Kpx_BF16"

# Captured for each mode: input + output of relu::15
CAPTURED = {"sequential": {}, "triton_sequential": {}}


def to_torch_cpu(t):
    if isinstance(t, NBXTensor):
        return nbx_to_torch(t).detach().cpu().contiguous()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().contiguous()
    return None


def build_executor(mode: str):
    nbx_path = find_model(MODEL)
    container = NBXContainer.load(str(nbx_path))
    cache_path = container._cache_path
    assert cache_path is not None
    defaults_path = cache_path / "runtime" / "defaults.json"
    defaults = json.load(defaults_path.open()) if defaults_path.exists() else {}
    height = defaults.get("height", 1024)
    width = defaults.get("width", 1024)
    vae_scale = defaults.get("vae_scale_factor", 8)
    hardware_id = get_or_create_default_profile()
    hw_profile = load_profile(hardware_id)
    input_config = InputConfig(batch_size=2, height=height, width=width,
                                dtype="float16", vae_scale=vae_scale)
    solver = PrismSolver()
    plan = solver.solve_smart(container, hw_profile, input_config)
    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))
    executor = RuntimeExecutor(pkg, plan, mode=mode)
    executor.setup()
    minimal_inputs = {"global.prompt": "a red apple", "global.num_inference_steps": 12}
    md = executor._prepare_defaults(minimal_inputs)
    executor._init_variable_resolver(minimal_inputs, md)
    executor._set_runtime_resolution_on_executors(md)
    executor._init_helpers()
    executor._ensure_weights_loaded("vae")
    return executor


def make_interceptor(mode):
    def interceptor(*args, **kwargs):
        x = args[0]
        x_cpu = to_torch_cpu(x)
        CAPTURED[mode]["input"] = x_cpu
        # Mode-correct dispatch
        if mode == "sequential":
            assert isinstance(x, torch.Tensor)
            result = torch.ops.aten.relu(x)  # type: ignore
        else:
            assert isinstance(x, NBXTensor)
            from neurobrix.kernels.wrappers import relu as nbx_relu
            result = nbx_relu(x)
        CAPTURED[mode]["output"] = to_torch_cpu(result)
        return result
    return interceptor


def run_mode(mode: str):
    print("=" * 70)
    print(f"CAPTURE relu::15 mode={mode}")
    print("=" * 70)
    executor = build_executor(mode)
    vae_exec = executor.executors["vae"]
    if not hasattr(vae_exec, "_op_uid_interceptors"):
        vae_exec._op_uid_interceptors = {}
    vae_exec._op_uid_interceptors["aten.relu::15"] = make_interceptor(mode)

    saved = torch.load(DUMP_PATH, weights_only=False)
    target_dev = None
    for v in saved.values():
        if v.get("kind") == "torch":
            target_dev = v["device"]
            break
    target_dev = target_dev or "cuda:2"
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
        print(f"[capture {mode}] run stopped: {type(e).__name__}: {str(e)[:200]}")
    inp = CAPTURED[mode].get("input")
    out = CAPTURED[mode].get("output")
    print(f"[capture {mode}] input shape={tuple(inp.shape) if inp is not None else None} "
          f"dtype={inp.dtype if inp is not None else None}")
    print(f"[capture {mode}] output shape={tuple(out.shape) if out is not None else None} "
          f"dtype={out.dtype if out is not None else None}")


run_mode("sequential")
import gc; gc.collect(); torch.cuda.empty_cache()
run_mode("triton_sequential")
gc.collect(); torch.cuda.empty_cache()


def stats(seq_t, tri_t, label):
    print(f"\n=== {label} diff seq vs tri ===")
    if seq_t is None or tri_t is None:
        print(f"missing: seq={seq_t is not None} tri={tri_t is not None}")
        return
    if seq_t.shape != tri_t.shape:
        print(f"shape mismatch: seq={seq_t.shape} tri={tri_t.shape}")
        return
    if seq_t.dtype != tri_t.dtype:
        print(f"dtype mismatch: seq={seq_t.dtype} tri={tri_t.dtype}")
    diff = (seq_t.float() - tri_t.float()).abs()
    seq_max = float(seq_t.abs().float().max())
    tri_max = float(tri_t.abs().float().max())
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    n_total = seq_t.numel()
    n_diff = int((diff > 0).sum())
    n_diff_1e3 = int((diff > 1e-3).sum())
    n_diff_1e2 = int((diff > 1e-2).sum())
    rel_max = max(abs(seq_max), abs(tri_max), 1e-9)
    print(f"  shape={tuple(seq_t.shape)} dtype={seq_t.dtype}")
    print(f"  seq max_abs={seq_max:.4g} | tri max_abs={tri_max:.4g} | rel_diff={abs(seq_max-tri_max)/rel_max:.4f}")
    print(f"  max_d={max_d:.4g} mean_d={mean_d:.4g}")
    print(f"  n_diff > 0    = {n_diff} / {n_total}  ({100*n_diff/n_total:.4f}%)")
    print(f"  n_diff > 1e-3 = {n_diff_1e3} / {n_total}  ({100*n_diff_1e3/n_total:.4f}%)")
    print(f"  n_diff > 1e-2 = {n_diff_1e2} / {n_total}  ({100*n_diff_1e2/n_total:.4f}%)")


seq_in = CAPTURED["sequential"].get("input")
tri_in = CAPTURED["triton_sequential"].get("input")
seq_out = CAPTURED["sequential"].get("output")
tri_out = CAPTURED["triton_sequential"].get("output")

stats(seq_in, tri_in, "INPUT to relu::15")
stats(seq_out, tri_out, "OUTPUT of relu::15")

# Verdict
if seq_in is not None and tri_in is not None and seq_in.shape == tri_in.shape:
    in_diff = (seq_in.float() - tri_in.float()).abs()
    in_seq_max = float(seq_in.abs().float().max())
    in_tri_max = float(tri_in.abs().float().max())
    in_rel = abs(in_seq_max - in_tri_max) / max(abs(in_seq_max), abs(in_tri_max), 1e-9)
    in_max_d = float(in_diff.max())
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if in_rel > 0.05 or in_max_d > 1.0:
        print(f"INPUT TO relu::15 ALREADY DIVERGES (rel={in_rel:.3f}, max_d={in_max_d:.4g})")
        print("relu::15 is INNOCENT — root cause is upstream.")
        print("Walk-back required to find true root.")
    else:
        print(f"INPUT TO relu::15 MATCHES (rel={in_rel:.3f}, max_d={in_max_d:.4g})")
        print("relu::15 is the ROOT — wrapper-level value divergence.")
        print("Audit kernel path next.")
