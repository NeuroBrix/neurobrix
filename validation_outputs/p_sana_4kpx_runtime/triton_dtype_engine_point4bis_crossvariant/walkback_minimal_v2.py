"""POINT 4-bis — cross-variant walkback, Sana 1024 + 4Kpx, ops 0..100.

Same monkey-patched dispatcher capture as walkback_minimal.py but
parameterized on MODEL env var. For Sana 1024 (no saved latent),
use synthetic Gaussian fp32 latent shape derived from graph.json.
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
from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher
from neurobrix.triton.sequential import TritonSequentialDispatcher

MODEL = os.environ.get("MODEL", "Sana_1600M_4Kpx_BF16")
print(f"[walkback_v2] MODEL={MODEL}")

GRAPH_PATH = Path(f"/home/mlops/.neurobrix/cache/{MODEL}/components/vae/graph.json")
SAVED_LATENT = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/vae_isolation_input.pt"

MAX_OP_INDEX = 100

SEQ_LOG: dict = {}
TRI_LOG: dict = {}


def _stats(t):
    if t is None:
        return None
    if isinstance(t, NBXTensor):
        try:
            t = nbx_to_torch(t)
        except Exception:
            return None
    if not isinstance(t, torch.Tensor):
        return None
    if t.numel() == 0:
        return None
    if not t.is_floating_point():
        return ("non_float", str(t.dtype), tuple(t.shape))
    flat = t.flatten()
    n = flat.numel()
    idxs = [0, n // 8, n // 4, n // 2, 3 * n // 4][:5]
    fp = [float(flat[i].item()) for i in idxs]
    return (float(t.abs().float().max()), fp, str(t.dtype), tuple(t.shape))


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


_orig_seq = NativeATenDispatcher.dispatch
_orig_tri = TritonSequentialDispatcher.dispatch
COUNTER = {"sequential": [0], "triton_sequential": [0]}


def _logged_seq(self, op_type, inputs, attributes):
    result = _orig_seq(self, op_type, inputs, attributes)
    if COUNTER["sequential"][0] < MAX_OP_INDEX:
        op_uid = _get_caller_op_uid() or op_type
        if isinstance(result, torch.Tensor):
            SEQ_LOG[op_uid] = (COUNTER["sequential"][0], _stats(result))
        elif isinstance(result, (list, tuple)) and result and isinstance(result[0], torch.Tensor):
            SEQ_LOG[op_uid] = (COUNTER["sequential"][0], _stats(result[0]))
    COUNTER["sequential"][0] += 1
    return result


def _logged_tri(self, op_type, inputs, attributes):
    result = _orig_tri(self, op_type, inputs, attributes)
    if COUNTER["triton_sequential"][0] < MAX_OP_INDEX:
        op_uid = _get_caller_op_uid() or op_type
        if isinstance(result, NBXTensor):
            TRI_LOG[op_uid] = (COUNTER["triton_sequential"][0], _stats(result))
        elif isinstance(result, (list, tuple)) and result and isinstance(result[0], NBXTensor):
            TRI_LOG[op_uid] = (COUNTER["triton_sequential"][0], _stats(result[0]))
    COUNTER["triton_sequential"][0] += 1
    return result


def get_latent(target_dev):
    """Use saved latent if Sana 4Kpx; synthesize for Sana 1024."""
    if "4Kpx" in MODEL or "4kpx" in MODEL:
        # Saved latent
        saved = torch.load(SAVED_LATENT, weights_only=False)
        for k, p in saved.items():
            if p.get("kind") == "torch":
                return {k: p["tensor"].to(target_dev)}
        return {}
    # Synthetic for 1024 (and others)
    g = json.load(GRAPH_PATH.open())
    ops = g.get("ops")
    ops_list = list(ops.values()) if isinstance(ops, dict) else ops
    first_op = ops_list[0]
    in_shapes = first_op.get("input_shapes", [[1, 32, 32, 32]])
    shape = tuple(in_shapes[0])
    print(f"  [synth_latent] {MODEL} shape={shape}")
    torch.manual_seed(42)
    return {"z": torch.randn(shape, dtype=torch.float32, device=target_dev) * 1.5}


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


def run_mode(mode):
    print(f"\n=== {mode} {MODEL} ===")
    executor = build_executor(mode)
    vae_exec = executor.executors["vae"]
    target_dev = "cuda:2"
    comp_inputs = get_latent(target_dev)
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
    log = SEQ_LOG if mode == "sequential" else TRI_LOG
    print(f"  captured {len(log)} ops")


NativeATenDispatcher.dispatch = _logged_seq
TritonSequentialDispatcher.dispatch = _logged_tri

run_mode("sequential")
import gc; gc.collect(); torch.cuda.empty_cache()
run_mode("triton_sequential")
gc.collect(); torch.cuda.empty_cache()

NativeATenDispatcher.dispatch = _orig_seq
TritonSequentialDispatcher.dispatch = _orig_tri

# Cross-reference and save TSV
g = json.load(GRAPH_PATH.open())
exec_order = g.get("execution_order", [])
ops = g.get("ops", {})
ops_list = list(ops.values()) if isinstance(ops, dict) else ops
ordered_uids = exec_order if exec_order and isinstance(exec_order[0], str) else [o.get("op_uid") for o in ops_list]

OUT_TSV = Path(f"/tmp/walkback_{('1024' if '1024' in MODEL else '4kpx')}_first100.tsv")
with open(OUT_TSV, "w") as f:
    f.write("\t".join(["op_idx", "op_uid", "shape", "seq_max", "tri_max", "rel_max_abs",
                       "fp_max_d", "seq_dtype", "tri_dtype"]) + "\n")
    for op_idx, uid in enumerate(ordered_uids[:MAX_OP_INDEX]):
        if not uid:
            continue
        s_entry = SEQ_LOG.get(uid)
        t_entry = TRI_LOG.get(uid)
        if not (s_entry and t_entry):
            continue
        s_stats = s_entry[1]
        t_stats = t_entry[1]
        if s_stats is None or t_stats is None or isinstance(s_stats[0], str) or isinstance(t_stats[0], str):
            continue
        s_max, s_fp, s_dt, s_sh = s_stats
        t_max, t_fp, t_dt, t_sh = t_stats
        rel = abs(s_max - t_max) / max(abs(s_max), abs(t_max), 1e-9)
        fp_diffs = [abs(a - b) for a, b in zip(s_fp, t_fp)]
        fp_max_d = max(fp_diffs) if fp_diffs else 0
        sh = "x".join(str(x) for x in s_sh)
        f.write("\t".join([
            str(op_idx), uid, sh,
            f"{s_max:.6g}", f"{t_max:.6g}", f"{rel:.6f}",
            f"{fp_max_d:.6g}", s_dt, t_dt,
        ]) + "\n")

print(f"\nTSV: {OUT_TSV}")
