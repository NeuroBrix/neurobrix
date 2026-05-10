"""POINT 4 walk-back minimal — find FIRST value-divergent op among first 50.

Capture only max_abs and a small fingerprint per op (no full tensor
storage). Run both modes; compare per-op stats. First op with rel
diff > 0.001 in max_abs OR diverging fingerprint = root candidate.
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

DUMP_PATH = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/vae_isolation_input.pt"
MODEL = "Sana_1600M_4Kpx_BF16"

# Cap at first N ops worth of stats
MAX_OP_INDEX = 500

SEQ_LOG: dict = {}
TRI_LOG: dict = {}


def _stats(t):
    """Returns (max_abs, fingerprint5, dtype_str, shape) or None."""
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
    # fingerprint at 5 positions
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
SEQ_COUNTER = [0]
TRI_COUNTER = [0]


def _logged_seq(self, op_type, inputs, attributes):
    result = _orig_seq(self, op_type, inputs, attributes)
    if SEQ_COUNTER[0] < MAX_OP_INDEX:
        op_uid = _get_caller_op_uid() or op_type
        if isinstance(result, torch.Tensor):
            SEQ_LOG[op_uid] = (SEQ_COUNTER[0], _stats(result))
        elif isinstance(result, (list, tuple)) and result and isinstance(result[0], torch.Tensor):
            SEQ_LOG[op_uid] = (SEQ_COUNTER[0], _stats(result[0]))
    SEQ_COUNTER[0] += 1
    return result


def _logged_tri(self, op_type, inputs, attributes):
    result = _orig_tri(self, op_type, inputs, attributes)
    if TRI_COUNTER[0] < MAX_OP_INDEX:
        op_uid = _get_caller_op_uid() or op_type
        if isinstance(result, NBXTensor):
            TRI_LOG[op_uid] = (TRI_COUNTER[0], _stats(result))
        elif isinstance(result, (list, tuple)) and result and isinstance(result[0], NBXTensor):
            TRI_LOG[op_uid] = (TRI_COUNTER[0], _stats(result[0]))
    TRI_COUNTER[0] += 1
    return result


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
    print(f"\n=== {mode} ===")
    executor = build_executor(mode)
    vae_exec = executor.executors["vae"]
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

# Cross-reference
graph_path = Path(f"/home/mlops/.neurobrix/cache/{MODEL}/components/vae/graph.json")
g = json.load(graph_path.open())
exec_order = g.get("execution_order", [])
ops = g.get("ops", {})
ops_list = list(ops.values()) if isinstance(ops, dict) else ops
ordered_uids = exec_order if exec_order and isinstance(exec_order[0], str) else [o.get("op_uid") for o in ops_list]

print("\n" + "=" * 110)
print("FIRST 500 ops + value divergence (only ops with rel > 0.005 OR fp_max_d > 0.1)")
print("=" * 110)
print(f"{'op_idx':>6} {'op_uid':<28} {'shape':<22} {'seq_max':>10} {'tri_max':>10} {'rel':>8} {'fp_max_d':>10} {'verdict'}")
first_div = None
for op_idx, uid in enumerate(ordered_uids[:500]):
    if not uid:
        continue
    s_entry = SEQ_LOG.get(uid)
    t_entry = TRI_LOG.get(uid)
    if not (s_entry and t_entry):
        continue
    s_stats = s_entry[1]
    t_stats = t_entry[1]
    if s_stats is None or t_stats is None:
        continue
    if isinstance(s_stats[0], str):  # non_float
        continue
    if isinstance(t_stats[0], str):
        continue
    s_max = s_stats[0]
    t_max = t_stats[0]
    s_fp = s_stats[1]
    t_fp = t_stats[1]
    if len(s_fp) != len(t_fp):
        continue
    rel = abs(s_max - t_max) / max(abs(s_max), abs(t_max), 1e-9)
    fp_diffs = [abs(a - b) for a, b in zip(s_fp, t_fp)]
    fp_max_d = max(fp_diffs) if fp_diffs else 0
    sh = "x".join(str(x) for x in s_stats[3])
    is_div_strict = rel > 0.005 or fp_max_d > 0.1
    is_div_strong = rel > 0.05 or fp_max_d > 5.0
    if is_div_strict:
        flag = "STRONG" if is_div_strong else "weak"
        print(f"{op_idx:>6} {uid:<28} {sh:<22} {s_max:>10.4g} {t_max:>10.4g} {rel:>8.4f} {fp_max_d:>10.4g} {flag}")
    if first_div is None and is_div_strong:
        first_div = (op_idx, uid, rel, fp_max_d, s_fp, t_fp)

print()
if first_div:
    print(f"FIRST value divergence:")
    print(f"  op_idx={first_div[0]} op_uid={first_div[1]} rel={first_div[2]:.6f} fp_max_d={first_div[3]:.4g}")
    print(f"  seq fingerprint: {first_div[4]}")
    print(f"  tri fingerprint: {first_div[5]}")
else:
    print("NO value divergence found in first 50 ops")
