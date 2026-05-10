"""POINT 6 H2 microtest — discriminate kernel-bug vs upstream-divergence
on op 667 conv::55 of the Sana 4Kpx VAE.

Strategy: capture pre_input + weight at conv::55 in BOTH modes; then run
`_fused_upsample_conv2d_torch` and `_fused_upsample_conv2d_nbx` on the
SAME captured inputs (bit-equal). Two outcomes:

  A. nbx output bit-equal to torch output  → kernel innocent; the
     45% rel divergence in the H2 audit came from upstream
     (pre_input or weight already diverged between modes)
  B. nbx output diverges                  → kernel bug remains in
     `_fused_upsample_conv2d_nbx` despite POINT 5 halo fix

The microtest also captures the conv::55 input from each mode and
compares them — that orthogonally tells us if the divergence is
input-driven (upstream interceptor problem) vs kernel-driven.
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
from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch, DeviceAllocator


def _t2n(t):
    """torch.Tensor -> NBXTensor via numpy. fp16/fp32 supported.
    bf16 cast to fp32 first (numpy has no bf16)."""
    if t is None:
        return None
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return NBXTensor.from_numpy(t.contiguous().cpu().numpy())
from neurobrix.kernels.ops.fused_upsample_conv import (
    _fused_upsample_conv2d_torch, _fused_upsample_conv2d_nbx,
    FusionUpsampleProxy,
)

DUMP_PATH = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/vae_isolation_input.pt"
MODEL = "Sana_1600M_4Kpx_BF16"
CONV55_UID = "aten.convolution::55"
UPSAMPLE3_UID = "aten.upsample_nearest2d::3"

CAPTURED = {"seq": {}, "tri": {}}


def _capture_proxy_state(proxy, mode):
    state = {
        "pre_input": nbx_to_torch(proxy.pre_input).cpu() if isinstance(proxy.pre_input, NBXTensor) else proxy.pre_input.cpu(),
        "scales_h": proxy.scales_h,
        "scales_w": proxy.scales_w,
        "output_shape": proxy.output_shape,
    }
    CAPTURED[mode]["proxy_state"] = state


def _capture_conv55_call(args, kwargs, mode):
    inp = args[0]
    w = args[1]
    b = args[2] if len(args) > 2 else kwargs.get("bias")
    if isinstance(inp, FusionUpsampleProxy):
        _capture_proxy_state(inp, mode)
        CAPTURED[mode]["got_proxy"] = True
    else:
        CAPTURED[mode]["got_proxy"] = False
        CAPTURED[mode]["input_tensor"] = (
            nbx_to_torch(inp).cpu() if isinstance(inp, NBXTensor) else inp.cpu()
        )

    w_t = nbx_to_torch(w).cpu() if isinstance(w, NBXTensor) else w.cpu()
    CAPTURED[mode]["weight"] = w_t

    if b is not None:
        b_t = nbx_to_torch(b).cpu() if isinstance(b, NBXTensor) else b.cpu()
        CAPTURED[mode]["bias"] = b_t
    else:
        CAPTURED[mode]["bias"] = None

    # Positional fallback: _conv signature is (input, weight, bias,
    # stride, padding, dilation, transposed, output_padding, groups)
    def _arg(i, name, default):
        if i < len(args):
            return args[i]
        return kwargs.get(name, default)
    CAPTURED[mode]["stride"]   = _arg(3, "stride",   1)
    CAPTURED[mode]["padding"]  = _arg(4, "padding",  0)
    CAPTURED[mode]["dilation"] = _arg(5, "dilation", 1)
    CAPTURED[mode]["groups"]   = _arg(8, "groups",   1)


def build_executor(mode):
    nbx = find_model(MODEL)
    container = NBXContainer.load(str(nbx))
    defaults = json.load((container._cache_path / "runtime" / "defaults.json").open())
    hw_profile = load_profile(get_or_create_default_profile())
    ic = InputConfig(batch_size=2, height=4096, width=4096, dtype="float16",
                     vae_scale=defaults.get("vae_scale_factor", 8))
    plan = PrismSolver().solve_smart(container, hw_profile, ic)
    pkg = NBXRuntimeLoader().load(str(nbx))
    executor = RuntimeExecutor(pkg, plan, mode=mode)
    executor.setup()
    md = executor._prepare_defaults({"global.prompt": "a", "global.num_inference_steps": 12})
    executor._init_variable_resolver({"global.prompt": "a", "global.num_inference_steps": 12}, md)
    executor._set_runtime_resolution_on_executors(md)
    executor._init_helpers()
    executor._ensure_weights_loaded("vae")
    return executor


def run_mode(mode):
    executor = build_executor(mode)
    vae_exec = executor.executors["vae"]

    # Wrap conv::55 interceptor to capture
    if hasattr(vae_exec, "_op_uid_interceptors") and CONV55_UID in vae_exec._op_uid_interceptors:
        orig = vae_exec._op_uid_interceptors[CONV55_UID]
        capture_key = "seq" if mode == "sequential" else "tri"
        def wrapped(*args, **kwargs):
            _capture_conv55_call(args, kwargs, capture_key)
            return orig(*args, **kwargs)
        vae_exec._op_uid_interceptors[CONV55_UID] = wrapped
        print(f"  wrapped {CONV55_UID} in {mode} mode")
    else:
        print(f"  WARNING: {CONV55_UID} not in interceptors in {mode} mode")
        return

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
            si = next((v for v in comp_inputs.values() if hasattr(v, 'dim') and v.dim() == 4), None)
            if si is not None and vae_tiling.should_tile(si):
                input_name = next(iter(comp_inputs.keys()))
                vae_tiling.tiled_execute(si, lambda t: vae_exec.run({input_name: t}))
            else:
                vae_exec.run(comp_inputs)
        else:
            vae_exec.run(comp_inputs)
    except Exception as e:
        print(f"  {mode} stopped: {type(e).__name__}: {str(e)[:120]}")


print("=== capture sequential ===")
run_mode("sequential")
import gc; gc.collect(); torch.cuda.empty_cache()
print("=== capture triton_sequential ===")
run_mode("triton_sequential")
gc.collect(); torch.cuda.empty_cache()


# --- Analysis ---
print("\n=== capture summary ===")
for mode in ("seq", "tri"):
    if not CAPTURED[mode]:
        print(f"  {mode}: NO CAPTURE")
        continue
    c = CAPTURED[mode]
    print(f"  {mode}: got_proxy={c.get('got_proxy')}")
    if c.get("got_proxy"):
        ps = c["proxy_state"]
        pre = ps["pre_input"]
        print(f"    pre_input.shape={tuple(pre.shape)} dtype={pre.dtype}")
        print(f"    pre_input.max_abs={float(pre.abs().float().max()):.4g}")
        print(f"    scales=({ps['scales_h']},{ps['scales_w']}) out_shape={ps['output_shape']}")
    w = c["weight"]
    print(f"    weight.shape={tuple(w.shape)} dtype={w.dtype} max_abs={float(w.abs().float().max()):.4g}")
    if c["bias"] is not None:
        b = c["bias"]
        print(f"    bias.shape={tuple(b.shape)} max_abs={float(b.abs().float().max()):.4g}")

# Cross-mode comparison
print("\n=== seq vs tri input bit-equality ===")
if CAPTURED["seq"] and CAPTURED["tri"]:
    s, t = CAPTURED["seq"], CAPTURED["tri"]
    if s.get("got_proxy") and t.get("got_proxy"):
        sp = s["proxy_state"]["pre_input"].float()
        tp = t["proxy_state"]["pre_input"].float()
        if sp.shape == tp.shape:
            max_d = float((sp - tp).abs().max())
            rel = max_d / max(float(sp.abs().max()), 1e-9)
            frac_diff = float((sp - tp).abs().gt(1e-3).float().mean())
            print(f"  pre_input: max_abs_diff={max_d:.4g}  rel={rel:.4f}  frac>1e-3={frac_diff:.4f}")
        else:
            print(f"  pre_input shape MISMATCH: seq={tuple(sp.shape)} tri={tuple(tp.shape)}")
    sw = s["weight"].float()
    tw = t["weight"].float()
    if sw.shape == tw.shape:
        max_d = float((sw - tw).abs().max())
        print(f"  weight: max_abs_diff={max_d:.4g}")

# Run both kernel paths on the SAME captured inputs (use TRI's capture
# so dtype matches what the runtime tri path actually sees: fp16)
print("\n=== same-input kernel diff: torch vs nbx path (tri-capture fp16 input) ===")
if CAPTURED["tri"] and CAPTURED["tri"].get("got_proxy"):
    s = CAPTURED["tri"]
    ps = s["proxy_state"]
    device = "cuda:2"
    DeviceAllocator.set_device(2)

    pre_t = ps["pre_input"].to(device).contiguous()
    w_t = s["weight"].to(device).contiguous()
    b_t = s["bias"].to(device).contiguous() if s["bias"] is not None else None

    proxy = FusionUpsampleProxy(pre_t, ps["scales_h"], ps["scales_w"], ps["output_shape"])
    stride = s["stride"]
    padding = s["padding"]
    dilation = s["dilation"]
    groups = s["groups"]

    # Skip tile_factor=1 (full-frame at 4Kpx OOMs the torch F.conv2d workspace).
    for tile_factor in (2, 4, 8, 16):
        print(f"\n  --- tile_factor={tile_factor} ---")
        out_torch = _fused_upsample_conv2d_torch(
            proxy, w_t, b_t, stride, padding, dilation, False, 0, groups, tile_factor,
        )
        pre_nbx = _t2n(pre_t)
        w_nbx = _t2n(w_t)
        b_nbx = _t2n(b_t) if b_t is not None else None
        proxy_nbx = FusionUpsampleProxy(pre_nbx, ps["scales_h"], ps["scales_w"], ps["output_shape"])
        out_nbx = _fused_upsample_conv2d_nbx(
            proxy_nbx, w_nbx, b_nbx, stride, padding, dilation, False, 0, groups, tile_factor,
        )
        out_nbx_t = nbx_to_torch(out_nbx).cpu().float()
        out_torch_cpu = out_torch.cpu().float()
        if out_nbx_t.shape != out_torch_cpu.shape:
            print(f"  SHAPE MISMATCH: torch={tuple(out_torch_cpu.shape)} nbx={tuple(out_nbx_t.shape)}")
            continue
        diff = (out_nbx_t - out_torch_cpu).abs()
        max_d = float(diff.max())
        rel = max_d / max(float(out_torch_cpu.abs().max()), 1e-9)
        frac = float(diff.gt(1e-3).float().mean())
        frac_big = float(diff.gt(1.0).float().mean())
        print(f"  torch.max_abs={float(out_torch_cpu.abs().max()):.4g}  nbx.max_abs={float(out_nbx_t.abs().max()):.4g}")
        print(f"  max_abs_diff={max_d:.4g}  rel={rel:.4f}  frac>1e-3={frac:.4f}  frac>1.0={frac_big:.4f}")
        # Free NBX
        del out_nbx, pre_nbx, w_nbx
        if b_nbx is not None: del b_nbx
        import gc; gc.collect()
