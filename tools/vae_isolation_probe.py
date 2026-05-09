"""VAE-only isolation probe (Phase 1 path 1, refactor v2).

Discriminant goal: feed sequential 4Kpx final-latent (post-handler-
transform) into triton-sequential VAE component standalone — bypassing
text_encoder + transformer entirely. Result discriminates:

  Cas A — coherent (red apple): VAE triton OK at 4Kpx; bug is upstream
          in transformer triton.
  Cas B — garbage (green texture): VAE triton has shape-dependent bug.
  Cas C — OOM: even VAE alone exceeds budget (unlikely on V100 32GB
          since VAE peak is ~5-10 GB).

Refactor over v1: v1 monkey-patched RuntimeExecutor._execute_component
which still required the FULL diffusion pipeline to run before the
override fired (text_encoder + 12 transformer iters → 30 GB → OOM
before VAE). v2 uses public APIs only:

  1. NBXContainer.load(nbx_path)              — load .nbx
  2. PrismSolver().solve_smart(...)           — get execution plan
  3. NBXRuntimeLoader().load(nbx_path)        — hydrate runtime pkg
  4. RuntimeExecutor(pkg, plan, mode)         — build executors dict
  5. executor._ensure_weights_loaded("vae")   — lazy-load VAE only
  6. executor.executors["vae"].run({"z": z})  — direct VAE invocation
  7. save output as PNG

Steps 5-7 use the per-component VAE GraphExecutor directly, skipping
the `executor.execute()` orchestration which would run pre_loop +
loop phases. No NeuroBrix code change, no monkey-patching.

Usage:
    python tools/vae_isolation_probe.py --capture            # phase A
    python tools/vae_isolation_probe.py --vae-only-decode    # phase B'
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System")
OUT_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05"
DUMP_PATH = OUT_DIR / "vae_isolation_input.pt"
SEQ_PNG = OUT_DIR / "vae_isolation_seq_decode.png"
TRI_PNG = OUT_DIR / "vae_isolation_tri_decode.png"

sys.path.insert(0, str(ROOT / "src"))


def _capture_phase(model_name: str, output_png: Path, dump_path: Path) -> int:
    """Phase A: full sequential 4Kpx run, capture VAE inputs at the
    moment they reach executor.run(). Sequential mode peaks ~10 GB on
    V100 → fits within budget."""
    worker = f"""
import sys, os, torch
sys.path.insert(0, "{ROOT}/src")
from neurobrix.core.runtime.executor import RuntimeExecutor

DUMP_PATH = "{dump_path}"
_orig = RuntimeExecutor._execute_component
_dumped = {{"done": False}}

def _patched(self, comp_name, phase="default", loop_timestep=None):
    if comp_name == "vae" and not _dumped["done"]:
        executor = self.executors.get(comp_name)
        if executor is not None and hasattr(executor, "run"):
            _orig_run = executor.run
            def _captured_run(comp_inputs):
                payload = {{}}
                for k, v in comp_inputs.items():
                    if isinstance(v, torch.Tensor):
                        payload[k] = {{"kind": "torch",
                                       "tensor": v.detach().cpu().contiguous(),
                                       "dtype": str(v.dtype),
                                       "device": str(v.device),
                                       "shape": tuple(v.shape)}}
                    else:
                        payload[k] = {{"kind": "scalar", "value": v}}
                torch.save(payload, DUMP_PATH)
                print(f"[VAE_ISO] captured to {{DUMP_PATH}}", flush=True)
                _dumped["done"] = True
                executor.run = _orig_run
                return _orig_run(comp_inputs)
            executor.run = _captured_run
    return _orig(self, comp_name, phase, loop_timestep)

RuntimeExecutor._execute_component = _patched

from neurobrix.cli import main as run_main
sys.argv = ["neurobrix", "run", "--model", "{model_name}",
            "--sequential", "--prompt", "a red apple",
            "--steps", "12", "--output", "{output_png}"]
run_main()
"""
    env = os.environ.copy()
    env["NBX_ALLOC_POOL"] = "1"
    return subprocess.run([sys.executable, "-c", worker], env=env).returncode


def _vae_only_decode(model_name: str, dump_path: Path, output_png: Path,
                      mode: str = "triton_sequential") -> int:
    """VAE-only decode: build runtime, lazy-load VAE weights only, feed
    saved latent directly to vae_executor.run(). No text_encoder, no
    transformer. ~5-10 GB peak on V100 32GB."""
    import torch
    from neurobrix.nbx import NBXContainer
    from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.executor import RuntimeExecutor
    from neurobrix.cli.commands.run import find_model

    nbx_path = find_model(model_name)
    print(f"[VAE_ISO] loading container: {nbx_path}", flush=True)
    container = NBXContainer.load(str(nbx_path))
    cache_path = container._cache_path
    assert cache_path is not None
    defaults_path = cache_path / "runtime" / "defaults.json"
    cached_defaults = json.load(defaults_path.open()) if defaults_path.exists() else {}

    height = cached_defaults.get("height", 1024)
    width = cached_defaults.get("width", 1024)
    vae_scale = cached_defaults.get("vae_scale_factor", 8)
    print(f"[VAE_ISO] resolution {height}x{width}, vae_scale {vae_scale}", flush=True)

    hardware_id = get_or_create_default_profile()
    hw_profile = load_profile(hardware_id)
    input_config = InputConfig(batch_size=2, height=height, width=width,
                                dtype="float16", vae_scale=vae_scale)

    print(f"[VAE_ISO] solving Prism plan...", flush=True)
    solver = PrismSolver()
    execution_plan = solver.solve_smart(container, hw_profile, input_config)
    print(f"[VAE_ISO] strategy: {execution_plan.strategy}", flush=True)
    for cn, alloc in execution_plan.components.items():
        print(f"[VAE_ISO]   {cn} → {alloc.device}", flush=True)

    print(f"[VAE_ISO] loading runtime package...", flush=True)
    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))

    print(f"[VAE_ISO] building executor mode={mode}...", flush=True)
    executor = RuntimeExecutor(pkg, execution_plan, mode=mode)
    print(f"[VAE_ISO] running executor.setup() (modules + executors + strategy)...",
          flush=True)
    executor.setup()
    print(f"[VAE_ISO] executor.executors keys: {list(executor.executors)}",
          flush=True)

    if "vae" not in executor.executors:
        print(f"[VAE_ISO] FATAL: no 'vae' in executors. keys={list(executor.executors)}",
              flush=True)
        return 2
    vae_exec = executor.executors["vae"]

    print(f"[VAE_ISO] loading VAE weights only...", flush=True)
    executor._ensure_weights_loaded("vae")

    print(f"[VAE_ISO] loading saved latent: {dump_path}", flush=True)
    saved = torch.load(dump_path, weights_only=False)
    print(f"[VAE_ISO] payload keys: {list(saved.keys())}", flush=True)

    target_dev = None
    for v in saved.values():
        if v.get("kind") == "torch":
            target_dev = v["device"]
            break
    if target_dev is None:
        target_dev = "cuda:2"
    print(f"[VAE_ISO] target device: {target_dev}", flush=True)

    comp_inputs = {}
    for k, payload in saved.items():
        if payload.get("kind") == "torch":
            t = payload["tensor"].to(target_dev)
            comp_inputs[k] = t
            print(f"[VAE_ISO]   [{k}] shape={tuple(t.shape)} dtype={t.dtype}",
                  flush=True)
        else:
            comp_inputs[k] = payload.get("value")

    # Reproduce the standard pipeline's tiling integration
    # (executor.py:870-882). At Sana 4Kpx, VAE input (1,32,128,128)
    # exceeds trace size — the TilingEngine slices spatially and
    # invokes vae_exec.run on each tile, then blends. Without this,
    # VAE allocates full 4Kpx upsample chain immediately → > 32 GB.
    vae_tiling = executor._component_tiling.get("vae")
    print(f"[VAE_ISO] _component_tiling['vae'] = {vae_tiling!r}", flush=True)
    if vae_tiling is not None:
        # Pick the first 4D spatial tensor in comp_inputs (= our 'z').
        spatial_input = None
        for v in comp_inputs.values():
            if hasattr(v, 'dim') and v.dim() == 4:
                spatial_input = v
                break
        if spatial_input is not None and vae_tiling.should_tile(spatial_input):
            input_name = next(iter(comp_inputs.keys()))
            print(f"[VAE_ISO] tiling enabled: input='{input_name}' "
                  f"shape={tuple(spatial_input.shape)}", flush=True)

            def execute_tile(tile):
                result = vae_exec.run({input_name: tile})
                if isinstance(result, dict):
                    result = next(iter(result.values()))
                return result

            print(f"[VAE_ISO] invoking tiling.tiled_execute()...", flush=True)
            output = vae_tiling.tiled_execute(spatial_input, execute_tile)
        else:
            print(f"[VAE_ISO] tiling configured but should_tile=False; "
                  f"running vae_exec.run() directly", flush=True)
            output = vae_exec.run(comp_inputs)
    else:
        print(f"[VAE_ISO] no tiling for VAE; running vae_exec.run() directly",
              flush=True)
        output = vae_exec.run(comp_inputs)
    print(f"[VAE_ISO] VAE forward done. output type: {type(output).__name__}",
          flush=True)

    print(f"[VAE_ISO] saving output as PNG to {output_png}...", flush=True)
    import torch as _torch
    if isinstance(output, dict):
        output_tensor = None
        for v in output.values():
            if hasattr(v, 'shape') and len(getattr(v, 'shape', [])) >= 3:
                output_tensor = v
                break
        if output_tensor is None:
            output_tensor = next(iter(output.values()))
    else:
        output_tensor = output

    if hasattr(output_tensor, 'detach'):
        out_t = output_tensor.detach().cpu().float()
    elif hasattr(output_tensor, 'to_torch'):
        out_t = output_tensor.to_torch().detach().cpu().float()
    else:
        out_t = _torch.tensor(output_tensor)

    if out_t.dim() == 4:
        img = out_t[0]
    else:
        img = out_t
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    img = (img * 255.0).round().to(_torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    from PIL import Image
    Image.fromarray(img).save(output_png)
    print(f"[VAE_ISO] saved: {output_png}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--vae-only-decode", action="store_true",
                    help="Decode saved latent through triton-sequential VAE only")
    ap.add_argument("--model", default="Sana_1600M_4Kpx_BF16")
    args = ap.parse_args()

    if not (args.capture or args.vae_only_decode):
        ap.error("specify --capture and/or --vae-only-decode")

    if args.capture:
        print("=" * 60)
        print("PHASE A — capture sequential VAE inputs")
        print("=" * 60)
        rc = _capture_phase(args.model, SEQ_PNG, DUMP_PATH)
        if rc != 0 or not DUMP_PATH.exists():
            print(f"[FAIL] capture rc={rc}, dump exists={DUMP_PATH.exists()}")
            sys.exit(rc or 2)
        print(f"[OK] {DUMP_PATH} ({DUMP_PATH.stat().st_size} bytes)")

    if args.vae_only_decode:
        print("=" * 60)
        print("PHASE B' — VAE-only decode of saved latent (triton)")
        print("=" * 60)
        if not DUMP_PATH.exists():
            print(f"[FAIL] need {DUMP_PATH} (run --capture first)")
            sys.exit(2)
        rc = _vae_only_decode(args.model, DUMP_PATH, TRI_PNG,
                               mode="triton_sequential")
        if rc != 0:
            sys.exit(rc)

    print()
    print("=" * 60)
    print("DISCRIMINANT — visual R29")
    print("=" * 60)
    print(f"  oracle (sequential): {SEQ_PNG}")
    print(f"  triton VAE only:    {TRI_PNG}")
    print("  Cas A: triton coherent → VAE triton OK; bug in transformer")
    print("  Cas B: triton garbage  → VAE triton bug shape-dependent")


if __name__ == "__main__":
    main()
