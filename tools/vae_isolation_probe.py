"""VAE isolation discriminant: feed sequential-pipeline VAE input into triton VAE.

Phase 1 of P-SANA-4KPX-RUNTIME post-page-blanche investigation. The
cross-variant analysis showed VAE ops have rel_ratio (4Kpx/1024) up to
2.9M× — by far the largest shape-specific divergences. Open question:

  - Is the triton VAE itself buggy at Sana 4Kpx shape, OR
  - Does it just decode a corrupted latent that the triton transformer
    produced?

This tool answers the question binarily by capturing the latent that
the SEQUENTIAL pipeline feeds to its VAE (Sana 4Kpx sequential PNG is
coherent, so this latent is "correct"), then forcing the TRITON VAE to
decode the same latent.

Result interpretation:
  - **Cas A**: triton VAE output coherent (red apple) → VAE triton OK,
    bug is in transformer triton 4Kpx. Path 2: audit transformer.
  - **Cas B**: triton VAE output garbage → VAE triton has a shape-
    dependent intrinsic bug. Path 3: microtest VAE TOP ops
    (silu::18-23, pixel_shuffle::3, convolution::61) at exact shapes.
  - **Cas C**: partial / structured-but-wrong → mixed; both paths
    matter.

Implementation: monkey-patches `RuntimeExecutor._execute_component` at
import time inside this script. No NeuroBrix code modification.

Usage:
    python tools/test_vae_isolation.py --capture     # phase A (sequential)
    python tools/test_vae_isolation.py --replay      # phase B (triton)
    python tools/test_vae_isolation.py --both        # phase A then B

Outputs in validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/:
  - vae_isolation_input.pt           (saved torch tensors comp_inputs of seq VAE)
  - vae_isolation_seq_decode.png     (sequential VAE output — sanity check)
  - vae_isolation_tri_decode.png     (triton VAE output of SAME latent — discriminant)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System")
OUT_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05"
DUMP_PATH = OUT_DIR / "vae_isolation_input.pt"
SEQ_PNG = OUT_DIR / "vae_isolation_seq_decode.png"
TRI_PNG = OUT_DIR / "vae_isolation_tri_decode.png"

# Worker scripts run in subprocess (clean import state, isolation between
# sequential and triton paths). Each worker monkey-patches
# RuntimeExecutor._execute_component before the pipeline runs.

CAPTURE_WORKER = r'''
import sys, os, torch
sys.path.insert(0, "/home/mlops/NeuroBrix_System/src")
from neurobrix.core.runtime.executor import RuntimeExecutor

DUMP_PATH = os.environ["NBX_VAE_ISO_DUMP"]
_orig = RuntimeExecutor._execute_component
_dumped = {"done": False}

def _patched(self, comp_name, phase="default", loop_timestep=None):
    """Capture VAE inputs on first VAE invocation, then run normally."""
    if comp_name == "vae" and not _dumped["done"]:
        # Walk the prep path (resolve+synthesize+transform) up to where
        # comp_inputs is finalized just before executor.run(). We do
        # this by calling the original up-to-prep code via a partial:
        # easier path is to let the original run, but capture by
        # monkey-patching the inner executor.run wrapping.
        pass
    # Actually intercept via a wrap of the per-component executor.run:
    if comp_name == "vae" and not _dumped["done"]:
        executor = self.executors.get(comp_name)
        if executor is not None and hasattr(executor, "run"):
            _orig_run = executor.run
            def _captured_run(comp_inputs):
                # Save inputs (torch tensors) to disk before kernel run
                payload = {}
                for k, v in comp_inputs.items():
                    if isinstance(v, torch.Tensor):
                        payload[k] = {
                            "kind": "torch",
                            "tensor": v.detach().cpu().contiguous(),
                            "dtype": str(v.dtype),
                            "device": str(v.device),
                            "shape": tuple(v.shape),
                        }
                    else:
                        payload[k] = {"kind": "scalar", "value": v}
                torch.save(payload, DUMP_PATH)
                print(f"[VAE_ISO] captured VAE inputs to {DUMP_PATH}", flush=True)
                _dumped["done"] = True
                executor.run = _orig_run  # restore
                return _orig_run(comp_inputs)
            executor.run = _captured_run
    return _orig(self, comp_name, phase, loop_timestep)

RuntimeExecutor._execute_component = _patched

# Now run neurobrix CLI as if normal, via direct import
from neurobrix.cli import main as run_main
sys.argv = [
    "neurobrix", "run",
    "--model", "Sana_1600M_4Kpx_BF16",
    "--sequential",
    "--prompt", "a red apple",
    "--steps", "12",
    "--output", os.environ["NBX_VAE_ISO_OUTPUT"],
]
run_main()
'''

REPLAY_WORKER = r'''
import sys, os, torch
sys.path.insert(0, "/home/mlops/NeuroBrix_System/src")
from neurobrix.core.runtime.executor import RuntimeExecutor

DUMP_PATH = os.environ["NBX_VAE_ISO_DUMP"]
print(f"[VAE_ISO] loading saved VAE inputs from {DUMP_PATH}", flush=True)
_saved_payload = torch.load(DUMP_PATH, weights_only=False)
print(f"[VAE_ISO] payload keys: {list(_saved_payload.keys())}", flush=True)

_orig = RuntimeExecutor._execute_component
_replayed = {"done": False}

def _torch_to_nbx_or_torch(t, target_device):
    """Move a torch.Tensor to the executor's expected representation.
    For triton-sequential, the executor receives torch.Tensor at the
    component-graph boundary (NBX path is internal to the kernels).
    The torch tensor must be on the target device."""
    if isinstance(target_device, str) and target_device.startswith("cuda"):
        return t.to(target_device)
    return t.cuda()

def _patched(self, comp_name, phase="default", loop_timestep=None):
    if comp_name == "vae" and not _replayed["done"]:
        executor = self.executors.get(comp_name)
        if executor is not None and hasattr(executor, "run"):
            _orig_run = executor.run
            def _replayed_run(comp_inputs):
                # Override comp_inputs with saved tensors from sequential capture
                target_dev = None
                for v in comp_inputs.values():
                    if isinstance(v, torch.Tensor):
                        target_dev = v.device
                        break
                if target_dev is None:
                    target_dev = "cuda:2"
                new_inputs = {}
                for k, payload in _saved_payload.items():
                    if payload.get("kind") == "torch":
                        t = payload["tensor"].to(target_dev)
                        new_inputs[k] = t
                        print(f"[VAE_ISO] override [{k}] shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}", flush=True)
                    else:
                        new_inputs[k] = payload.get("value")
                _replayed["done"] = True
                executor.run = _orig_run
                return _orig_run(new_inputs)
            executor.run = _replayed_run
    return _orig(self, comp_name, phase, loop_timestep)

RuntimeExecutor._execute_component = _patched

from neurobrix.cli import main as run_main
sys.argv = [
    "neurobrix", "run",
    "--model", "Sana_1600M_4Kpx_BF16",
    "--triton-sequential",
    "--prompt", "a red apple",
    "--steps", "12",
    "--output", os.environ["NBX_VAE_ISO_OUTPUT"],
]
run_main()
'''


def run_worker(worker_code: str, output_png: Path, env_extra: dict | None = None):
    """Spawn a clean Python subprocess running the worker code."""
    env = os.environ.copy()
    env["NBX_VAE_ISO_DUMP"] = str(DUMP_PATH)
    env["NBX_VAE_ISO_OUTPUT"] = str(output_png)
    env["NBX_ALLOC_POOL"] = "1"
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", worker_code],
        env=env,
    )
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()

    if args.both:
        args.capture = True
        args.replay = True
    if not (args.capture or args.replay):
        ap.error("specify --capture, --replay, or --both")

    if args.capture:
        print("=" * 60)
        print("PHASE A — capture sequential VAE inputs")
        print("=" * 60)
        rc = run_worker(CAPTURE_WORKER, SEQ_PNG)
        print(f"capture exit {rc}")
        if rc != 0:
            sys.exit(rc)
        if not DUMP_PATH.exists():
            print(f"[FAIL] dump file not created: {DUMP_PATH}")
            sys.exit(2)
        print(f"[OK] captured: {DUMP_PATH} ({DUMP_PATH.stat().st_size} bytes)")

    if args.replay:
        print("=" * 60)
        print("PHASE B — replay saved latent through triton VAE")
        print("=" * 60)
        if not DUMP_PATH.exists():
            print(f"[FAIL] need {DUMP_PATH} (run --capture first)")
            sys.exit(2)
        rc = run_worker(REPLAY_WORKER, TRI_PNG)
        print(f"replay exit {rc}")
        if rc != 0:
            sys.exit(rc)
        print(f"[OK] decoded: {TRI_PNG}")

    print()
    print("=" * 60)
    print("DISCRIMINANT — visual R29 inspection required")
    print("=" * 60)
    print(f"  sequential decode: {SEQ_PNG}")
    print(f"  triton    decode: {TRI_PNG}  <-- compare to seq")
    print("  Cas A: triton coherent  → VAE triton OK, bug in transformer")
    print("  Cas B: triton garbage   → VAE triton bug shape-dependent")


if __name__ == "__main__":
    main()
