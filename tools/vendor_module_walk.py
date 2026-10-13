#!/usr/bin/env python3
"""The vendor's transformer, module by module, on OUR pinned latent.

CHANTIER 2, step two. The step ladder established where to look and what shape
the answer must have: our post-denoise latent differs from the vendor's by
**34 % after a SINGLE forward pass**, with a decaying gain (1.243 -> 1.112) over a
**stable ~24 % structural residual present from step one**. So the target is a
structural difference inside ONE pass, not a drift that accumulates — and the
image-side signature is near-Nyquist in ROWS, so something that treats rows
differently from columns is the thread.

WHAT THIS PRODUCES
------------------
One record per module invocation, in execution order:

    {"call": k, "module": "blocks.12.attn1.to_q", "class": "Linear",
     "shape": [...], "l2_norm": ..., "head10": [...], "dtype": "..."}

`call` counts invocations, because classifier-free guidance runs the transformer
more than once per step and the two passes must not be averaged into one row.

WHY IT ALIGNS WITH OURS WITHOUT A MAPPING TABLE
-----------------------------------------------
Our container records `parent_module` per op — the vendor module path the op was
traced from — and `NBX_DUMP_TIDS` already writes `(component, op_uid, op_type,
shape, head10, l2_norm)` per op. So our ops carry the vendor's own module names,
and the walk aligns by name. Nothing here needs a hand-written correspondence,
which is what makes it maintainable across models.

Run under the diffusers_video venv (torch is allowed under tools/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--latents", required=True, help=".npy initial latent (ours)")
    ap.add_argument("--steps", type=int, default=1,
                    help="1 isolates a single forward pass, which is where the "
                         "structural residual already sits")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", default="transformer",
                    choices=["transformer", "vae"],
                    help="which sub-model to walk")
    ap.add_argument("--out", required=True, help="jsonl of module records")
    ap.add_argument("--final-latent-out", default=None)
    a = ap.parse_args()

    from diffusers import AutoencoderKLWan, WanPipeline
    vae = AutoencoderKLWan.from_pretrained(a.snapshot, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(a.snapshot, vae=vae, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    target = getattr(pipe, a.target)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")
    state = {"call": 0, "n": 0}

    def summarise(t):
        if not isinstance(t, torch.Tensor):
            return None
        flat = t.detach().reshape(-1)
        try:
            l2 = float(torch.linalg.vector_norm(flat, dtype=torch.float32).item())
        except Exception:
            l2 = float(flat[:4096].float().norm().item())
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "l2_norm": l2,
                "head10": [float(v) for v in flat[:10].float().cpu().tolist()]}

    def hook(name, mod):
        def fn(_m, _inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = t[0] if t else None
            s = summarise(t)
            if s is None:
                return
            s.update({"call": state["call"], "module": name,
                      "class": type(mod).__name__})
            fh.write(json.dumps(s) + "\n")
            state["n"] += 1
        return fn

    handles = [m.register_forward_hook(hook(n, m))
               for n, m in target.named_modules() if n]
    # the top-level call boundary, so records can be split per invocation
    def bump(_m, _i, _o):
        state["call"] += 1
    handles.append(target.register_forward_hook(bump))
    print(f"  hooked {len(handles)-1} named modules of {a.target}", flush=True)

    lat = torch.from_numpy(np.ascontiguousarray(np.load(a.latents)))
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    res = pipe(prompt=a.prompt, num_inference_steps=a.steps,
               guidance_scale=a.guidance, num_frames=a.frames,
               height=a.height, width=a.width, generator=g,
               latents=lat.to("cuda", torch.float16), output_type="latent")
    for h in handles:
        h.remove()
    fh.close()

    if a.final_latent_out:
        z = res.frames if hasattr(res, "frames") else res[0]
        z = z[0] if isinstance(z, (list, tuple)) else z
        np.save(a.final_latent_out, z.detach().float().cpu().numpy())

    print(f"  {state['n']} module record(s) over {state['call']} invocation(s) "
          f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
