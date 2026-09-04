#!/usr/bin/env python3
"""Which step does a resumed render first diverge at?

`D-RENDER-RESUME-NOT-BIT-IDENTICAL`: a render resumed from its checkpoint
produces an image differing from the uninterrupted run in 99.7 % of pixels.
Four causes were eliminated by measurement (the save is byte-identical to not
saving; the reference is bit-reproducible; the latent dtype was a real bug and
fixing it changed nothing; `sigmas` placement was mine and was reverted). The
registered next probe is this one, and it is the op-by-op differential this
project uses for every numerical chantier: **dump the latent per step in both
runs and name the first step that differs.**

A whole-image diff says "wrong". The first differing STEP says which component
owns it — if step N is the first divergence, whatever ran between the resume
point and step N is the suspect, and everything before it is exonerated.

    # uninterrupted reference
    python3 tools/render_resume_stepdiff.py --model PixArt-Sigma-XL-1024 \\
        --steps 40 --tag ref
    # resumed (kill the first, then re-run with --resume)
    python3 tools/render_resume_stepdiff.py ... --tag resumed --resume
    python3 tools/render_resume_stepdiff.py --compare ref resumed

R33-neutral: hashes the latent through numpy at the flow-handler boundary, so
it works on both engines without importing torch here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "validation_outputs" / "render_resume_stepdiff"


def _install(tag: str):
    """Hash the loop state after every step, keyed by step index."""
    import neurobrix.core.flow.iterative_process as ip

    records: list[tuple[int, str, str]] = []
    original = ip.IterativeProcessHandler._checkpoint_save

    def traced(self, ck, step_idx, num_steps, state_key, driver):
        try:
            cur = self.ctx.variable_resolver.get(state_key)
            arr = cur.detach().cpu().numpy() if hasattr(cur, "detach") else \
                (cur.numpy() if hasattr(cur, "numpy") else None)
            if arr is not None:
                records.append((int(step_idx),
                                hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                                f"{arr.dtype}/{arr.shape}"))
        except Exception:                                     # noqa: BLE001
            pass
        return original(self, ck, step_idx, num_steps, state_key, driver)

    ip.IterativeProcessHandler._checkpoint_save = traced
    return records


def compare(a: str, b: str) -> int:
    fa, fb = OUT / f"{a}.json", OUT / f"{b}.json"
    if not (fa.exists() and fb.exists()):
        print(f"missing {fa if not fa.exists() else fb}", file=sys.stderr)
        return 2
    ra = {int(k): v for k, v in json.loads(fa.read_text()).items()}
    rb = {int(k): v for k, v in json.loads(fb.read_text()).items()}
    shared = sorted(set(ra) & set(rb))
    if not shared:
        print("no overlapping steps — the resumed run starts after the "
              "reference's last recorded step; lower the save interval")
        return 1
    print(f"  steps compared: {len(shared)}  "
          f"(ref {min(ra)}..{max(ra)}, resumed {min(rb)}..{max(rb)})")
    first = None
    for st in shared:
        same = ra[st][0] == rb[st][0]
        if not same and first is None:
            first = st
        if first is None or st <= (first + 2):
            print(f"    step {st:>3}  ref {ra[st][0]}  resumed {rb[st][0]}  "
                  f"{'same' if same else '<== DIFFERS'}")
    if first is None:
        print("\n  every shared step is IDENTICAL — divergence is after the "
              "loop (decode/VAE/writer), not inside it.")
        return 0
    print(f"\n  FIRST DIVERGENCE AT STEP {first}.")
    print("  Everything before it is exonerated; whatever the resume restores "
          "and the loop touches between the resume point and this step owns it.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--model")
    ap.add_argument("--prompt", default="a red apple on a wooden table")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not (args.model and args.tag):
        ap.error("--model and --tag are required unless --compare is used")

    OUT.mkdir(parents=True, exist_ok=True)
    records = _install(args.tag)

    sys.argv = ["neurobrix", "run", "--model", args.model,
                "--prompt", args.prompt, "--steps", str(args.steps),
                "--seed", str(args.seed),
                "--output", str(OUT / f"{args.tag}.png")]
    from neurobrix.cli import main as cli_main
    try:
        cli_main()
    except SystemExit:
        pass
    (OUT / f"{args.tag}.json").write_text(json.dumps(
        {str(s): (h, meta) for s, h, meta in records}, indent=1))
    print(f"\n  {len(records)} step hashes -> {OUT / (args.tag + '.json')}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.exit(main())
