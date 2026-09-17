#!/usr/bin/env python
"""The CUDA door of a stack change (owner, 2026-09-16): a torch wheel is
accepted on this rack only if its embedded CUDA still carries sm_70 (CUDA 13
dropped Volta) AND it sees every card the driver sees, each with its
compute capability. Refuses by name, before anything is installed into the
engine's environment; run it with the CANDIDATE interpreter:

    /path/to/candidate/bin/python tools/stack_door.py [--expect-cards 4] [--json]

Exit 0 = the wheel may serve this rack; 1 = refused, the reason printed.
This is a door, not a note: nothing downstream runs past a refusal.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def driver_cards() -> list:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            out.append({"index": int(parts[0]), "name": parts[1], "compute_cap": parts[2]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-cards", type=int, default=None, help="refuse unless torch sees exactly this many cards (default: what the driver reports)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rec = {"python": sys.version.split()[0], "interpreter": sys.executable}
    refusals = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        rec["torch"] = None
        refusals.append(f"torch does not import in {sys.executable}: {type(exc).__name__}: {exc}")
        return finish(rec, refusals, a.json)
    rec["torch"] = torch.__version__
    rec["cuda_runtime"] = torch.version.cuda
    rec["arch_list"] = list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []
    try:
        import triton
        rec["triton"] = triton.__version__
    except Exception:
        rec["triton"] = None
    cards = driver_cards()
    rec["driver_cards"] = cards
    want = a.expect_cards if a.expect_cards is not None else len(cards)
    if not torch.cuda.is_available():
        refusals.append("torch.cuda.is_available() is False: this wheel sees no card of this rack")
        return finish(rec, refusals, a.json)
    seen = []
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        seen.append({"index": i, "name": torch.cuda.get_device_name(i), "compute_cap": f"{cap[0]}.{cap[1]}"})
    rec["torch_cards"] = seen
    if len(seen) != want:
        refusals.append(f"torch sees {len(seen)} card(s), the rack has {want}")
    for c in seen:
        sm = "sm_" + c["compute_cap"].replace(".", "")
        if sm not in rec["arch_list"]:
            refusals.append(f"card {c['index']} ({c['name']}, {sm}) is not in the wheel's embedded CUDA arch list {rec['arch_list']}: "
                            f"a kernel would JIT from PTX or fail — CUDA 13 dropped Volta; this wheel does not serve this rack")
    drv = {c["compute_cap"] for c in cards}
    if drv and not drv <= {c["compute_cap"] for c in seen}:
        refusals.append(f"the driver reports capabilities {sorted(drv)} that torch does not see")
    # The arch list says what the wheel's CUDA was built for; it says nothing
    # about the cuDNN and cuBLAS it bundles (a cuDNN that dropped Volta fails
    # at the first convolution, on the card, at run time). So the door RUNS
    # one convolution and one matmul on every card and reads the answer
    # against the CPU's — the second half of the door (owner, 2026-09-16).
    rec["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    rec["library_runs"] = []
    for c in seen:
        i = c["index"]
        row = {"index": i}
        try:
            x = torch.randn(2, 8, 32, 32); w = torch.randn(16, 8, 3, 3)
            ref = torch.nn.functional.conv2d(x, w, padding=1)
            got = torch.nn.functional.conv2d(x.cuda(i), w.cuda(i), padding=1).cpu()
            row["conv2d_max_abs_diff"] = float((ref - got).abs().max())
            a_ = torch.randn(64, 96); b_ = torch.randn(96, 48)
            got2 = (a_.cuda(i).half() @ b_.cuda(i).half()).float().cpu()
            row["mm_fp16_max_abs_diff"] = float((a_ @ b_ - got2).abs().max())
            torch.cuda.synchronize(i)
            if row["conv2d_max_abs_diff"] > 1e-2 or row["mm_fp16_max_abs_diff"] > 1.0:
                refusals.append(f"card {i}: a library result disagrees with the CPU's ({row})")
        except Exception as exc:  # noqa: BLE001 — the refusal names the exception
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            refusals.append(f"card {i} ({c['name']}): the bundled libraries do not run a convolution or a matmul on it — {row['error']}")
        rec["library_runs"].append(row)
    return finish(rec, refusals, a.json)


def finish(rec: dict, refusals: list, as_json: bool) -> int:
    rec["refused"] = refusals
    if as_json:
        print(json.dumps(rec, indent=1))
    else:
        print(f"[stack door] python {rec['python']} torch {rec.get('torch')} cuda {rec.get('cuda_runtime')} cudnn {rec.get('cudnn')} "
              f"triton {rec.get('triton')} archs {rec.get('arch_list')}")
        for row in rec.get("library_runs", []):
            print(f"   card {row['index']}: conv2d |diff| {row.get('conv2d_max_abs_diff')} · mm fp16 |diff| {row.get('mm_fp16_max_abs_diff')}"
                  + (f" · {row['error']}" if row.get("error") else ""))
        for c in rec.get("torch_cards", []):
            print(f"   card {c['index']}: {c['name']} sm_{c['compute_cap'].replace('.', '')}")
        for r in refusals:
            print(f"REFUSED: {r}")
        if not refusals:
            print("[stack door] ACCEPTED: every card of the rack is served by this wheel's embedded CUDA")
    return 1 if refusals else 0


if __name__ == "__main__":
    sys.exit(main())
