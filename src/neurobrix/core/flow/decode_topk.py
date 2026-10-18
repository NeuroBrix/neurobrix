"""NBX_DECODE_TOPK=<jsonl>: per-step top-4 logits and the top-2 margin of a decode — the measuring tool of the
identity proofs (Hocine 18:05: a token that differs between two runs is noise when the margin at the first divergent
step is within the dtype's drift, a defect when it is not). One record per step, appended; default off. The VLM,
encoder-decoder and autoregressive flows carry their own copy of this record; the audio-LLM, TTS-LLM and dual-AR
flows write it through this module (compiled engine). Mirror: triton/flow/decode_topk.py (R33, numpy at the boundary).
"""
from __future__ import annotations

import json
import os

import torch


def log_topk(logits: torch.Tensor, step: int) -> None:
    path = os.environ.get("NBX_DECODE_TOPK")
    if not path:
        return
    flat = logits.reshape(-1).float()
    top = torch.topk(flat, 4)
    vals = [float(x) for x in top.values]
    rec = {"engine": "torch", "step": int(step), "ids": [int(i) for i in top.indices],
           "vals": [round(v, 6) for v in vals], "margin12": round(vals[0] - vals[1], 6)}
    with open(path, "a") as f:
        json.dump(rec, f)
        f.write("\n")
