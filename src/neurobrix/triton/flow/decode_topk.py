"""NBX_DECODE_TOPK=<jsonl>: per-step top-4 logits and the top-2 margin of a decode — mirror of
core/flow/decode_topk.py for the Triton engine (R33: the NBXTensor crosses to numpy at the diagnostic boundary,
CPU glue). Same record, so the two engines' and the two runs' records pair by step."""
from __future__ import annotations

import json
import os

import numpy as np


def log_topk(logits, step: int) -> None:
    path = os.environ.get("NBX_DECODE_TOPK")
    if not path:
        return
    arr = np.asarray(logits.numpy() if hasattr(logits, "numpy") else logits).reshape(-1).astype(np.float32)
    idx = np.argpartition(arr, -4)[-4:]
    order = idx[np.argsort(-arr[idx])]
    vals = [float(arr[i]) for i in order]
    rec = {"engine": "triton", "step": int(step), "ids": [int(i) for i in order],
           "vals": [round(v, 6) for v in vals], "margin12": round(vals[0] - vals[1], 6)}
    with open(path, "a") as f:
        json.dump(rec, f)
        f.write("\n")
