#!/usr/bin/env python3
"""An fp64 Llama forward in numpy, for deciding which engine path is right.

Two engine strategies can disagree — a prefill of N tokens against a
prefill of N-1 plus a cached decode step — and the disagreement is only a
defect once something outside both says which side is wrong. This is that
something: the same architecture, evaluated in float64 from the weights on
disk, with no engine code in the path.

Torch is never imported. Weights arrive as bf16 bits in a uint16 container
and are widened by the exact shift (bf16 is the top 16 bits of an fp32).

    python tools/llama_fp64_oracle.py --ids ids.npy --out out.json --top 5
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np


def _read_safetensors_header(path: Path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data_start = 8 + n
    return header, data_start


class Shard:
    """bf16 safetensors, read lazily and widened to fp64 on demand."""

    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_start = _read_safetensors_header(path)
        self._fh = open(path, "rb")

    def get(self, key: str) -> np.ndarray:
        info = self.header[key]
        if info["dtype"] != "BF16":
            raise RuntimeError(f"{key}: expected BF16, found {info['dtype']}")
        start, end = info["data_offsets"]
        self._fh.seek(self.data_start + start)
        raw = self._fh.read(end - start)
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(info["shape"]).astype(np.float64)

    def has(self, key: str) -> bool:
        return key in self.header


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def rope_tables(seq: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seq, dtype=np.float64)
    freqs = np.outer(pos, inv)                      # [S, hd/2]
    emb = np.concatenate([freqs, freqs], axis=-1)   # [S, hd]
    return np.cos(emb), np.sin(emb)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def forward(model: Shard, head: Shard, ids: np.ndarray, cfg: dict,
            trace: list | None = None) -> np.ndarray:
    """Returns the logits of the LAST position, float64.

    `trace`, when given, collects the L2 norm of the residual stream after
    every attention and every MLP add — the coarse per-layer signature an
    engine dump can be lined up against to find the first layer that
    departs from float64."""
    S = ids.shape[0]
    H = cfg["num_heads"]
    KVH = cfg["num_kv_heads"]
    hd = cfg["hidden_size"] // H
    eps = cfg["rms_norm_eps"]

    x = model.get("token_embed.weight")[ids]                      # [S, D]
    cos, sin = rope_tables(S, hd, cfg["rope_theta"])
    causal = np.triu(np.full((S, S), -np.inf, dtype=np.float64), 1)

    for layer in range(cfg["num_layers"]):
        p = f"block.{layer}."
        h = rms_norm(x, model.get(p + "input_norm.weight"), eps)

        q = (h @ model.get(p + "attn.query.weight").T).reshape(S, H, hd)
        k = (h @ model.get(p + "attn.key.weight").T).reshape(S, KVH, hd)
        v = (h @ model.get(p + "attn.value.weight").T).reshape(S, KVH, hd)

        q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]

        group = H // KVH
        k = np.repeat(k, group, axis=1)                            # [S, H, hd]
        v = np.repeat(v, group, axis=1)

        qt = q.transpose(1, 0, 2)                                  # [H, S, hd]
        kt = k.transpose(1, 0, 2)
        vt = v.transpose(1, 0, 2)
        scores = qt @ kt.transpose(0, 2, 1) / np.sqrt(hd) + causal
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        attn = (w @ vt).transpose(1, 0, 2).reshape(S, -1)          # [S, D]
        x = x + attn @ model.get(p + "attn.out.weight").T
        if trace is not None:
            trace.append({"after": f"block.{layer}.attn", "l2": float(np.linalg.norm(x))})

        h = rms_norm(x, model.get(p + "post_attn_norm.weight"), eps)
        gate = h @ model.get(p + "ffn.gate.weight").T
        up = h @ model.get(p + "ffn.up.weight").T
        act = gate / (1.0 + np.exp(-gate)) * up                    # SiLU
        x = x + act @ model.get(p + "ffn.down.weight").T
        if trace is not None:
            trace.append({"after": f"block.{layer}.ffn", "l2": float(np.linalg.norm(x))})

    x = rms_norm(x, model.get("norm.weight"), eps)
    return x[-1] @ head.get("weight").T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/.neurobrix/cache/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--ids", required=True, help=".npy of token ids")
    ap.add_argument("--length", type=int, default=0,
                    help="truncate the ids to this many tokens (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.model)
    # The components are named by the model's own topology, not by this file:
    # a llama-like is not obliged to call its decoder "model".
    topo_path = root / "topology.json"
    lm_name, head_name = "model", "lm_head"
    if topo_path.exists():
        gen = (json.loads(topo_path.read_text()).get("flow") or {}).get("generation") or {}
        lm_name = gen.get("lm_component") or lm_name
        head_name = gen.get("head_component") or head_name
    profile = json.loads((root / "components" / lm_name / "profile.json").read_text())
    conf = profile.get("config", {})
    def _need(*names):
        for n in names:
            for src in (profile, conf):
                if src.get(n) is not None:
                    return src[n]
        raise RuntimeError(
            f"{root.name}: none of {names} is in the decoder's profile. This "
            f"oracle reproduces one architecture exactly and will not guess a "
            f"missing hyper-parameter.")
    cfg = {
        "hidden_size": _need("hidden_size"),
        "num_layers": _need("num_layers", "num_hidden_layers"),
        "num_heads": _need("num_heads", "num_attention_heads"),
        "num_kv_heads": _need("num_kv_heads", "num_key_value_heads"),
        "rope_theta": _need("rope_theta"),
        "rms_norm_eps": _need("rms_norm_eps"),
    }

    ids = np.load(args.ids).astype(np.int64).ravel()
    if args.length:
        ids = ids[:args.length]

    model = Shard(root / "components" / lm_name / "weights/shard_000.safetensors")
    head = Shard(root / "components" / head_name / "weights/shard_000.safetensors")
    trace: list = []
    logits = forward(model, head, ids, cfg, trace)

    order = np.argsort(-logits)[:args.top]
    document = {
        "model": str(root),
        "arithmetic": "float64 (numpy), weights widened from bf16 by the exact shift",
        "tokens": int(ids.shape[0]),
        "last_ids": [int(i) for i in ids[-6:]],
        "argmax": int(order[0]),
        "top": [{"id": int(i), "logit": float(logits[i])} for i in order],
        "margin_top1_top2": float(logits[order[0]] - logits[order[1]]),
        "residual_l2": trace,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1))
    print(json.dumps(document, indent=1))
    if not out.exists():
        raise RuntimeError(f"refusing to conclude: {out} was not written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
