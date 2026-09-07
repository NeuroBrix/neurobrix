#!/usr/bin/env python3
"""The protocol's head_dim cell: every decoder model at a context length
exactly equal to its own head dimension.

Why this length and no other. PyTorch's SDPA math decomposition hands the
traced graph a K in (batch, heads, head_dim, seq), and the engine has to put
it back. Until 2026-09-07 five places decided whether to, by comparing
shapes — `k.shape[-2] == q.shape[-1] and k.shape[-1] != q.shape[-1]` — a test
that has no answer when the sequence length equals the head dimension. At
exactly that length, and only there, every model read K transposed: the
residual stream was wrong from block 0 and the output was wrong with no
refusal anywhere. Measured on TinyLlama (head_dim 64): at 64 tokens the
last-position logits gave argmax 29892 at 6.41 where float64 says 3864 at
22.36 — |delta| 23.27 across the vocabulary — while 63 and 65 tokens were
exact to 0.02.

A length that only one arithmetic in the model can reach is not something a
campaign finds by accident: 60, 61, 62, 63, 65, 66, 70 and 72 tokens all
passed. So the length is COMPUTED from each model's own config rather than
chosen, and this cell stays in the protocol.

What it measures, per model:

  * the four arms — ATen `--compiled` and `--sequential`, Triton `--triton`
    and `--triton-sequential` — on the SAME exact token ids, at a context
    length of exactly head_dim;
  * every arm against the **sequential oracle** (ATen `--sequential`, the
    op-by-op reference path): byte identity or the first differing character;
  * for llama-like decoders, additionally against `tools/llama_fp64_oracle.py`
    — the same architecture in float64 numpy, no engine code in the path —
    on the first generated token.

A model whose output changes at this length between two source trees was
wrong at it. Run the cell twice, `--label before` and `--label after`, and
`table` renders the two side by side.

    python tools/head_dim_length_cell.py run --out DIR --label after
    python tools/head_dim_length_cell.py run --out DIR --label before --src /path/to/tree
    python tools/head_dim_length_cell.py table --out DIR

It writes `cell_<label>.json` and `CELL.md`, and refuses to conclude without
them: a number nobody can read in its dated file is not a number.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.path.expanduser("~")) / ".neurobrix" / "cache"

ARMS = ("compiled", "sequential", "triton", "triton-sequential")
ORACLE_ARM = "sequential"          # ATen op-by-op: the reference path

#: Architectures the float64 oracle in tools/llama_fp64_oracle.py reproduces.
#: Named rather than guessed — the oracle implements RMSNorm + GQA + RoPE +
#: SwiGLU exactly, and nothing else.
LLAMA_LIKE = {"llama", "tinyllama", "mistral", "qwen2", "qwen3"}


# ---------------------------------------------------------------------------
# What the model says about itself
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def decoder_of(model_dir: Path) -> dict | None:
    """The model's own answer to "are you a decoder, and how wide is a head".

    Read from the topology's generation section and the named component's
    profile — never from the model's name, never from a table here.
    """
    topo = _read_json(model_dir / "topology.json")
    if not topo:
        return None
    flow = topo.get("flow") or {}
    if flow.get("type") != "autoregressive_generation":
        return None
    gen = flow.get("generation") or {}
    lm = gen.get("lm_component")
    if not lm:
        return None
    profile = _read_json(model_dir / "components" / lm / "profile.json")
    if not profile:
        return None
    config = profile.get("config") or {}
    heads = profile.get("num_heads") or config.get("num_attention_heads")
    hidden = config.get("hidden_size") or profile.get("hidden_size")
    head_dim = profile.get("head_dim") or config.get("head_dim")
    if not head_dim:
        if not (heads and hidden):
            return None
        if hidden % heads:
            return None
        head_dim = hidden // heads
    vocab = config.get("vocab_size") or profile.get("vocab_size")
    if not vocab:
        return None
    return {
        "lm_component": lm,
        "head_component": gen.get("head_component"),
        "model_type": (topo.get("model_type") or "").lower(),
        "head_dim": int(head_dim),
        "num_heads": int(heads) if heads else None,
        "num_kv_heads": profile.get("num_kv_heads"),
        "hidden_size": hidden,
        "vocab_size": int(vocab),
        "num_layers": profile.get("num_layers") or config.get("num_layers"),
    }


def ids_for(head_dim: int, vocab: int) -> list:
    """`head_dim` token ids, deterministic and free of special tokens.

    The cell is about a LENGTH, not about a sentence: what the tokens mean
    does not enter any comparison here, because every arm and the float64
    oracle are given the same ones. They are generated rather than tokenized
    so that the length is exact on every model and every machine — the number
    of tokens a sentence becomes is the tokenizer's to decide, and it differs
    per model.

    Ids are drawn from [1000, vocab - 1000) with Knuth's multiplicative
    constant, which keeps them clear of the special-token bands at both ends.
    """
    lo, hi = 1000, max(2000, vocab - 1000)
    span = hi - lo
    return [lo + (i * 2654435761) % span for i in range(head_dim)]


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

def run_arm(model: str, ids: list, arm: str, max_tokens: int,
            outdir: Path, tag: str, src: Path | None, timeout: int) -> dict:
    out_path = outdir / f"out_{tag}.txt"
    out_path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str((src or REPO) / "src")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", model,
           "--prompt", "x",                       # unused: ids win (Priority 0)
           "--set", f"global.input_token_ids={json.dumps(ids)}",
           "--max-tokens", str(max_tokens),
           "--temperature", "0",
           "--output", str(out_path), f"--{arm}"]
    started = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout, cwd=str(src or REPO))
        rc, tail = proc.returncode, (proc.stderr or proc.stdout)[-600:]
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"timeout after {timeout}s"
    wall = round(time.time() - started, 3)
    text = out_path.read_text() if out_path.exists() else ""
    return {"arm": arm, "rc": rc, "wall_s": wall, "chars": len(text),
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text": text, "out_file": f"out_{tag}.txt",
            "error_tail": "" if rc == 0 else tail}


def run_fp64_oracle(model_dir: Path, ids: list, outdir: Path, tag: str,
                    src: Path | None) -> dict | None:
    """The float64 reference, for the architectures it reproduces exactly."""
    tool = (src or REPO) / "tools" / "llama_fp64_oracle.py"
    if not tool.exists():
        return None
    import numpy as np
    ids_path = outdir / f"ids_{tag}.npy"
    np.save(ids_path, np.asarray([ids], dtype=np.int64))
    out_json = outdir / f"oracle_{tag}.json"
    proc = subprocess.run(
        [sys.executable, str(tool), "--model", str(model_dir),
         "--ids", str(ids_path), "--out", str(out_json), "--top", "3"],
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not out_json.exists():
        return {"ran": False, "why": (proc.stderr or proc.stdout)[-400:]}
    doc = json.loads(out_json.read_text())
    return {"ran": True, "argmax": doc["argmax"],
            "top1": doc["top"][0]["logit"],
            "margin": doc["margin_top1_top2"],
            "file": out_json.name}


def first_difference(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------

def run(args) -> int:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    per_model_root = outdir / args.label
    per_model_root.mkdir(parents=True, exist_ok=True)

    wanted = [m.strip() for m in args.models.split(",")] if args.models else None
    models = sorted(p.name for p in CACHE.iterdir() if p.is_dir()) if CACHE.is_dir() else []
    if wanted:
        models = [m for m in models if m in wanted]

    rows, skipped = [], []
    for model in models:
        model_dir = CACHE / model
        info = decoder_of(model_dir)
        if info is None:
            skipped.append({"model": model, "why": "no autoregressive decoder in its topology"})
            continue
        head_dim = info["head_dim"]
        ids = ids_for(head_dim, info["vocab_size"])
        here = per_model_root / model
        here.mkdir(parents=True, exist_ok=True)
        print(f"[{args.label}] {model}: head_dim {head_dim} "
              f"({info['model_type'] or 'unknown type'})", flush=True)

        arms = {}
        for arm in (args.arm or list(ARMS)):
            tag = f"{model}_{arm}"
            record = run_arm(model, ids, arm, args.max_tokens, here, tag,
                             args.src_path, args.timeout)
            arms[arm] = record
            print(f"    {arm:20s} rc={record['rc']} sha={record['sha256'][:8]} "
                  f"{record['wall_s']}s", flush=True)

        # The premise of the whole cell is that the engine ran on THESE ids at
        # THIS length. A runtime variable that were silently ignored would
        # leave the arms measuring the "x" prompt instead, at a length nobody
        # chose, and every row would look green. So it is proved rather than
        # assumed: one control run on the fastest arm with a single id
        # changed. If the output does not move, the ids were not used.
        control = None
        if args.control:
            probe_arm = (args.arm or list(ARMS))[0]
            moved = list(ids)
            moved[-1] = ids[0] if ids[-1] != ids[0] else ids[1]
            control = run_arm(model, moved, probe_arm, args.max_tokens, here,
                              f"{model}_control", args.src_path, args.timeout)
            base = arms.get(probe_arm)
            control["distinguishes"] = bool(
                base and base["rc"] == 0 == control["rc"]
                and base["sha256"] != control["sha256"])
            print(f"    control (one id moved) sha={control['sha256'][:8]} "
                  f"{'ids are used' if control['distinguishes'] else 'IDS IGNORED'}",
                  flush=True)
            if not control["distinguishes"] and base and base["rc"] == 0:
                raise RuntimeError(
                    f"{model}: changing a token id did not change the output, "
                    f"so `global.input_token_ids` was not what the engine ran "
                    f"on. Refusing to report a length the run did not have.")

        oracle = None
        if info["model_type"] in LLAMA_LIKE:
            oracle = run_fp64_oracle(model_dir, ids, here, model, args.src_path)
            if oracle and oracle.get("ran"):
                print(f"    float64 oracle       argmax={oracle['argmax']} "
                      f"top1={oracle['top1']:.4f}", flush=True)

        ref = arms.get(ORACLE_ARM)
        verdict = {}
        for arm, record in arms.items():
            if record["rc"] != 0:
                verdict[arm] = "refused"
            elif ref is None or ref["rc"] != 0:
                verdict[arm] = "no oracle arm"
            elif record["sha256"] == ref["sha256"]:
                verdict[arm] = "identical"
            else:
                verdict[arm] = f"differs at char {first_difference(record['text'], ref['text'])}"

        rows.append({"model": model, **info, "ids_len": len(ids),
                     "ids_head": ids[:6], "ids_tail": ids[-3:],
                     "arms": arms, "verdict": verdict, "fp64_oracle": oracle,
                     "control": control})

    document = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/head_dim_length_cell.py",
        "cell": "context length == head_dim",
        "label": args.label,
        "source_tree": str(args.src_path or REPO),
        "source_rev": _rev(args.src_path or REPO),
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "max_tokens": args.max_tokens,
        "oracle_arm": ORACLE_ARM,
        "rows": rows,
        "skipped": skipped,
    }
    doc_path = outdir / f"cell_{args.label}.json"
    doc_path.write_text(json.dumps(document, indent=1))
    if not doc_path.exists():
        raise RuntimeError(f"refusing to conclude: {doc_path} was not written")
    print(f"\nwritten: {doc_path}")
    return table(args)


def _rev(tree: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "?"
    except Exception:
        return "?"


def table(args) -> int:
    outdir = Path(args.out)
    docs = {}
    for path in sorted(outdir.glob("cell_*.json")):
        docs[path.stem[len("cell_"):]] = json.loads(path.read_text())
    if not docs:
        raise RuntimeError(f"refusing to render a table: no cell_*.json under {outdir}")

    lines = ["# The head_dim cell — every decoder at a context length equal to its head dimension",
             ""]
    any_doc = next(iter(docs.values()))
    lines += [f"Generated **{any_doc['generated']}** by `tools/head_dim_length_cell.py` "
              f"on {any_doc['machine']}.", "",
              "The length is computed from each model's own config, not chosen: it is the one "
              "length at which a K in (b, h, head_dim, seq) and a K in (b, h, seq, head_dim) "
              "have the same shape, so an engine that reads the layout off the shape has no "
              "answer there — and gave a wrong one, in silence, on every model.", "",
              f"Every arm is compared against the **sequential oracle** (`--{ORACLE_ARM}`, the "
              f"ATen op-by-op path) on the same exact token ids; llama-like decoders are also "
              f"compared against `tools/llama_fp64_oracle.py`.", ""]

    for label, doc in docs.items():
        lines += [f"## `{label}` — source tree `{doc['source_rev']}`", "",
                  "| model | head_dim | " + " | ".join(f"`--{a}`" for a in ARMS)
                  + " | float64 oracle |",
                  "|---|---:|" + "---|" * (len(ARMS) + 1)]
        for row in doc["rows"]:
            cells = []
            for arm in ARMS:
                rec = row["arms"].get(arm)
                if rec is None:
                    cells.append("—")
                elif rec["rc"] != 0:
                    cells.append("**refused**")
                else:
                    cells.append(f"`{rec['sha256'][:8]}` {row['verdict'][arm]}")
            orc = row.get("fp64_oracle")
            if not orc:
                oracle_cell = "n/a"
            elif not orc.get("ran"):
                oracle_cell = "refused"
            else:
                oracle_cell = f"argmax {orc['argmax']} @ {orc['top1']:.3f}"
            lines.append(f"| {row['model']} | {row['head_dim']} | "
                         + " | ".join(cells) + f" | {oracle_cell} |")
        if doc["skipped"]:
            lines += ["", "Not a decoder, and said so: "
                      + ", ".join(f"`{s['model']}`" for s in doc["skipped"]), ""]
        lines.append("")

    if len(docs) > 1:
        lines += ["## What changed between the trees", ""]
        labels = list(docs)
        base, *rest = labels
        for other in rest:
            lines += [f"### `{base}` against `{other}`", "",
                      "| model | arm | " + f"{base} | {other} | verdict |",
                      "|---|---|---|---|---|"]
            by_model = {r["model"]: r for r in docs[other]["rows"]}
            for row in docs[base]["rows"]:
                twin = by_model.get(row["model"])
                if twin is None:
                    continue
                for arm in ARMS:
                    a = row["arms"].get(arm)
                    b = twin["arms"].get(arm)
                    if a is None or b is None:
                        continue
                    same = a["sha256"] == b["sha256"]
                    lines.append(
                        f"| {row['model']} | `--{arm}` | `{a['sha256'][:8]}` | "
                        f"`{b['sha256'][:8]}` | "
                        + ("unchanged" if same else "**changed — it was wrong at this length**")
                        + " |")
            lines.append("")

    md = outdir / "CELL.md"
    md.write_text("\n".join(lines) + "\n")
    if not md.exists():
        raise RuntimeError(f"refusing to conclude: {md} was not written")
    print(f"written: {md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="after")
    r.add_argument("--models", default=None, help="comma-separated subset")
    r.add_argument("--arm", action="append", default=None,
                   help=f"repeatable; default {', '.join(ARMS)}")
    r.add_argument("--max-tokens", type=int, default=8)
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--no-control", dest="control", action="store_false",
                   help="skip the run that proves the ids were the ones used "
                        "(they are proved by default, and the cell refuses "
                        "when they were not)")
    r.add_argument("--src", default=None,
                   help="a frozen source tree to measure instead of this one")
    r.set_defaults(func=run)

    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.set_defaults(func=table)

    args = ap.parse_args()
    args.src_path = Path(args.src).resolve() if getattr(args, "src", None) else None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
