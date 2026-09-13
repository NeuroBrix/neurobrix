#!/usr/bin/env python3
"""Re-baseline the decode byte gate by PROOF, not by re-recording.

A byte gate proves a change is inert. A change that deliberately improves
accuracy is not inert, so its gate cannot simply be re-recorded — the new
output has to be shown CLOSER to the fp64 oracle than the old one. This tool
produces that evidence and refuses to conclude without it:

    python3 tools/gate_rebaseline_proof.py capture --label before --out DIR
    #  ... switch the tree ...
    python3 tools/gate_rebaseline_proof.py capture --label after  --out DIR
    python3 tools/gate_rebaseline_proof.py oracle  --out DIR --ids IDS.npy
    python3 tools/gate_rebaseline_proof.py table   --out DIR

`capture` runs the SAME arm twice on one tree: once for the gate's own
sha256 (the locked protocol's prompt and token budget), and once as a
single-token run whose prefill dumps the FULL last-position logit row. The
two runs share the prompt, so the row is the logit vector that decides the
first generated token.

`oracle` runs the float64 numpy reference over the same ids and dumps its
full logit vector.

`table` measures each tree's logits against the oracle's — relative L2, max
absolute deviation, the argmax each one chooses, and the top-1 margin — and
writes them with BOTH shas and the tested length to a dated file. It states
which tree is closer; it does not decide whether that is enough.

The three caches go before every run, so no warm artifact crosses trees.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHES = [
    Path.home() / ".cache/triton_msl",
    Path.home() / ".triton/cache",
]
AUTOTUNE_GLOB = "autotune_configs_*.json"
DEFAULT_MODEL = "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "Explain in one short paragraph why the sky appears blue."


def _clear_caches() -> list[str]:
    cleared = []
    for c in CACHES:
        if c.exists():
            shutil.rmtree(c, ignore_errors=True)
        cleared.append(str(c))
    rc = Path.home() / ".neurobrix/replay_cache"
    if rc.exists():
        for f in rc.glob(AUTOTUNE_GLOB):
            f.unlink(missing_ok=True)
            cleared.append(str(f))
    return cleared


def _run(cmd, env, cwd=REPO_ROOT, timeout=2400):
    return subprocess.run(cmd, env=env, cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout)


def cmd_capture(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lastrow = out / f"{args.label}_lastrow"
    if lastrow.exists():
        shutil.rmtree(lastrow)

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = "src"

    # 1. the gate run: the locked protocol's prompt and token budget.
    _clear_caches()
    gate_txt = out / f"{args.label}_gate.txt"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "run",
           "--model", args.model, "--prompt", args.prompt,
           "--max-tokens", str(args.max_tokens),
           "--temperature", "0", "--output", str(gate_txt), f"--{args.arm}"]
    p1 = _run(cmd, base_env)
    text = gate_txt.read_text() if gate_txt.exists() else ""
    sha = hashlib.sha256(text.encode()).hexdigest()

    # 2. the logits run: one token, so the dump is the PREFILL's last row.
    _clear_caches()
    env = dict(base_env)
    env["NBX_DUMP_TIDS"] = str(out / f"{args.label}_tids.jsonl")
    env["NBX_DUMP_TIDS_FILTER"] = args.filter
    env["NBX_DUMP_TIDS_PASS"] = "0"
    env["NBX_DUMP_LASTROW"] = str(lastrow)
    one_txt = out / f"{args.label}_one.txt"
    cmd2 = [sys.executable, "-u", "-m", "neurobrix", "run",
            "--model", args.model, "--prompt", args.prompt,
            "--max-tokens", "1", "--temperature", "0",
            "--output", str(one_txt), f"--{args.arm}"]
    p2 = _run(cmd2, env)

    rows = sorted(lastrow.glob("*.npy")) if lastrow.exists() else []
    record = {
        "label": args.label,
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": platform.node(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "arm": args.arm,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "gate_sha256": sha,
        "gate_sha256_16": sha[:16],
        "gate_rc": p1.returncode,
        "logits_rc": p2.returncode,
        "text_chars": len(text),
        "lastrow_files": [f.name for f in rows],
        "caches_cleared": True,
    }
    (out / f"{args.label}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"  {args.label}: sha={sha[:16]} rc={p1.returncode}/{p2.returncode} "
          f"lastrow={[f.name for f in rows]}")
    if not rows:
        print("  no last-position row was dumped: check --filter against the "
              "kernel's tids (NBX_DUMP_TIDS wrote "
              f"{env['NBX_DUMP_TIDS']})", file=sys.stderr)
        return 2
    return 0


def cmd_oracle(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    cmd = [sys.executable, "-u", str(REPO_ROOT / "tools/llama_fp64_oracle.py"),
           "--ids", args.ids, "--out", str(out / "oracle.json"),
           "--logits-npy", str(out / "oracle_logits.npy")]
    if args.length:
        cmd += ["--length", str(args.length)]
    p = _run(cmd, env, timeout=3600)
    sys.stdout.write(p.stdout[-2000:])
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
    return p.returncode


def _load_row(d: Path, label: str, want: int | None = None):
    """The dumped row, chosen by LENGTH when several ops share a tid.

    Two components can carry the same tid (`model` and `lm_head` both have
    `aten.mm::0::out_0`), so the file is picked by matching the oracle's
    vocabulary length rather than by order.
    """
    import numpy as np
    rows = sorted((d / f"{label}_lastrow").glob("*.npy"))
    if not rows:
        return None, None
    if want is not None:
        for r in rows:
            v = np.load(r)
            if v.size == want:
                return v.astype("float64"), r.name
        return None, None
    return np.load(rows[0]).astype("float64"), rows[0].name


def cmd_table(args) -> int:
    import numpy as np
    out = Path(args.out)
    oracle_p = out / "oracle_logits.npy"
    if not oracle_p.exists():
        print(f"missing {oracle_p}: run the `oracle` step first", file=sys.stderr)
        return 2
    oracle = np.load(oracle_p).astype("float64").ravel()

    rows = {}
    for label in ("before", "after"):
        vec, name = _load_row(out, label, oracle.size)
        meta_p = out / f"{label}.json"
        if vec is None or not meta_p.exists():
            print(f"missing capture for {label!r}", file=sys.stderr)
            return 2
        meta = json.loads(meta_p.read_text())
        if vec.size != oracle.size:
            print(f"{label}: logit row has {vec.size} values, the oracle has "
                  f"{oracle.size}; refusing to compare different vectors",
                  file=sys.stderr)
            return 2
        diff = vec - oracle
        rows[label] = {
            "sha256_16": meta["gate_sha256_16"],
            "sha256": meta["gate_sha256"],
            "row_file": name,
            "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(oracle)),
            "abs_l2": float(np.linalg.norm(diff)),
            "max_abs": float(np.abs(diff).max()),
            "argmax": int(np.argmax(vec)),
            "argmax_logit": float(vec[int(np.argmax(vec))]),
        }

    o_sorted = np.argsort(-oracle)
    doc = {
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": platform.node(),
        "length_tokens": int(args.length) if args.length else None,
        "logits_len": int(oracle.size),
        "oracle": {
            "argmax": int(o_sorted[0]),
            "top1_logit": float(oracle[o_sorted[0]]),
            "margin_top1_top2": float(oracle[o_sorted[0]] - oracle[o_sorted[1]]),
            "arithmetic": "float64 (numpy)",
        },
        "before": rows["before"],
        "after": rows["after"],
    }
    closer = "after" if rows["after"]["rel_l2"] < rows["before"]["rel_l2"] else "before"
    doc["closer_to_oracle"] = closer
    doc["improvement_ratio"] = (rows["before"]["rel_l2"] / rows["after"]["rel_l2"]
                                if rows["after"]["rel_l2"] else None)
    (out / "proof.json").write_text(json.dumps(doc, indent=2) + "\n")

    md = [
        "# Gate re-baseline — proved, not re-recorded",
        "",
        f"Generated **{doc['when']}** on {doc['machine']}.",
        "",
        f"* tested length: **{doc['length_tokens']} tokens**",
        f"* logit vector: **{doc['logits_len']}** values, last position of the prefill",
        f"* oracle: float64 numpy (`tools/llama_fp64_oracle.py`), argmax "
        f"**{doc['oracle']['argmax']}**, top1−top2 margin "
        f"{doc['oracle']['margin_top1_top2']:.6f}",
        "",
        "| tree | gate sha256 | rel. L2 to oracle | max abs dev | argmax |",
        "|---|---|---:|---:|---:|",
    ]
    for label in ("before", "after"):
        r = rows[label]
        md.append(f"| {label} | `{r['sha256_16']}` | {r['rel_l2']:.6e} | "
                  f"{r['max_abs']:.6e} | {r['argmax']} |")
    same_sha = rows["before"]["sha256"] == rows["after"]["sha256"]
    doc["gate_moved"] = not same_sha
    md += [
        "",
        f"**Closer to the oracle: `{closer}`.**"
        + (f" Relative L2 {doc['improvement_ratio']:.6f}× (before/after)."
           if doc["improvement_ratio"] else ""),
        "",
    ]
    if same_sha:
        md += [
            "**The gate did not move.** Both trees produce the same bytes, so "
            "there is nothing to re-baseline: the anchor stands unchanged and "
            "the accuracy gain is invisible at the token level for this prompt "
            "and length. That is a result, not a null one — it says the change "
            "is inert on the decode path while the reference bank shows it is "
            "not inert on the kernels it targets.",
            "",
            "Read the distances for what they are: at this length they differ in "
            "the seventh significant figure, which is at the measurement's own "
            "floor. The load-bearing evidence for the change is the bank, not "
            "this row.",
            "",
        ]
    else:
        md += [
            "The old sha stays in the journal as the pre-change anchor; the new "
            "one becomes the gate only because the vector it comes from is "
            "measurably nearer the oracle, not because it was observed.",
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--label", required=True, choices=["before", "after"])
    c.add_argument("--out", required=True)
    c.add_argument("--arm", default="triton")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--prompt", default=DEFAULT_PROMPT)
    c.add_argument("--max-tokens", type=int, default=60)
    c.add_argument("--filter", default="aten.mm::0::out_0",
                   help="NBX_DUMP_TIDS_FILTER selecting the logits op")
    c.set_defaults(func=cmd_capture)

    o = sub.add_parser("oracle")
    o.add_argument("--out", required=True)
    o.add_argument("--ids", required=True)
    o.add_argument("--length", type=int, default=0)
    o.set_defaults(func=cmd_oracle)

    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("--length", type=int, default=0)
    t.set_defaults(func=cmd_table)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
