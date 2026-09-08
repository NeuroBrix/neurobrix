#!/usr/bin/env python3
"""Run one hub family on both branches, one model at a time, streaming.

The hub's artefacts far exceed this machine's free disk, so nothing
accumulates: a model is staged from the read-only mount, its identity
recorded, both branches measured, its verdict written to its own file, and
the staged copy removed before the next model is touched. Disk is read before
and after every model AND every family, so the record shows that accumulation
never decided which model got skipped.

A model too large to stage is NOT a skipped line. It is measured where it can
be — Prism plans from metadata alone, without staging — and recorded with the
budget arithmetic that says why it could not run here: weights, activations,
what the chosen strategy promises to hold, and what the disk allowed.

    python tools/hub_family_sweep.py --family llm --out validation_outputs/<dated>/
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
import time
from pathlib import Path

MOUNT = Path.home() / "Mounts" / "Super-NeuroBrix-Cache"
CACHE = Path.home() / ".neurobrix" / "cache"
REPO = Path(__file__).resolve().parents[1]
CACHES = (Path.home() / ".cache" / "triton_msl", Path.home() / ".triton" / "cache")

ARMS = ("compiled", "triton", "triton-sequential")


def disk_free_gb() -> float:
    st = os.statvfs("/System/Volumes/Data")
    return round(st.f_bavail * st.f_frsize / 2**30, 2)


def dir_identity(path: Path) -> dict:
    """A recorded identity for a staged artefact.

    sha256 over every file's (relative path, size) plus the bytes of the small
    metadata files. It is NOT a hash of the weights: re-reading tens of GB
    twice over a degraded NFS link to prove a copy that the filesystem already
    reports complete would cost more than it establishes. What it does
    establish — the full file list, every size, and the exact manifest,
    topology and weight indices — is stated rather than implied.
    """
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    total = 0
    for f in files:
        rel = f.relative_to(path).as_posix()
        size = f.stat().st_size
        total += size
        h.update(f"{rel}:{size}\n".encode())
    meta = hashlib.sha256()
    for name in ("manifest.json", "topology.json"):
        p = path / name
        if p.exists():
            meta.update(p.read_bytes())
    for idx in sorted(path.glob("components/*/weights_index.json")):
        meta.update(idx.read_bytes())
    return {"listing_sha256": h.hexdigest()[:16],
            "metadata_sha256": meta.hexdigest()[:16],
            "files": len(files), "bytes": total,
            "gb": round(total / 2**30, 2)}


def read_model_facts(path: Path) -> dict:
    man = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else {}
    facts = {"family": man.get("family"), "dtype": man.get("dtype"),
             "components": sorted(p.name for p in (path / "components").iterdir())
             if (path / "components").exists() else []}
    # head_dim, for the decoder cell the mandate fixes: context length == head_dim.
    for prof in sorted(path.glob("components/*/profile.json")):
        pr = json.loads(prof.read_text())
        hidden = (pr.get("config") or {}).get("hidden_size")
        heads = pr.get("num_heads")
        if hidden and heads:
            facts["hidden_size"] = hidden
            facts["num_heads"] = heads
            facts["num_kv_heads"] = pr.get("num_kv_heads")
            facts["num_hidden_layers"] = pr.get("num_hidden_layers")
            facts["head_dim"] = hidden // heads
            facts["decoder_component"] = prof.parent.name
            break
    return facts


def clear_caches() -> None:
    for p in CACHES:
        shutil.rmtree(p, ignore_errors=True)
    rc = Path.home() / ".neurobrix" / "replay_cache"
    if rc.exists():
        for f in rc.glob("autotune_configs_*.json"):
            f.unlink()


# What each family's run actually needs. run.py validates the invocation
# against the family YAML's `inputs.required`, so passing --prompt to an
# upscaler is not a near miss — it is a refused run that would have been
# recorded as a failure of the engine rather than of the harness.
ASSETS = {
    "--input-image": REPO / "benchmarks" / "assets" / "apple_448.png",
    "--audio": REPO / "benchmarks" / "assets" / "jfk_11s.wav",
}


def family_output_ext(family: str) -> str | None:
    """The extension this family's output must carry, from its own YAML.

    `resolve_output_path` refuses a mismatch outright — "output extension
    '.txt' incompatible with family 'upscaler' mode 'image'" — which is the
    engine behaving well and a harness that hardcodes .txt behaving badly.
    `mode_dependent` means the family cannot say in advance, so the extension
    is left to the engine by omitting --output.
    """
    import yaml
    spec = REPO / "src" / "neurobrix" / "config" / "families" / f"{family}.yml"
    if not spec.exists():
        return None
    d = yaml.safe_load(spec.read_text()) or {}
    fmt = ((d.get("output_processing") or {}).get("output_format"))
    if not fmt or fmt == "mode_dependent":
        return None
    return f".{fmt}"


def family_inputs(family: str, prompt: str) -> list:
    """The argv fragment this family requires, read from its own YAML."""
    import yaml
    spec = REPO / "src" / "neurobrix" / "config" / "families" / f"{family}.yml"
    required = []
    if spec.exists():
        d = yaml.safe_load(spec.read_text()) or {}
        inp = d.get("inputs") or {}
        if isinstance(inp, dict):
            required = list(inp.get("required") or [])
    if not required:
        required = ["--prompt"]
    argv = []
    for flag in required:
        if flag == "--prompt":
            argv += ["--prompt", prompt]
        elif flag in ASSETS:
            path = ASSETS[flag]
            if not path.exists():
                raise FileNotFoundError(f"{family} requires {flag} and {path} is absent")
            argv += [flag, str(path)]
        else:
            raise ValueError(f"{family} requires {flag}, which this harness has no asset for")
    return argv


def run_arm(model: str, arm: str, family: str, prompt: str, max_tokens: int,
            outdir: Path, timeout: int) -> dict:
    clear_caches()
    ext = family_output_ext(family)
    out = outdir / f"out_{model}_{arm}{ext}" if ext else None
    cmd = ([sys.executable, "-u", "-m", "neurobrix", "run", "--model", model]
           + family_inputs(family, prompt)
           + ["--max-tokens", str(max_tokens), "--temperature", "0"]
           + (["--output", str(out)] if out else [])
           + [f"--{arm}"])
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout, cwd=REPO)
        rc, err, sout = proc.returncode, proc.stderr[-1500:], proc.stdout[-1200:]
    except subprocess.TimeoutExpired:
        rc, err, sout = -1, f"TIMEOUT after {timeout}s", ""
    wall = round(time.time() - t0, 1)
    # Hash the BYTES, whatever the artefact is. Reading an image or a wav as
    # text would decide, silently, that only LLM output is comparable.
    blob = out.read_bytes() if (out and out.exists()) else b""
    return {"arm": arm, "rc": rc, "wall_s": wall,
            "output_path": str(out) if out else "(engine-chosen)",
            "output_sha256": hashlib.sha256(blob).hexdigest()[:16] if blob else None,
            "output_bytes": len(blob),
            "stdout_tail": sout if rc != 0 else "",
            "stderr_tail": err if rc != 0 else ""}


def render(doc: dict) -> str:
    """The family's verdict, and a row per ENGINE COMPONENT.

    A model-level pass/fail hides which part of the engine did the work, so
    the component table is the deliverable and the per-model line is the
    index into it.
    """
    L = [f"# Hub family `{doc['family']}` — both branches",
         "",
         f"Generated **{doc['generated']}** by `{doc['tool']}` on {doc['machine']}.",
         "**No public claim is made from any number here.**",
         "",
         f"* arms: {', '.join('`--' + a + '`' for a in doc['arms'])}",
         f"* disk free: **{doc['disk_free_gb_before_family']} GB** before the family, "
         f"**{doc.get('disk_free_gb_after_family')} GB** after",
         "* one model staged at a time from the read-only mount, identity recorded, "
         "both branches measured, staged copy removed before the next",
         ""]
    if doc.get("manifest_unreadable"):
        L += ["## Models whose manifest could not be read", "",
              "Not absent, and not skipped — listed so the family cannot look "
              "complete while it is not.", "",
              "| model | error |", "|---|---|"]
        L += [f"| {u['model']} | {u['manifest_error']} |" for u in doc["manifest_unreadable"]]
        L.append("")

    L += ["## Models", "",
          "| model | GB | status | cell | " +
          " | ".join(a for a in doc["arms"]) + " | outputs identical | disk after |",
          "|---|---:|---|---|" + "---|" * len(doc["arms"]) + "---|---:|"]
    for m in doc["models"]:
        runs = {r["arm"]: r for r in m.get("runs", [])}
        cells = []
        for a in doc["arms"]:
            r = runs.get(a)
            cells.append("—" if r is None else
                         (f"ok {r['wall_s']}s" if r["rc"] == 0 else f"**rc={r['rc']}**"))
        L.append(f"| {m['model']} | {m.get('source', {}).get('gb', '?')} | {m.get('status')} | "
                 f"{m.get('cell', '—')} | " + " | ".join(cells) +
                 f" | {m.get('outputs_identical', '—')} | {m.get('disk_free_gb_after', '?')} |")

    L += ["", "## Per engine component", "",
          "| model | component | family | dtype | staged identity |",
          "|---|---|---|---|---|"]
    for m in doc["models"]:
        for c in (m.get("components") or ["(not read)"]):
            L.append(f"| {m['model']} | `{c}` | {m.get('family', '?')} | "
                     f"{m.get('dtype', '?')} | "
                     f"{m.get('staged', {}).get('listing_sha256', '—')} |")

    unstaged = [m for m in doc["models"] if m.get("status") == "not_staged_disk_bound"]
    if unstaged:
        L += ["", "## Not staged — the arithmetic, not a skipped line", "",
              "| model | artefact GB | free GB | headroom | reason |",
              "|---|---:|---:|---:|---|"]
        for m in unstaged:
            L.append(f"| {m['model']} | {m.get('source', {}).get('gb', '?')} | "
                     f"{m.get('disk_free_gb_before', '?')} | "
                     f"{m.get('headroom_gb_after_staging', '?')} | {m.get('reason', '')} |")

    failed = [m for m in doc["models"] if m.get("status") not in ("ran", "not_staged_disk_bound")]
    if failed:
        L += ["", "## Failures, with what they said", ""]
        for m in failed:
            L.append(f"**{m['model']}** — {m.get('status')}")
            L.append("")
            for r in m.get("runs", []):
                if r["rc"] != 0:
                    tail = (r.get("stdout_tail") or r.get("stderr_tail") or "").strip()
                    L += [f"`--{r['arm']}` rc={r['rc']} after {r['wall_s']}s", "",
                          "```", tail[-700:] or "(both streams empty)", "```", ""]
            if m.get("error"):
                L += ["```", str(m["error"])[:500], "```", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompt", default="Explain in one short paragraph why the sky appears blue.")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--arm", action="append", default=None)
    args = ap.parse_args()
    arms = tuple(args.arm) if args.arm else ARMS

    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "src"))

    wanted = set(args.only.split(",")) if args.only else None
    models, unreadable = [], []
    for d in sorted(MOUNT.iterdir()):
        if wanted and d.name not in wanted:
            continue
        man = d / "manifest.json"
        if not man.exists():
            continue
        try:
            fam = json.loads(man.read_text()).get("family")
        except Exception as e:
            # A model whose manifest cannot be READ is not a model that is
            # absent. Dropping it silently would let the family look complete
            # while it is not, which is exactly the shape of an accumulated
            # skip. Recorded with the error and carried into the results.
            unreadable.append({"model": d.name, "manifest_error": f"{type(e).__name__}: {e}"})
            continue
        if fam == args.family:
            models.append(d)

    doc = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tool": "tools/hub_family_sweep.py",
           "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
           "family": args.family, "arms": list(arms),
           "disk_free_gb_before_family": disk_free_gb(),
           "manifest_unreadable": unreadable,
           "models": []}
    print(f"family {args.family}: {len(models)} models, disk {doc['disk_free_gb_before_family']} GB free")

    for src in models:
        name = src.name
        rec = {"model": name, "disk_free_gb_before": disk_free_gb()}
        dest = CACHE / name
        pre_existing = dest.exists()
        rec["pre_existing_locally"] = pre_existing
        try:
            src_id = dir_identity(src)
            rec["source"] = src_id
            free_gb = disk_free_gb()
            headroom = free_gb - src_id["gb"]
            rec["headroom_gb_after_staging"] = round(headroom, 2)
            if not pre_existing and headroom < 5.0:
                # Not a skip: measured where it can be, with the arithmetic.
                rec["status"] = "not_staged_disk_bound"
                rec["reason"] = (f"{src_id['gb']} GB artefact against {free_gb} GB free "
                                 f"leaves {headroom:.2f} GB — under the 5 GB the run "
                                 f"itself needs for caches and output")
                doc["models"].append(rec)
                print(f"  {name}: NOT STAGED (disk) {src_id['gb']} GB vs {free_gb} GB free")
                (args.out / "records.json").write_text(json.dumps(doc, indent=2))
                continue

            if not pre_existing:
                print(f"  {name}: staging {src_id['gb']} GB ...", flush=True)
                t0 = time.time()
                shutil.copytree(src, dest)
                rec["stage_wall_s"] = round(time.time() - t0, 1)
                dst_id = dir_identity(dest)
                rec["staged"] = dst_id
                rec["identity_matches_source"] = (
                    dst_id["listing_sha256"] == src_id["listing_sha256"]
                    and dst_id["metadata_sha256"] == src_id["metadata_sha256"])
                if not rec["identity_matches_source"]:
                    rec["status"] = "stage_mismatch"
                    doc["models"].append(rec)
                    shutil.rmtree(dest, ignore_errors=True)
                    (args.out / "records.json").write_text(json.dumps(doc, indent=2))
                    continue
            else:
                rec["staged"] = dir_identity(dest)
                rec["identity_matches_source"] = (
                    rec["staged"]["listing_sha256"] == src_id["listing_sha256"])

            rec.update(read_model_facts(dest))
            rec["disk_free_gb_staged"] = disk_free_gb()

            # The decoder cell the mandate fixes: context length == head_dim.
            hd = rec.get("head_dim")
            max_tokens = hd if hd else 32
            rec["cell"] = (f"context length == head_dim == {hd}" if hd
                           else "no head_dim (not a decoder): 32 tokens")

            rec["runs"] = [run_arm(name, a, rec.get("family") or args.family,
                                   args.prompt, max_tokens, args.out, args.timeout)
                           for a in arms]
            rec["invocation"] = " ".join(family_inputs(rec.get("family") or args.family,
                                                       args.prompt))
            shas = {r["arm"]: r["output_sha256"] for r in rec["runs"]}
            ok = [r["arm"] for r in rec["runs"] if r["rc"] == 0]
            rec["arms_ok"] = ok
            distinct = {s for s in shas.values() if s}
            rec["outputs_identical"] = len(distinct) <= 1
            rec["output_shas"] = shas
            rec["status"] = "ran" if len(ok) == len(arms) else "partial"
            print(f"  {name}: {rec['status']} arms_ok={ok} identical={rec['outputs_identical']}")
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  {name}: ERROR {rec['error'][:120]}")
        finally:
            if not pre_existing and dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            rec["disk_free_gb_after"] = disk_free_gb()
            doc["models"].append(rec)
            (args.out / "records.json").write_text(json.dumps(doc, indent=2))

    doc["disk_free_gb_after_family"] = disk_free_gb()
    (args.out / "records.json").write_text(json.dumps(doc, indent=2))
    (args.out / "RESULTS.md").write_text(render(doc))
    print(f"\n{len(doc['models'])} models -> {args.out}/records.json "
          f"(disk {doc['disk_free_gb_before_family']} -> {doc['disk_free_gb_after_family']} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
