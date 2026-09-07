#!/usr/bin/env python3
"""Retrace, rebuild, gate and re-upload the containers that carry a corrupted
symbolic dimension — resumable, one card, one model at a time.

Owner's word, 2026-09-07: the 98 component graphs (47 containers) carrying an
integer symbolic dim that contradicts the trace are retraced with the corrected
tracer, each in its native dtype, and the zoo re-uploaded — with a gate before
every upload: on its sequential oracle and on its family's locked protocol the
retraced artifact produces a byte-identical output to the old one where the
old was right, and a right output where the old was not, every difference
explained by the closed defect and nothing else. An artifact that fails the
gate is not uploaded; it is a chantier opened and closed first. The old
artifacts stay on the hub (a replace keeps the previous object downloadable);
nothing is deleted. A cut run resumes from its last validated artifact.

Steps per model (state in <out>/<model>/state.json, each idempotent):
  old_outputs  the installed container, run on the sequential oracle and on the
               family protocol (Triton engine) from the given source tree
  trace        forge trace (corrected tracer) into the toolchain's graph cache
  build        forge build → <models-root>/<model>/model.nbx
  backup       the old container copied to <backup>/<model> (restorable)
  install      forge local --overwrite
  new_outputs  the same two runs on the new container
  gate         bytes old vs new on both runs + graph diff (only symbolic_shape
               annotations may differ) → PASS / NEEDS_EXPLANATION / FAIL
  upload       forge replace (an existing hub entry) or forge publish (a new
               one) — only when NEUROBRIX_API_TOKEN is present; else the
               model waits in READY_FOR_UPLOAD

    python tools/retrace_zoo.py --models a,b --gpu 0 --src /path/to/src \
        [--out validation_outputs/retrace_2026_09_07] [--models-root /home/mlops/nbx_builds]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import precision_zoo_campaign as C  # noqa: E402
import repo_env  # noqa: E402  — the repository's .env, loaded the way the build toolchain loads its own

repo_env.load()

PY = "/home/mlops/ml/venv/bin/python"
FORGE = REPO / "forge" / "forge.py"
CACHE = Path.home() / ".neurobrix" / "cache"
HUB_MAP = REPO / "validation_outputs" / "retrace_2026_09_07" / "hub_map.json"
FAMILIES = REPO / "validation_outputs" / "retrace_2026_09_07" / "families.json"
ANNOTATION_KEYS = {"symbolic_shape"}       # the only tensor fields the closed defect touches
#: The toolchain's registry key when it differs from the installed container's name (the hub's name).
REGISTRY_ALIAS = {"Sana-1600M-MultiLing": "Sana_1600M_1024px_MultiLing"}


def log(msg):
    print(f"[retrace {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def rotate(logfile: Path) -> None:
    """A step's log holds one attempt: an earlier attempt's log is kept aside
    as <name>.<n>.log, so a reader (or a count of the tracer's guard lines)
    never mixes two passes."""
    if not logfile.exists() or logfile.stat().st_size == 0:
        return
    n = 1
    while (logfile.with_name(f"{logfile.stem}.{n}{logfile.suffix}")).exists():
        n += 1
    logfile.rename(logfile.with_name(f"{logfile.stem}.{n}{logfile.suffix}"))


def run(cmd, env, logfile: Path, timeout: int, cwd=None) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    rotate(logfile)
    with open(logfile, "a") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n"); fh.flush()
        try:
            return subprocess.run([str(c) for c in cmd], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout, cwd=cwd).returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[retrace] TIMEOUT after {timeout} s\n")
            return -9


def _trace_value(v):
    """The trace value a shape argument claims: an integer, or a symbol's trace."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, dict) and v.get("type") == "symbol":
        tv = v.get("trace", v.get("trace_value"))
        return tv if isinstance(tv, int) and not isinstance(tv, bool) else None
    return None


def _leaf_diffs(a, b, path=()):
    """Every leaf where two JSON trees of the same structure differ; None when the structure itself differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return None
        out = []
        for k in a:
            d = _leaf_diffs(a[k], b[k], path + (k,))
            if d is None:
                return None
            out += d
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            d = _leaf_diffs(x, y, path + (i,))
            if d is None:
                return None
            out += d
        return out
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        # a symbol {"type": "symbol", ...} against a bare integer is a leaf pair
        if (isinstance(a, dict) and a.get("type") == "symbol" and isinstance(b, int)) or \
           (isinstance(b, dict) and b.get("type") == "symbol" and isinstance(a, int)):
            return [(path, a, b)]
        return None
    return [] if a == b else [(path, a, b)]


def witnessed_arg_changes(old_op: dict, new_op: dict, tensors_new: dict):
    """The differences between two records of one op when each is the closed
    defect at the argument level: a shape argument (a `size`/`shape` list, or
    the `args` list it mirrors) whose old value — a symbol, or an integer —
    claimed a trace value that contradicts the extent the op's own output
    tensor witnessed at that position, replaced by that witnessed integer.
    Returns the sites, or None when any difference is of another kind."""
    if {k for k in set(old_op) | set(new_op) if old_op.get(k) != new_op.get(k)} != {"attributes"}:
        return None
    diffs = _leaf_diffs(old_op.get("attributes"), new_op.get("attributes"))
    if not diffs:
        return None
    witnessed = []
    for tid in new_op.get("output_tensor_ids") or []:
        m = tensors_new.get(tid) or {}
        conc = (m.get("symbolic_shape") or {}).get("concrete") or m.get("shape") or []
        if conc:
            witnessed.append(list(conc))
    sites = []
    for path, a, b in diffs:
        if not isinstance(b, int) or isinstance(b, bool):
            return None
        tv = _trace_value(a)
        if tv is None or tv == b:
            return None
        if not path or not isinstance(path[-1], int):
            return None
        pos = path[-1]
        parent = new_op["attributes"]
        for k in path[:-1]:
            parent = parent[k]
        if not any(len(parent) == len(c) and c[pos] == b for c in witnessed):
            return None
        sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b})
    return sites


HUB_STORE_HEALTH = "http://10.0.0.36:9000/minio/health/cluster"


def hub_store_health(url: str = HUB_STORE_HEALTH, timeout: float = 10.0):
    """The hub object store's cluster health code (200 = read/write quorum), or the error's name."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


class Model:
    def __init__(self, name: str, args):
        self.name = name
        self.args = args
        self.dir = Path(args.out) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"model": name, "steps": {}}
        fams = json.loads(FAMILIES.read_text()) if FAMILIES.exists() else {}
        self.family = fams.get(name) or C.family_of(name)
        hub = json.loads(HUB_MAP.read_text()) if HUB_MAP.exists() else {}
        self.hub = hub.get(name)                       # "org/name" or None (a new entry)
        self.registry_name = REGISTRY_ALIAS.get(name, name)
        self.new_name = (self.state["steps"].get("install") or {}).get("installed_name") or name

    def done(self, step): return (self.state["steps"].get(step) or {}).get("ok") is True
    def mark(self, step, ok, **info):
        self.state["steps"][step] = {"ok": ok, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        self.state_path.write_text(json.dumps(self.state, indent=1))

    # -- environment -------------------------------------------------------
    def env(self, tree: bool = True):
        e = {**os.environ}
        if self.args.gpu not in (None, ""):
            e["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        if tree and self.args.src:
            e["PYTHONPATH"] = str(Path(self.args.src).resolve())
        e["TMPDIR"] = str(Path(self.args.tmp).resolve()); Path(self.args.tmp).mkdir(parents=True, exist_ok=True)
        e["NEUROBRIX_MODELS_ROOT"] = str(Path(self.args.models_root).resolve()); Path(self.args.models_root).mkdir(parents=True, exist_ok=True)
        return e

    # -- the two runs of the locked protocol --------------------------------
    def outputs(self, tag: str) -> dict:
        """The sequential oracle and the family protocol on the Triton engine, outputs hashed."""
        name = self.name if tag == "old" else self.new_name
        req = C.request_args(self.name, self.family, list(self.args.extra))
        ext = C.output_ext(self.family, req)
        res = {}
        for arm, flag in (("sequential", ["--sequential"]), ("triton", ["--triton"])):
            outp = self.dir / f"{tag}_{arm}{ext}"
            if outp.exists() and (self.dir / f"{tag}_{arm}.log").exists() and sha(outp):
                res[arm] = {"rc": 0, "sha": sha(outp), "output": str(outp), "cached": True}
                continue
            cmd = [PY, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())", "run", "--model", name] + req + flag + ["--output", str(outp)]
            t0 = time.time()
            rc = run(cmd, self.env(), self.dir / f"{tag}_{arm}.log", self.args.timeout)
            logtext = (self.dir / f"{tag}_{arm}.log").read_text(errors="replace")
            unsupported = "UNSUPPORTED PATH" in logtext and "encoding" in logtext
            res[arm] = {"rc": rc, "sha": sha(outp), "output": str(outp), "seconds": round(time.time() - t0, 1),
                        "n_a": bool(unsupported and arm == "sequential")}
        return res

    # -- steps --------------------------------------------------------------
    def step_old_outputs(self):
        if self.done("old_outputs"): return True
        if not (CACHE / self.name / "manifest.json").exists():
            self.mark("old_outputs", False, error="no installed container"); return False
        res = self.outputs("old")
        ok = all(v["rc"] == 0 or v.get("n_a") for v in res.values())
        self.mark("old_outputs", ok, runs=res)
        return ok

    def snapshot(self):
        """The model's snapshot: the export first, then the toolchain's own download directory."""
        for root in (Path("/home/mlops/hf_snapshots"), Path.home() / ".cache" / "neurobrix" / "hf_snapshots"):
            for nm in (self.registry_name, self.name):
                p = root / nm
                if p.is_dir() and any(p.iterdir()):
                    return p
        return None

    def step_trace(self):
        if self.done("trace"): return True
        snap = self.snapshot()
        if snap is None:
            self.mark("trace", False, error="no snapshot on the export or in the download directory"); return False
        t0 = time.time()
        cmd = [PY, str(FORGE), "trace", "--model", self.registry_name, "--family", self.family, "--device", "cuda:0", "--path", str(snap)]
        rc = run(cmd, self.env(tree=False), self.dir / "trace.log", self.args.trace_timeout, cwd=str(REPO / "forge"))
        self.mark("trace", rc == 0, rc=rc, seconds=round(time.time() - t0, 1))
        return rc == 0

    def step_build(self):
        if self.done("build"): return True
        snap = self.snapshot()
        if snap is None:
            self.mark("build", False, error="no snapshot on the export or in the download directory"); return False
        t0 = time.time()
        cmd = [PY, str(FORGE), "build", "--snapshot-path", str(snap), "--family", self.family, "--overwrite"]
        rc = run(cmd, self.env(tree=False), self.dir / "build.log", self.args.trace_timeout, cwd=str(REPO / "forge"))
        root = Path(self.args.models_root)
        found = None                                   # the builder writes <models-root>/<family>/<name>/model.nbx
        for nm in (self.registry_name, self.name):
            for cand in [root / self.family / nm / "model.nbx", root / nm / "model.nbx"] + sorted(root.glob(f"*/{nm}/*.nbx")):
                if cand.exists():
                    found = cand; break
            if found:
                break
        ok = rc == 0 and found is not None
        self.mark("build", ok, rc=rc, nbx=str(found) if found else None, seconds=round(time.time() - t0, 1),
                  gb=round(found.stat().st_size / 2**30, 2) if found else None)
        return ok

    def step_backup(self):
        if self.done("backup"): return True
        src = CACHE / self.name
        dst = Path(self.args.backup) / self.name
        if dst.exists() and (dst / "manifest.json").exists():
            self.mark("backup", True, path=str(dst), cached=True); return True
        t0 = time.time()
        shutil.copytree(src, dst, symlinks=True)
        self.mark("backup", True, path=str(dst), seconds=round(time.time() - t0, 1))
        return True

    def step_install(self):
        if self.done("install"): return True
        nbx = (self.state["steps"].get("build") or {}).get("nbx")
        if not nbx or not Path(nbx).exists():
            self.mark("install", False, error="no built .nbx"); return False
        import zipfile
        try:
            with zipfile.ZipFile(nbx) as zf:
                installed = json.loads(zf.read("manifest.json")).get("model_name") or self.name
        except Exception:
            installed = self.name
        rc = run([PY, str(FORGE), "local", nbx, "--overwrite"], self.env(tree=False), self.dir / "install.log", 3600, cwd=str(REPO / "forge"))
        ok = rc == 0 and (CACHE / installed / "manifest.json").exists()
        self.new_name = installed
        self.mark("install", ok, rc=rc, installed_name=installed)
        return ok

    def step_new_outputs(self):
        if self.done("new_outputs"): return True
        res = self.outputs("new")
        ok = all(v["rc"] == 0 or v.get("n_a") for v in res.values())
        self.mark("new_outputs", ok, runs=res)
        return ok

    def graph_diff(self) -> dict:
        """Old vs new graph.json per component: every difference must be the
        closed defect — the symbolic-shape annotation, or a shape argument whose
        false symbol the corrected tracer replaced by the extent its own output
        witnessed (`witnessed_arg_changes`); anything else — an op, a shape, a
        dtype, another attribute — is a difference the gate refuses."""
        old_root = Path(self.args.backup) / self.name / "components"
        new_root = CACHE / self.new_name / "components"
        report = {"components": {}, "beyond_annotation": 0, "annotation_changes": 0, "arg_witnessed": 0, "corrupted_before": 0, "corrupted_after": 0}
        for comp_dir in sorted(new_root.glob("*")):
            og, ng = old_root / comp_dir.name / "graph.json", comp_dir / "graph.json"
            if not og.exists() or not ng.exists():
                report["components"][comp_dir.name] = {"error": "graph missing on one side"}; report["beyond_annotation"] += 1; continue
            o, n = json.loads(og.read_text()), json.loads(ng.read_text())
            ops_o = o["ops"] if isinstance(o.get("ops"), list) else list((o.get("ops") or {}).values())
            ops_n = n["ops"] if isinstance(n.get("ops"), list) else list((n.get("ops") or {}).values())
            rec = {"ops_old": len(ops_o), "ops_new": len(ops_n), "op_diffs": 0, "tensor_diffs_beyond": 0, "annotation_changes": 0,
                   "arg_witnessed": 0, "arg_witnessed_sites": [], "corrupted_before": 0, "corrupted_after": 0}
            to, tn = o.get("tensors") or {}, n.get("tensors") or {}
            if len(ops_o) != len(ops_n):
                rec["op_diffs"] = abs(len(ops_o) - len(ops_n))
            else:
                for a, b in zip(ops_o, ops_n):
                    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                        sites = witnessed_arg_changes(a, b, tn)
                        if sites is None:
                            rec["op_diffs"] += 1
                        else:
                            rec["arg_witnessed"] += len(sites)
                            rec["arg_witnessed_sites"] = (rec["arg_witnessed_sites"] + sites)[:20]
            for tid in set(to) | set(tn):
                a, b = to.get(tid), tn.get(tid)
                if a is None or b is None:
                    rec["tensor_diffs_beyond"] += 1; continue
                for k in set(a) | set(b):
                    if a.get(k) != b.get(k):
                        if k in ANNOTATION_KEYS:
                            rec["annotation_changes"] += 1
                        else:
                            rec["tensor_diffs_beyond"] += 1
            for tens, key in ((to, "corrupted_before"), (tn, "corrupted_after")):
                for m in tens.values():
                    dims = (m.get("symbolic_shape") or {}).get("dims") or []; shp = m.get("shape") or []
                    if any(isinstance(d, int) and not isinstance(d, bool) and i < len(shp) and isinstance(shp[i], int) and d != shp[i] for i, d in enumerate(dims)):
                        rec[key] += 1
            report["components"][comp_dir.name] = rec
            report["beyond_annotation"] += rec["op_diffs"] + rec["tensor_diffs_beyond"]
            report["annotation_changes"] += rec["annotation_changes"]
            report["arg_witnessed"] += rec["arg_witnessed"]
            report["corrupted_before"] += rec["corrupted_before"]; report["corrupted_after"] += rec["corrupted_after"]
        return report

    def step_gate(self):
        if self.done("gate"): return True
        old = (self.state["steps"].get("old_outputs") or {}).get("runs") or {}
        new = (self.state["steps"].get("new_outputs") or {}).get("runs") or {}
        bytes_verdict = {}
        for arm in ("sequential", "triton"):
            o, nn = old.get(arm) or {}, new.get(arm) or {}
            if o.get("n_a") and nn.get("n_a"):
                bytes_verdict[arm] = "N/A (no ATen oracle for an encoded build)"; continue
            if o.get("rc") != 0 or nn.get("rc") != 0:
                bytes_verdict[arm] = f"FAILED (old rc {o.get('rc')}, new rc {nn.get('rc')})"; continue
            same = o.get("sha") and o.get("sha") == nn.get("sha")
            if same:
                bytes_verdict[arm] = "IDENTICAL"
            else:
                try:
                    d = C.gate(Path(o["output"]), Path(nn["output"]))
                except Exception as e:  # noqa: BLE001
                    d = {"error": str(e)}
                bytes_verdict[arm] = {"DIFFERENT": d}
        gd = self.graph_diff()
        identical = all(v == "IDENTICAL" or str(v).startswith("N/A") for v in bytes_verdict.values())
        failed = any(str(v).startswith("FAILED") for v in bytes_verdict.values())
        if failed or gd["beyond_annotation"] or gd["corrupted_after"]:
            verdict = "FAIL"
        elif identical and gd["corrupted_before"] > 0:
            verdict = "PASS"                     # the old was right; the new is byte-identical and carries no corrupted dim
        elif identical:
            verdict = "PASS (no corrupted dim was present in the old container)"
        else:
            verdict = "NEEDS_EXPLANATION"        # bytes differ: the difference must be the closed defect and nothing else
        self.mark("gate", verdict.startswith("PASS"), verdict=verdict, bytes=bytes_verdict, graph=gd)
        log(f"{self.name}: gate {verdict} — bytes {bytes_verdict}; graph: {gd['annotation_changes']} annotation change(s), "
            f"{gd['arg_witnessed']} shape argument(s) to the witnessed extent, "
            f"{gd['beyond_annotation']} beyond, corrupted dims {gd['corrupted_before']} → {gd['corrupted_after']}")
        return verdict.startswith("PASS")

    def step_upload(self):
        if self.done("upload"): return True
        nbx = (self.state["steps"].get("build") or {}).get("nbx")
        try:
            repo_env.require("NEUROBRIX_API_TOKEN")
        except repo_env.MissingVariable as exc:
            # An explicit refusal that names the variable and the file — never a state that waits without saying why.
            self.mark("upload", False, state="REFUSED", reason=str(exc), nbx=nbx)
            log(f"{self.name}: upload {exc}")
            return False
        health = hub_store_health()
        if health != 200:
            # The hub's object store refuses writes (its cluster health answers other than 200):
            # said by name, the artifact stays ready, the chain moves on; a later pass uploads it.
            self.mark("upload", False, state="DEFERRED", reason=f"hub object store cluster health {health} (writes refused); retry when it answers 200", nbx=nbx)
            log(f"{self.name}: upload DEFERRED — the hub object store's cluster health answers {health}; the artifact is gated and ready, a later pass uploads it")
            return False
        if self.hub:
            org, name = self.hub.split("/", 1)
            cmd = [PY, str(FORGE), "replace", "--org", org, "--name", name, nbx]
        else:
            cmd = [PY, str(FORGE), "publish", nbx]
        rc = run(cmd, self.env(tree=False), self.dir / "upload.log", 7200, cwd=str(REPO / "forge"))
        self.mark("upload", rc == 0, rc=rc, command=" ".join(cmd[2:]))
        return rc == 0

    def run_all(self):
        order = [("old_outputs", self.step_old_outputs), ("trace", self.step_trace), ("build", self.step_build),
                 ("backup", self.step_backup), ("install", self.step_install), ("new_outputs", self.step_new_outputs),
                 ("gate", self.step_gate), ("upload", self.step_upload)]
        for name, fn in order:
            if self.done(name):
                continue
            log(f"{self.name}: {name} …")
            ok = fn()
            if not ok:
                st = self.state["steps"].get(name) or {}
                log(f"{self.name}: {name} → {st.get('state') or 'STOPPED'} ({st.get('verdict') or st.get('error') or st.get('reason') or ('rc ' + str(st.get('rc')))})")
                return False
        log(f"{self.name}: complete")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--gpu", default=None, help="the card(s) to pin: an ordinal or a comma list (a two-card trace)")
    ap.add_argument("--src", default=None, help="the engine source tree the runs use (PYTHONPATH)")
    ap.add_argument("--out", default=str(REPO / "validation_outputs" / "retrace_2026_09_07"))
    ap.add_argument("--models-root", default="/home/mlops/nbx_builds", help="where forge build writes the .nbx (root fs, not the export)")
    ap.add_argument("--tmp", default="/home/mlops/nbx_tmp", help="TMPDIR for the build staging (root fs)")
    ap.add_argument("--backup", default=str(Path.home() / ".neurobrix" / "backups" / "retrace_2026_09_07"))
    ap.add_argument("--extra", default="", help="extra request args for the family protocol")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--trace-timeout", type=int, default=14400)
    ap.add_argument("--stop-at", default=None, help="stop after this step (e.g. gate)")
    args = ap.parse_args()
    import shlex
    args.extra = shlex.split(args.extra)
    summary = {}
    for m in [x for x in args.models.split(",") if x]:
        model = Model(m, args)
        if args.stop_at:
            # run steps up to and including stop_at
            for name, fn in [("old_outputs", model.step_old_outputs), ("trace", model.step_trace), ("build", model.step_build),
                             ("backup", model.step_backup), ("install", model.step_install), ("new_outputs", model.step_new_outputs),
                             ("gate", model.step_gate), ("upload", model.step_upload)]:
                if not model.done(name):
                    log(f"{m}: {name} …")
                    if not fn():
                        break
                if name == args.stop_at:
                    break
        else:
            model.run_all()
        summary[m] = {k: (v.get("verdict") or v.get("state") or ("ok" if v.get("ok") else "failed")) for k, v in model.state["steps"].items()}
        sp = Path(args.out) / "summary.json"
        merged = json.loads(sp.read_text()) if sp.exists() else {}
        merged.update(summary)
        sp.write_text(json.dumps(merged, indent=1))
    print(json.dumps(summary, indent=1))
    # The exit code tells the truth: a model whose chain stopped (a gate that did not pass, a refused,
    # deferred or failed upload) makes the run fail, so a marker chained on this command cannot say DONE.
    incomplete = [m for m, st in summary.items() if not (st.get("upload") == "ok")]
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
