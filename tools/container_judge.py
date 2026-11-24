#!/usr/bin/env python3
"""The container judge: the old container's arms, the new container's arms, and the gate between them.

A rebuilt container is judged by the engine, on the engine's own tools, against the container it replaces. The
builder and the engine share nothing (RED_LINES line 28): this tool never imports, calls or runs the builder, puts no
builder path on any path, and reads what the builder recorded only as files.

One run root, `<out>/<model>/`:

  builder/  written by the builder only; READ here, never written:
            previous.json   {container, cache_path, created_at, ...}: the container the old arms run on;
            build.json      the run manifest {model, family, installed_name, cache_path, created_at, nbx_path, sha256
                            (or nbx_sha256), bytes, snapshot, ...}: the container the new arms run on, and the staged
                            .nbx file the gate's verdict is about (nbx_path: absolute, or relative to builder/);
            graph_diff.json the builder's graph comparison of the two containers, read by the gate as it is.
  judge/    written by this tool only, each file whole (a temporary file, then a rename):
            state.json (old_outputs, new_outputs, gate), the arms `old_{sequential,triton}.<ext>` and `new_*` with
            their logs, autotune_directory/ and autotune_replay/, vendor_seed*.png with its json, and verdict.json =
            {verdict, pass, bytes, vendor, request, policy, autotune, judged: {created_at, nbx_sha256},
            old_container: the created_at from previous.json}. `judged.nbx_sha256` is the sha256 the judge COMPUTED
            on the staged .nbx file build.json names; a file that is absent, or whose sha256 (or byte count) is not
            build.json's, is REFUSED BY NAME and the verdict is red. `autotune` adds what the shared replay cache
            holds: {replayed, missing, missing_count} — every kernel key the arms swept because the frozen
            directory did not hold it, counted and named. A key swept during the gate turns the gate RED, by name
            (`swept_keys_refusal`: the gate judges at its own default request, a conventional size).

An arm runs on the container the builder's record names, or not at all: the cached container's `created_at` must
equal previous.json's (old arm) or build.json's (new arm); any other container is REFUSED BY NAME.

    python tools/container_judge.py --out <out> --model <model> --arm old
    python tools/container_judge.py --out <out> --model <model> --arm new --gate

Exit code: 0 a green step, 1 a red one (its state says why), 2 a refusal (the message names it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atelier as _atelier  # noqa: E402  (the workshop's roots)
import one_name  # noqa: E402  (the one name a command line passes)
import precision_zoo_campaign as C  # noqa: E402  (the engine's campaign: request, output kind, gate)
from process_group import run_group  # noqa: E402  (the process-group runner)

REPO = Path(__file__).resolve().parents[1]
PY = C.PY
ENGINE_PY = os.environ.get("NBX_PYTHON", PY)
VENDOR_PY = _atelier.interpreter("competitors")       # the vendor pipelines' own venv (torch under tools/, never under src/)
CACHE = C.CACHE
POLICY_ENV = {"NBX_ACTIVATIONS_FP16_SAFE": "0"}
POLICY = "conservative (NBX_ACTIVATIONS_FP16_SAFE=0)"
BUILDER_FILES = ("previous.json", "build.json", "graph_diff.json")
ARG_KINDS = (("witnessed", "witnessed"), ("symbolized", "symbolized"), ("re-expressed", "re-expressed"),
             ("batch split restored", "batch-split-restored"), ("batch factor restored", "batch-factor-restored"),
             ("unit factor corrected", "unit-factor-corrected"), ("slice end symbolized", "slice-end-symbolized"),
             ("slice end to the end", "slice-end-to-the-end"), ("slice start negated", "slice-start-negated"),
             ("slice start symbolized", "slice-start-symbolized"), ("inference restored", "inference-restored"),
             ("unit-only literalized", "unit-only-literalized"))


class Refused(RuntimeError):
    """A step this tool will not take; the message names why."""


# --------------------------------------------------------------------------------------------------- small helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sha(path: Path):
    """The sha256 of a file, or None when it is absent or empty."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_streamed(path: Path, chunk: int = 1 << 24) -> tuple:
    """(sha256, byte count) of a file read in chunks: a container can be many GB, never read whole."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def autotune_account(directory: Path, replay: Path) -> dict:
    """What the arms' shared replay cache holds against the frozen directory: `replayed` = the keys it holds that the
    frozen directory also holds, `missing` = every key it holds that the frozen directory does NOT (a kernel key the
    arms swept at runtime), named `<kernel>::<key>` and sorted, `missing_count` their number.

    The frozen directory: `<vendor>/<profile>/<kernel>.<dtype>.json`, {"entries": {key text: ...}}. The replay cache:
    `autotune_configs_*.json`, {"<module>.<kernel>::<key text>": config} (the library's capture)."""
    held = set()
    for f in Path(directory).rglob("*.json") if Path(directory).is_dir() else ():
        try:
            entries = json.loads(f.read_text()).get("entries") or {}
        except (OSError, ValueError, AttributeError):
            continue
        kernel = f.name.split(".", 1)[0]
        held.update((kernel, k) for k in entries)
    replayed, missing = 0, []
    for f in sorted(Path(replay).glob("autotune_configs_*.json")) if Path(replay).is_dir() else ():
        try:
            stored = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for ident in stored if isinstance(stored, dict) else ():
            qual, sep, key = str(ident).partition("::")
            kernel = qual.rsplit(".", 1)[-1]
            if sep and (kernel, key) in held:
                replayed += 1
            else:
                missing.append(f"{kernel}::{key}" if sep else str(ident))
    missing = sorted(set(missing))
    return {"replayed": replayed, "missing": missing, "missing_count": len(missing)}


SWEPT_KEYS_RULING = (
    "A KEY SWEPT AT RUN TIME DURING THE GATE TURNS THE GATE RED, by name: the gate judges a container at its own "
    "default request: a CONVENTIONAL size (the owner's condition 6, red line 33): at a conventional size no autotune "
    "runs on the V100 nor on the Mac: every key is certified; a key that is swept there is a certificate missing: the "
    "gate says which keys, and the remedy (the census and certification of those keys). Outside a conventional size "
    "(a user's own size) one autotune is allowed and said: that is the engine's run, not the gate's.")


def swept_keys_refusal(stamp: dict):
    """The project lead's ruling (09:44, 24/11), as its text says — SWEPT_KEYS_RULING:

    A KEY SWEPT AT RUN TIME DURING THE GATE TURNS THE GATE RED, by name: the gate judges a container at its own
    default request: a CONVENTIONAL size (the owner's condition 6, red line 33): at a conventional size no autotune
    runs on the V100 nor on the Mac: every key is certified; a key that is swept there is a certificate missing: the
    gate says which keys, and the remedy (the census and certification of those keys). Outside a conventional size
    (a user's own size) one autotune is allowed and said: that is the engine's run, not the gate's.

    None when the arms swept no key on the gate's request; else the reason the gate is red, naming every
    `<kernel>::<key>` the replay cache holds that the frozen directory does not, and the remedy."""
    missing = list(stamp.get("missing") or [])
    count = int(stamp.get("missing_count", len(missing)) or 0)
    if count <= 0 and not missing:
        return None
    return (f"{count} kernel key(s) swept at run time on the gate's own request, a conventional size: a certificate "
            f"missing for each — " + ", ".join(missing)
            + "; remedy: the census and certification of those keys at this conventional size")


def write_whole(path: Path, obj) -> None:
    """A JSON file written whole: a temporary file beside it, then a rename."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def run(cmd, env, logpath: Path, timeout: int) -> int:
    """The command in its own process group, its output in `logpath` after the command line."""
    with open(logpath, "w") as fh:
        fh.write("$ " + shlex.join(str(c) for c in cmd) + "\n")
        fh.flush()
        rc = run_group(cmd, env, fh, timeout)
        if rc == -9:
            fh.write(f"\nTIMEOUT after {timeout}s\n")
    return rc


def vendor_verdict(vendor: dict) -> str:
    """The gate's verdict from a vendor reproduction of the request when the two sequential arms
    differ: the container that reproduces the vendor is the right one. PixArt-XL-2, 2026-09-07:
    the old container's T5 graph froze transformers' "mask all ones, skip it" shortcut and never
    applied the padding mask — 20.6 dB from the vendor's render; the rebuilt one 41.4 dB."""
    old, new = vendor.get("old") or {}, vendor.get("new") or {}
    po, pn = old.get("psnr_db"), new.get("psnr_db")
    if not new.get("pass"):
        return f"FAIL (vendor: the new container does not reproduce the vendor's render — {pn} dB; old {po} dB)"
    if not old.get("pass"):
        return (f"PASS (corrected: the old container was wrong — new vs vendor {pn} dB, old {po} dB; "
                f"the graph change is the correction)")
    if pn is not None and po is not None and pn >= po:
        return f"PASS (both within the vendor gate, the new arm closer: new {pn} dB, old {po} dB)"
    return f"NEEDS_EXPLANATION (both within the vendor gate but the old arm closer: old {po} dB, new {pn} dB)"


# ------------------------------------------------------------------------------------------- the builder's records
def read_record(builder_dir: Path, name: str) -> dict:
    """One of the builder's files, read; an absent or unreadable one is refused by name."""
    if name not in BUILDER_FILES:
        raise Refused(f"{name}: not a file the builder records for the judge ({', '.join(BUILDER_FILES)})")
    path = Path(builder_dir) / name
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise Refused(f"{path}: absent — the builder has not recorded it; the judge does not guess it") from None
    except (OSError, json.JSONDecodeError) as e:
        raise Refused(f"{path}: unreadable ({e})") from None


def container_of(record: dict, record_name: str) -> tuple:
    """(cache entry name, recorded created_at) of the container a builder record names. The record's cache_path
    must be the engine's own cache entry: the judge runs `neurobrix run --model <entry>` against that cache."""
    cache_path = record.get("cache_path")
    created = record.get("created_at")
    if not cache_path or not created:
        raise Refused(f"{record_name}: names no container (cache_path {cache_path!r}, created_at {created!r})")
    name = record.get("installed_name") or Path(cache_path).name
    if Path(cache_path).resolve() != (Path(CACHE) / name).resolve():
        raise Refused(f"{record_name}: cache_path {cache_path} is not the engine's cache entry {Path(CACHE) / name}")
    return name, created


def cached_created_at(name: str):
    """The created_at the cached container's manifest carries, or None when the cache holds no such entry."""
    try:
        return json.loads((Path(CACHE) / name / "manifest.json").read_text()).get("created_at")
    except (OSError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------------------------------------------------ the judge
class Judge:
    """The judging steps of one model's run root."""

    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.root = Path(args.out) / args.model
        self.builder = self.root / "builder"
        self.dir = self.root / "judge"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("steps", {})
        self._request = {}

    # -- records and identity
    def record(self, name: str) -> dict:
        return read_record(self.builder, name)

    def arm_record(self, tag: str) -> tuple:
        return ("previous.json", self.record("previous.json")) if tag == "old" else ("build.json", self.record("build.json"))

    def arm_container(self, tag: str) -> tuple:
        """(cache entry, created_at) of the arm's container, checked against the cache: another container in the
        cache is refused by name, never judged in its place."""
        rec_name, rec = self.arm_record(tag)
        name, want = container_of(rec, rec_name)
        have = cached_created_at(name)
        if have is None:
            raise Refused(f"{tag} arm: the cache holds no container {name} ({Path(CACHE) / name / 'manifest.json'})")
        if have != want:
            raise Refused(f"{tag} arm: the cached container {name} was created at {have}, {rec_name} names the one "
                          f"created at {want} — another container, not judged in its place")
        return name, want

    def family(self, tag: str) -> str:
        return C.family_of(self.arm_container(tag)[0])

    def request(self, tag: str) -> list:
        name = self.arm_container(tag)[0]
        return C.request_args(name, self.family(tag), list(self.args.extra))

    def current_request(self, tag: str = "new"):
        """The request this attempt runs, memoised. An output and a verdict belong to it: the
        campaign's bound on a length the vendor leaves open (a denoiser's step count, 2026-09-08)
        changes it, and arms measured on another one are another measurement."""
        if tag not in self._request:
            try:
                self._request[tag] = " ".join(str(a) for a in self.request(tag))
            except Exception:                  # a container the cache does not hold yet: no check
                self._request[tag] = ""
        return self._request[tag]

    # -- state
    def save(self) -> None:
        write_whole(self.state_path, self.state)

    def mark(self, step: str, ok: bool, **kw) -> None:
        self.state["steps"][step] = {"ok": bool(ok), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}
        self.save()

    def done(self, step: str) -> bool:
        st = self.state["steps"].get(step) or {}
        if st.get("ok") is not True:
            return False
        if step in ("old_outputs", "new_outputs"):
            try:
                if st.get("created_at") != self.arm_container(step.split("_")[0])[1]:
                    return False               # measured on another container: re-run
            except Refused:
                return False
        if step in ("old_outputs", "new_outputs", "gate"):
            want = self.current_request("old" if step == "old_outputs" else "new")
            if want and st.get("request") not in (None, want):
                return False                   # measured on another request: re-run, re-gate
        if step in ("old_outputs", "new_outputs") and st.get("policy") != POLICY:
            return False                       # measured under another policy: another arm, re-run
        if step in ("old_outputs", "new_outputs"):
            fr = (self.state.get("autotune_freeze") or {}).get("snapshot")
            if not fr or st.get("autotune") != fr:
                return False                   # measured under another kernel-config state: re-run
        if step == "gate":
            fr = (self.state.get("autotune_freeze") or {}).get("snapshot")
            if st.get("policy") != POLICY or not fr or (st.get("autotune") or {}).get("snapshot") != fr:
                return False                   # a verdict on other arms than the stamped ones: re-gate
            if not (self.dir / "verdict.json").exists():
                return False
        return True

    def env(self, tree: bool = True) -> dict:
        """The arms' environment: this process's, with the engine source tree `--src` first on the path when one
        is named (an engine tree, never another repository's)."""
        env = dict(os.environ)
        if tree and self.args.src:
            src = str(Path(self.args.src).resolve())
            env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    # -- the arms
    def autotune_freeze(self) -> dict:
        """The kernel-config state BOTH arms of this gate run under: a snapshot of the engine's
        certified directory taken at the gate's first arm, and one replay cache shared by the
        arms (the first arm sweeps a missing key, the second loads the same choice). Without
        it the Triton arms compare kernel choices, not graphs: CogVideoX-2b, 2026-09-07 —
        the old arm served 3,139 certified keys and swept 19, the new arm 4,350 and 0 (the
        certifier filled the directory between them), 39.7 dB between two videos of one graph."""
        snap = self.dir / "autotune_directory"
        st = self.state.get("autotune_freeze")
        if st and snap.exists():
            return st
        src_dir = Path(self.args.src).resolve() / "neurobrix" / "config" / "autotune" if self.args.src else None
        if snap.exists():
            shutil.rmtree(snap)
        n = 0
        if src_dir and src_dir.exists():
            shutil.copytree(src_dir, snap)
            for f in snap.rglob("*.json"):
                try:
                    n += len(json.loads(f.read_text()).get("entries") or {})
                except Exception:  # noqa: BLE001
                    pass
        else:
            snap.mkdir(parents=True, exist_ok=True)          # no directory in this tree: both arms sweep into one cache
        st = {"snapshot": time.strftime("%Y-%m-%dT%H:%M:%S"), "entries": n, "directory": str(snap), "replay": str(self.dir / "autotune_replay")}
        self.state["autotune_freeze"] = st
        self.save()
        log(f"{self.model}: autotune state frozen for both arms — {n} certified entries at {st['snapshot']}, one replay cache")
        return st

    def outputs(self, tag: str) -> dict:
        """The sequential oracle and the family protocol on the Triton engine, outputs hashed."""
        name = self.arm_container(tag)[0]
        freeze = self.autotune_freeze()
        req = self.request(tag)
        ext = C.output_ext(self.family(tag), req)
        # The REQUEST is part of an output's identity. An arm measured on another request — or on
        # one this state never recorded — is set aside and re-run, never compared with this one's:
        # the campaign's bound on a denoiser's step count changed the video request on 2026-09-08,
        # and Wan2.1-T2V's sequential arm sat in the directory at the vendor's 100 steps beside a
        # triton arm that had timed out, so the gate would have read a DIFFERENT that is only the
        # request. Same discipline as the precision policy above.
        self.request_key = self.current_request(tag) or " ".join(str(a) for a in req)
        prev = (self.state["steps"].get(f"{tag}_outputs") or {}).get("request")
        if prev != self.request_key and any(self.dir.glob(f"{tag}_*")):
            self.set_aside(f"{tag}_", f"measured on {('the request ' + repr(prev)) if prev else 'a request this state did not record'}; "
                                      f"this attempt runs '{self.request_key}'")
        res = {}
        for arm, flag in (("sequential", ["--sequential"]), ("triton", ["--triton"])):
            outp = self.dir / f"{tag}_{arm}{ext}"
            if outp.exists() and (self.dir / f"{tag}_{arm}.log").exists() and sha(outp):
                res[arm] = {"rc": 0, "sha": sha(outp), "output": str(outp), "cached": True}
                continue
            cmd = [ENGINE_PY, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())", "run", "--model", one_name.cli_name(name)] + req + flag + ["--output", str(outp)]
            t0 = time.time()
            # One precision policy on both arms (POLICY above).
            env = self.env(); env.update(POLICY_ENV)
            env["NEUROBRIX_AUTOTUNE_CERTIFIED_DIR"] = freeze["directory"]
            env["NEUROBRIX_REPLAY_CACHE"] = freeze["replay"]
            rc = run(cmd, env, self.dir / f"{tag}_{arm}.log", self.args.timeout)
            logtext = (self.dir / f"{tag}_{arm}.log").read_text(errors="replace")
            unsupported = "UNSUPPORTED PATH" in logtext and "encoding" in logtext
            res[arm] = {"rc": rc, "sha": sha(outp), "output": str(outp), "seconds": round(time.time() - t0, 1),
                        "n_a": bool(unsupported and arm == "sequential")}
        return res

    def set_aside(self, prefix: str, why: str) -> None:
        """The judge's files of `prefix` moved under set_aside/<time>/ with the reason beside them: a previous
        measurement is kept for reading, never compared with this one."""
        files = [p for p in self.dir.glob(f"{prefix}*") if p.is_file()]
        if not files:
            return
        dest = self.dir / "set_aside" / f"{time.strftime('%Y%m%dT%H%M%S')}_{prefix.rstrip('_')}"
        dest.mkdir(parents=True, exist_ok=True)
        for p in files:
            os.replace(p, dest / p.name)
        (dest / "why.txt").write_text(why + "\n")
        log(f"{self.model}: {len(files)} file(s) {prefix}* set aside — {why}")

    def vendor_reproduction(self, old: dict, new: dict) -> dict:
        """Render the request with the vendor's own pipeline at the exact prompt, seed, steps and
        guidance (tools/vendor_image_repro.py under the diffusers venv; reused when a render of
        the same request sits beside the arms) and measure both sequential arms against it in
        the family's own measure. Image families only; others return {}."""
        if self.family("new") != "image":
            return {}
        name = self.arm_container("new")[0]
        req = self.request("new")
        def _opt(flag, default=None):
            return req[req.index(flag) + 1] if flag in req and req.index(flag) + 1 < len(req) else default
        prompt, seed = _opt("--prompt"), int(_opt("--seed", 42))
        gen = {}   # the container's run values: runtime/defaults.json (topology.json flow.generation holds structure only)
        try:
            gen = json.loads((Path(CACHE) / name / "runtime" / "defaults.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
        # The request's pin wins over the container's declaration: the arms render the length the
        # request names (the campaign's bound), and the vendor must render the same one.
        steps = _opt("--steps", gen.get("num_inference_steps"))
        guidance = _opt("--guidance", gen.get("guidance_scale"))
        snap = self.snapshot()
        if prompt is None or snap is None:
            return {"error": "no prompt in the request or no snapshot on this machine"}
        out = self.dir / f"vendor_seed{seed}.png"; meta = out.with_suffix(".json")
        want = {"prompt": prompt, "seed": seed, "steps": steps, "guidance": guidance}
        have = json.loads(meta.read_text()) if meta.exists() else {}
        if not (out.exists() and all(have.get(k) == v for k, v in want.items() if v is not None)):
            cmd = [VENDOR_PY, str(REPO / "tools" / "vendor_image_repro.py"), "--snapshot", str(snap), "--prompt", prompt,
                   "--seed", str(seed), "--out", str(out)]
            if steps is not None: cmd += ["--steps", str(steps)]
            if guidance is not None: cmd += ["--guidance", str(guidance)]
            with open(self.dir / "vendor.log", "a") as fh:
                rc = subprocess.run([str(c) for c in cmd], env=self.env(tree=False), stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not out.exists():
                return {"error": f"the vendor render exited {rc} (vendor.log)"}
            write_whole(meta, want)
        res = {"render": str(out), "request": want}
        for arm, rec in (("old", old.get("sequential") or {}), ("new", new.get("sequential") or {})):
            try:
                g = C.gate(out, Path(rec["output"]))
                res[arm] = {k: g.get(k) for k in ("psnr_db", "ssim", "pass", "identical")}
            except Exception as e:  # noqa: BLE001
                res[arm] = {"error": str(e), "pass": False}
        log(f"{self.model}: vendor reproduction — old {res['old']} / new {res['new']}")
        return res

    def snapshot(self):
        """The vendor snapshot build.json names, when this machine holds it (under hf_snapshots, or as named)."""
        snap = self.record("build.json").get("snapshot")
        path = (snap or {}).get("path") if isinstance(snap, dict) else snap
        if not path:
            return None
        for cand in (Path(path), _atelier.root("hf_snapshots") / Path(path).name):
            if cand.is_dir():
                return cand
        return None

    def _first_error(self, logname: str):
        """The first line of an arm's log that names an error, or None."""
        try:
            lines = (self.dir / logname).read_text(errors="replace").splitlines()
        except OSError:
            return None
        marks = ("Error", "ERROR", "ZERO FALLBACK", "REFUSED", "Exception")
        return next((ln.strip() for ln in lines if any(m in ln for m in marks)), None)

    def step_old_outputs(self):
        if self.done("old_outputs"): return True
        st = self.state["steps"].get("old_outputs") or {}
        if st.get("ok") is True and st.get("policy") != POLICY:
            self.set_aside("old_", f"measured under the policy '{st.get('policy') or 'default'}'; "
                                   f"the gate runs both arms under '{POLICY}'")
        elif st.get("ok") is True and st.get("request") == self.current_request("old"):
            # same policy, same request, yet not done: another container or kernel-config state
            self.set_aside("old_", "a previous attempt's outputs (another container or kernel-config state)")
        prev = self.record("previous.json")
        _, created = self.arm_container("old")
        freeze = self.autotune_freeze()
        res = self.outputs("old")
        ok = all(v["rc"] == 0 or v.get("n_a") for v in res.values())
        unrunnable = None
        on_hub_object = prev.get("container") == "hub previous object"
        if not ok and on_hub_object and all(v["rc"] != 0 and not v.get("n_a") for v in res.values()):
            # The hub's object does not run on this engine (canary, 2026-09-07: the June-09 build
            # users download — "ZERO FALLBACK: No allocation for component 'perception_encoder'").
            # That is a fact about the old arm, recorded by name; the chain goes on and the gate
            # judges the new container on its own two engines agreeing and on the graph diff.
            unrunnable = self._first_error("old_sequential.log") or self._first_error("old_triton.log") or "the runs exited non-zero"
            ok = True
            log(f"{self.model}: the hub's previous object does not run on this engine — {unrunnable[:160]}; the gate judges the new container on its own engines")
        self.mark("old_outputs", ok, runs=res, policy=POLICY, autotune=freeze["snapshot"], unrunnable=unrunnable,
                  request=getattr(self, "request_key", None), created_at=created,
                  container="the hub's previous object" if on_hub_object else "the installed container")
        return ok

    def step_new_outputs(self):
        if self.done("new_outputs"): return True
        # A previous attempt's outputs describe a previous container (or another policy): never reused as this one's.
        st = self.state["steps"].get("new_outputs") or {}
        _, created = self.arm_container("new")
        self.set_aside("new_", "a previous attempt's outputs" if st.get("policy") == POLICY else
                       f"measured under the policy '{st.get('policy') or 'default'}'; the gate runs both arms under '{POLICY}'")
        freeze = self.autotune_freeze()
        res = self.outputs("new")
        ok = all(v["rc"] == 0 or v.get("n_a") for v in res.values())
        self.mark("new_outputs", ok, runs=res, policy=POLICY, autotune=freeze["snapshot"],
                  request=getattr(self, "request_key", None), created_at=created)
        return ok

    # -- the gate
    def _gate_refusal(self, so: dict, sn: dict):
        """Why the two arms cannot be compared, by name, or None."""
        if not so.get("ok") or not sn.get("ok"):
            return (f"an arm is not green: old_outputs {so.get('ok')!r}, new_outputs {sn.get('ok')!r} — "
                    f"the gate judges two measured arms")
        _, old_created = container_of(self.record("previous.json"), "previous.json")
        _, new_created = container_of(self.record("build.json"), "build.json")
        if so.get("created_at") != old_created:
            return (f"the old arm ran on the container created at {so.get('created_at')}, previous.json names the "
                    f"one created at {old_created}")
        if sn.get("created_at") != new_created:
            return (f"the new arm ran on the container created at {sn.get('created_at')}, build.json names the "
                    f"one created at {new_created}")
        if so.get("policy") != sn.get("policy"):
            return (f"the two arms ran under different precision policies: old '{so.get('policy') or 'default'}', "
                    f"new '{sn.get('policy') or 'default'}' — a difference between them would not be the graph's")
        if so.get("autotune") != sn.get("autotune"):
            return (f"the two arms ran under different kernel-config states: old '{so.get('autotune') or 'unfrozen'}', "
                    f"new '{sn.get('autotune') or 'unfrozen'}' — a Triton difference between them would be a config choice, not the graph's")
        if so.get("request") != sn.get("request"):
            return (f"the two arms ran on different requests: old '{so.get('request')}', new '{sn.get('request')}' — "
                    f"a difference between them would be the request's")
        return None

    def _gate_unrunnable(self, so: dict, new: dict, gd: dict) -> tuple:
        """No old arm to compare bytes with: the new container's own engines must agree
        (the sequential oracle and the Triton engine, byte for byte), and the graph diff
        against the hub's graphs must be the closed defect alone."""
        ns, nt = new.get("sequential") or {}, new.get("triton") or {}
        # The object does not run because the engine's Prism plan follows today's registry
        # layout: a component the old topology had and the new one has not (or the reverse)
        # is that layout change, named under "topology_layout", not beyond.
        layout = [t for t in gd.get("topology") or [] if t.get("path", "").startswith("components.")]
        if layout:
            gd["topology"] = [t for t in gd["topology"] if not t.get("path", "").startswith("components.")]
            gd["topology_layout"] = layout
            gd["beyond_annotation"] -= len(layout)
        agree = ns.get("rc") == 0 and nt.get("rc") == 0 and ns.get("sha") and ns.get("sha") == nt.get("sha")
        cmp = None
        if ns.get("rc") == 0 and nt.get("rc") == 0 and not agree:
            try:
                cmp = C.gate(Path(ns["output"]), Path(nt["output"]))
            except Exception as e:  # noqa: BLE001
                cmp = {"error": str(e)}
        bytes_verdict = {"old": f"UNRUNNABLE on this engine: {so['unrunnable'][:200]}",
                         "new engines": "IDENTICAL" if agree else {"DIFFERENT": cmp} if cmp else "FAILED"}
        if gd["beyond_annotation"] or gd["corrupted_after"] or not (ns.get("rc") == 0 and nt.get("rc") == 0):
            verdict = "FAIL"
        elif agree:
            verdict = "PASS (the hub's object does not run on this engine; the new container's two engines agree byte for byte; graph = the closed defect)"
        else:
            verdict = "NEEDS_EXPLANATION"
        return verdict, bytes_verdict

    def _bytes_verdict(self, old: dict, new: dict) -> dict:
        bytes_verdict = {}
        for arm in ("sequential", "triton"):
            o, nn = old.get(arm) or {}, new.get(arm) or {}
            if o.get("n_a") and nn.get("n_a"):
                bytes_verdict[arm] = "N/A (no ATen oracle for an encoded build)"; continue
            if o.get("rc") not in (0, None) and nn.get("rc") not in (0, None):
                # The engine refuses BOTH containers alike on this arm (SANA-Video's sequential
                # arm, 2026-09-07: the VAE tiler's replicate pad on a 6-D tile, after 1.6 h of
                # denoising, June and September containers alike): an engine gap on that path,
                # named as such, not a difference between the graphs; the other arm decides.
                e_old, e_new = self._first_error(f"old_{arm}.log"), self._first_error(f"new_{arm}.log")
                if e_old and e_new and e_old.split(":")[-1].strip() == e_new.split(":")[-1].strip():
                    bytes_verdict[arm] = f"N/A (the engine refuses both containers alike: {e_new[:160]})"; continue
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
        return bytes_verdict

    def _gate_compared(self, old: dict, new: dict, gd: dict) -> tuple:
        bytes_verdict = self._bytes_verdict(old, new)
        identical = all(v == "IDENTICAL" or str(v).startswith("N/A") for v in bytes_verdict.values())
        failed = any(str(v).startswith("FAILED") for v in bytes_verdict.values())
        # The two SEQUENTIAL arms differ — the ATen semantics changed between the graphs: the vendor's
        # own render at the exact request decides which container is right (owner's rule: reproduce
        # with the vendor pipeline before declaring). A routing field removed or a corrupted dim
        # left behind still refuses, whatever the vendor says.
        vendor = {}
        seq_differ = isinstance(bytes_verdict.get("sequential"), dict) and "DIFFERENT" in bytes_verdict["sequential"]
        if seq_differ and getattr(self.args, "vendor_on_diff", False) and not failed:
            vendor = self.vendor_reproduction(old, new)
        removed = [t for t in gd.get("topology") or [] if t.get("kind") in ("removed", "changed")]
        if vendor and "error" not in vendor and not gd["corrupted_after"] and not removed:
            verdict = vendor_verdict(vendor)
        elif failed or gd["corrupted_after"]:
            verdict = "FAIL"
        elif gd["beyond_annotation"] and gd.get("stimulus_change"):
            # Every recorded shape moves when the stimulus moves, so the count is not evidence of
            # anything and must not be read as a refusal. What the gate can still say is what the
            # BYTES did, and it says it here rather than hiding behind a number it cannot interpret.
            moved = "; ".join(f"{c}: " + ", ".join(f"{k} {v['old']} -> {v['new']}" for k, v in m.items())
                              for c, m in gd["stimulus_change"].items())
            verdict = (f"NEEDS_EXPLANATION (the two graphs were built at DIFFERENT stimuli — {moved} — so "
                       f"{gd['beyond_annotation']} shape changes cannot be classified by a diff that assumes "
                       f"a common stimulus; the bytes are the evidence here: {bytes_verdict}. Judge the artefacts.)")
        elif gd["beyond_annotation"]:
            verdict = "FAIL"
        elif gd.get("topology_additions") and not identical:
            verdict = "NEEDS_EXPLANATION"       # bytes differ and the topology gained fields: the additions may be the cause
        elif identical and gd["corrupted_before"] > 0:
            verdict = "PASS"                     # the old was right; the new is byte-identical and carries no corrupted dim
        elif identical:
            verdict = "PASS (no corrupted dim was present in the old container)"
        else:
            verdict = "NEEDS_EXPLANATION"        # bytes differ: the difference must be the closed defect and nothing else
        return verdict, bytes_verdict, vendor

    def nbx_identity(self) -> tuple:
        """(refusal or None, the sha256 the judge computed or None): the staged .nbx file build.json names, hashed in
        chunks and compared with build.json's sha256 (and byte count, when it records one). An absent file, a record
        naming none, or another sha256 is REFUSED BY NAME: the verdict is about the file the upload carries."""
        build = self.record("build.json")
        want = build.get("nbx_sha256", build.get("sha256"))
        named = build.get("nbx_path")
        if not named:
            return "build.json names no staged .nbx file (nbx_path) — the judge does not guess which file it judges", None
        path = Path(named) if Path(named).is_absolute() else self.builder / named
        if not want:
            return f"build.json records no sha256 for the staged .nbx {path}", None
        if not path.is_file():
            return f"the staged .nbx {path} that build.json names is absent", None
        have, n = sha_streamed(path)
        if have != want:
            return (f"the staged .nbx {path} has sha256 {have}, build.json records {want} — another file than the one "
                    f"the builder recorded, not judged in its place"), have
        want_bytes = build.get("nbx_bytes", build.get("bytes"))
        if want_bytes is not None and int(want_bytes) != n:
            return f"the staged .nbx {path} is {n} bytes, build.json records {want_bytes}", have
        return None, have

    def autotune_stamp(self) -> dict:
        """The frozen kernel-config state with what the arms' replay cache holds against it (`autotune_account`)."""
        freeze = dict(self.state.get("autotune_freeze") or {})
        if freeze.get("directory") and freeze.get("replay"):
            freeze.update(autotune_account(Path(freeze["directory"]), Path(freeze["replay"])))
        return freeze

    def write_verdict(self, verdict: str, bytes_verdict: dict, vendor, nbx_sha256=None) -> dict:
        """verdict.json, whole: what the builder's upload reads, with the containers it judged. `judged.nbx_sha256`
        is the sha256 computed on the staged file when one was computed, else build.json's (on a red verdict)."""
        build, prev = self.record("build.json"), self.record("previous.json")
        out = {"verdict": verdict, "pass": verdict.startswith("PASS"), "bytes": bytes_verdict, "vendor": vendor or None,
               "request": self.current_request("new"), "policy": POLICY, "autotune": self.autotune_stamp(),
               "judged": {"created_at": build.get("created_at"),
                          "nbx_sha256": nbx_sha256 or build.get("nbx_sha256", build.get("sha256"))},
               "old_container": prev.get("created_at")}
        write_whole(self.dir / "verdict.json", out)
        return out

    def refuse_gate(self, reason: str, nbx_sha256=None) -> bool:
        self.mark("gate", False, verdict="FAIL", reason=reason, autotune=self.autotune_stamp())
        self.write_verdict(f"FAIL ({reason})", {}, None, nbx_sha256)
        log(f"{self.model}: gate FAIL — {reason}")
        return False

    def step_gate(self):
        # The staged file first, every time: a verdict already written is about the file as it was then.
        nbx_refusal, nbx_sha256 = self.nbx_identity()
        if nbx_refusal:
            return self.refuse_gate(nbx_refusal, nbx_sha256)
        if self.done("gate"): return True
        so = self.state["steps"].get("old_outputs") or {}
        sn = self.state["steps"].get("new_outputs") or {}
        reason = self._gate_refusal(so, sn)
        if reason:
            return self.refuse_gate(reason)
        gd = self.record("graph_diff.json")      # the builder's comparison of the two graphs, read as it is
        old, new = so.get("runs") or {}, sn.get("runs") or {}
        vendor = {}
        if so.get("unrunnable"):
            verdict, bytes_verdict = self._gate_unrunnable(so, new, gd)
        else:
            verdict, bytes_verdict, vendor = self._gate_compared(old, new, gd)
        stamp = self.autotune_stamp()
        swept = swept_keys_refusal(stamp)    # the ruling: a key swept on the gate's request turns the gate red
        if swept:
            verdict = f"FAIL ({swept}; the gate's verdict on the bytes: {verdict})"
        self.mark("gate", verdict.startswith("PASS"), verdict=verdict, bytes=bytes_verdict, graph=gd, policy=POLICY,
                  autotune=stamp, vendor=vendor or None, request=self.current_request("new"))
        self.write_verdict(verdict, bytes_verdict, vendor, nbx_sha256)
        log(f"{self.model}: gate {verdict} — both arms under {POLICY}; bytes {bytes_verdict}; " + graph_summary(gd)
            + f"; {autotune_summary(stamp)}")
        return verdict.startswith("PASS")


def autotune_summary(stamp: dict) -> str:
    """The replay cache's account, one clause for the log: counts, and every missing key by name."""
    missing = stamp.get("missing") or []
    return (f"autotune: {stamp.get('replayed', 0)} key(s) replayed from the frozen directory, "
            f"{stamp.get('missing_count', len(missing))} missing from it (swept)"
            + (": " + ", ".join(missing) if missing else ""))


def graph_summary(gd: dict) -> str:
    """The graph diff's counts, one line for the log."""
    comps = (gd.get("components") or {}).values()
    kinds = ", ".join(f"{label} {sum(r.get('arg_kinds', {}).get(key, 0) for r in comps)}" for label, key in ARG_KINDS)
    return (f"graph: {gd.get('annotation_changes')} annotation change(s), "
            f"{gd.get('arg_witnessed')} shape argument(s) of the closed defect ({kinds}), "
            f"{gd.get('pruned_dead_ops')} dead op(s) pruned, "
            f"{gd.get('beyond_annotation')} beyond ({len(gd.get('topology') or [])} in the topology; "
            f"{len(gd.get('topology_additions') or [])} topology field(s) added), "
            f"corrupted dims {gd.get('corrupted_before')} → {gd.get('corrupted_after')}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="the container judge: old arms, new arms, the gate (verdict.json)")
    ap.add_argument("--out", required=True, help="the work root; the run root is <out>/<model>/")
    ap.add_argument("--model", required=True, help="the run root's model directory under --out")
    ap.add_argument("--arm", required=True, choices=("old", "new"),
                    help="old: the container previous.json names; new: the one build.json names")
    ap.add_argument("--gate", action="store_true", help="with --arm new: judge both arms and write verdict.json")
    ap.add_argument("--vendor-on-diff", action="store_true",
                    help="when the two sequential arms differ, render the vendor's pipeline and let it decide")
    ap.add_argument("--src", default=None, help="an ENGINE source tree: its autotune directory and its src on the path")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --: extra request arguments")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.extra = [a for a in args.extra if a != "--"]
    try:
        if args.gate and args.arm != "new":
            raise Refused("--gate runs with --arm new: the gate follows the new container's arms")
        j = Judge(args)
        if args.arm == "old":
            return 0 if j.step_old_outputs() else 1
        if not j.step_new_outputs():
            return 1
        return 0 if (not args.gate or j.step_gate()) else 1
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
