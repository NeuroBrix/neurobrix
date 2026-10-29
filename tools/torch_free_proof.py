#!/usr/bin/env python3
"""The torch-free proof: the Triton branch of NeuroBrix and the nbxtensor library run in an environment where torch
is NOT INSTALLED (not masked), and not one import of torch (torchvision, torchaudio, functorch) is even attempted (Hocine's order, 2026-10-29 15:52).

WHY AN ENVIRONMENT, AND WHY A RECORDER
A blocker in sys.meta_path (tools/r33_import_without_torch.py) proves the modules it is handed import without torch.
It cannot prove what a USER meets: the package's dependencies, the subprocesses a run starts, the autotune's bench
(triton's own GPUDriver imports torch, triton/backends/driver.py), a `try: import torch / except ImportError:` that
silently takes another road. So the steps run in an interpreter where torch is absent — refused here if
`find_spec("torch")` answers anything — and a `sitecustomize` placed first on PYTHONPATH (inherited by every child
process) records EVERY attempt to import torch or a submodule with its call site, the caught ones included, and every
`find_spec` probe of it (a driver probe is a red too, R33). Any
attempt is a red that names its importer.

THE STEPS (each one's verdict and log in --out):
  library   the library's own tests, README examples included (`nbxtensor/tests`)
  models    the small reference models through the entry point (`neurobrix run --triton`, the request the gate
            derives, `tools/trace_request.gate_request`); an output's sha is compared with `--expect` when given
  sweep     an uncertified runtime autotune sweep: `NBX_AUTOTUNE_CERTIFIED=off`, an EMPTY replay cache
            (`NEUROBRIX_REPLAY_CACHE`), one model; red unless its run says it swept at least one key
  command   any further command (`--command`, repeatable: the card-less coverage, a backend's own suite)

    python3 tools/torch_free_proof.py --python <venv without torch>/bin/python --tree <worktree> --out <dir> \\
        [--steps library,models,sweep] [--models A,B] [--sweep-model M] [--expect <file: model sha16 per line>] \\
        [--command "<shell command run in the tree>"]

The machine's own values (which models, their expected shas) are arguments, never written here: the same tool
serves CUDA and Metal.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

RECORDER = '''\
"""Recorder of torch import attempts (tools/torch_free_proof.py): every `import torch` / `torch.*` attempted in this
process is appended to $NBX_TORCH_ATTEMPTS with its call site, then left to fail as it would."""
import importlib.abc, json, os, sys, traceback


class _TorchAttempts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "torchvision", "torchaudio", "functorch"):
            out = os.environ.get("NBX_TORCH_ATTEMPTS")
            if out:
                frames = [f"{f.filename}:{f.lineno} {f.name}" for f in traceback.extract_stack()[:-1]
                          if "importlib" not in f.filename and "<frozen" not in f.filename]
                with open(out, "a") as fh:
                    fh.write(json.dumps({"pid": os.getpid(), "argv": sys.argv[:3], "module": fullname,
                                         "site": frames[-1] if frames else "?", "stack": frames[-8:]}) + "\\n")
        return None


sys.meta_path.insert(0, _TorchAttempts())
'''


def _env(tree: Path, out: Path, extra=None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(out / "recorder"), str(tree / "src"), str(tree / "nbxtensor" / "src")])
    env["NBX_TORCH_ATTEMPTS"] = str(out / "torch_attempts.jsonl")
    env.update(extra or {})
    return env


def _run(cmd, cwd: Path, env: dict, log: Path) -> int:
    with open(log, "w") as fh:
        fh.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def _sha16(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _gate_request(py: str, tree: Path, out: Path, model: str) -> list:
    code = ("import sys, json; sys.path.insert(0, 'tools'); import trace_request as TR; "
            f"print(json.dumps(TR.gate_request({model!r})))")
    r = subprocess.run([py, "-c", code], cwd=tree, env=_env(tree, out), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{model}: the gate request cannot be derived:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _ext(model: str) -> str:
    """The output file the run writes, by what the model produces (read from its container's family)."""
    cache = Path(os.environ.get("NEUROBRIX_CACHE", Path.home() / ".neurobrix" / "cache"))
    family = json.loads((cache / model / "manifest.json").read_text()).get("family")
    return {"llm": "txt", "stt": "txt", "tts": "wav"}.get(family, "png")


def _model_run(py, tree, out, model, label, extra_env=None):
    req = _gate_request(py, tree, out, model)
    dest = out / f"{label}.{model}.{_ext(model)}"
    log = out / f"{label}.{model}.log"
    t = time.time()
    rc = _run([py, "-m", "neurobrix", "run", "--model", model, *req, "--triton", "--output", str(dest)],
              tree, _env(tree, out, extra_env), log)
    if dest.exists():
        sha = _sha16(dest)
    else:          # a text answer printed rather than written
        lines = [l for l in log.read_text().splitlines() if re.match(r"^(Output|Generated|Transcription|Text)", l)]
        sha = __import__("hashlib").sha256((lines[-1] if lines else "").encode()).hexdigest()[:16] if lines else "none"
    return rc, sha, int(time.time() - t), log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", required=True, help="the interpreter of an environment WITHOUT torch installed")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="library,models,sweep")
    ap.add_argument("--models", default="")
    ap.add_argument("--sweep-model", default="")
    ap.add_argument("--expect", default="", help="file: '<model> <sha16>' per line, this machine's reference outputs")
    ap.add_argument("--command", action="append", default=[])
    a = ap.parse_args(argv)
    tree, out = Path(a.tree).resolve(), Path(a.out).resolve()
    (out / "recorder").mkdir(parents=True, exist_ok=True)
    (out / "recorder" / "sitecustomize.py").write_text(RECORDER)
    attempts = out / "torch_attempts.jsonl"
    attempts.write_text("")

    probe = subprocess.run([a.python, "-c", "import importlib.util as u; print(u.find_spec('torch'))"],
                           capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                if k != "PYTHONPATH"})
    if probe.stdout.strip() != "None":
        raise SystemExit(f"{a.python}: torch is installed or reachable there ({probe.stdout.strip()}) — the proof "
                         f"needs an environment where it is ABSENT, not masked")
    rec = subprocess.run([a.python, "-c", "import sys; print(any(type(f).__name__ == '_TorchAttempts' "
                          "for f in sys.meta_path))"], capture_output=True, text=True, env=_env(tree, out))
    if rec.stdout.strip() != "True":
        raise SystemExit(f"the recorder is not loaded by {a.python} (another sitecustomize shadows it?): "
                         f"{rec.stdout}{rec.stderr}")

    expect = {}
    if a.expect:
        for line in Path(a.expect).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                m, s = line.split()[:2]
                expect[m] = s
    steps = [s for s in a.steps.split(",") if s]
    verdicts = []
    if "library" in steps:
        rc = _run([a.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "nbxtensor/tests"], tree,
                  _env(tree, out), out / "library.log")
        verdicts.append(("library", rc == 0, f"rc={rc} {(out / 'library.log').read_text().strip().splitlines()[-1]}"))
    if "models" in steps:
        for model in [m for m in a.models.split(",") if m]:
            rc, sha, secs, log = _model_run(a.python, tree, out, model, "models")
            same = expect.get(model) in (None, sha)
            verdicts.append((f"model {model}", rc == 0 and same,
                             f"rc={rc} {secs}s sha={sha}" + ("" if model not in expect else
                                                              f" expected={expect[model]}")))
    if "sweep" in steps and a.sweep_model:
        empty = out / "replay_cache_empty"
        empty.mkdir(exist_ok=True)
        if any(empty.iterdir()):
            raise SystemExit(f"{empty}: not empty — a sweep from an earlier proof would be replayed, not swept")
        rc, sha, secs, log = _model_run(a.python, tree, out, a.sweep_model, "sweep",
                                        {"NBX_AUTOTUNE_CERTIFIED": "off", "NEUROBRIX_REPLAY_CACHE": str(empty)})
        m = re.search(r"(\d+) swept at runtime", log.read_text())
        swept = int(m.group(1)) if m else 0
        verdicts.append((f"sweep {a.sweep_model}", rc == 0 and swept > 0, f"rc={rc} {secs}s swept={swept} sha={sha}"))
    for i, cmd in enumerate(a.command):
        log = out / f"command{i}.log"
        rc = _run(["/bin/sh", "-c", cmd], tree, _env(tree, out, {"PATH": f"{Path(a.python).parent}{os.pathsep}"
                                                                         f"{os.environ.get('PATH', '')}"}), log)
        verdicts.append((f"command {cmd}", rc == 0, f"rc={rc}"))

    tries = [json.loads(l) for l in attempts.read_text().splitlines() if l.strip()]
    sites = {}
    for t in tries:
        sites.setdefault(t["site"], set()).add(t["module"])
    lines = [f"{'PASS' if ok else 'RED '} {name}: {what}" for name, ok, what in verdicts]
    lines.append(f"{'PASS' if not tries else 'RED '} torch import attempts: {len(tries)}"
                 + "".join(f"\n     {site}  ({', '.join(sorted(mods))})" for site, mods in sorted(sites.items())))
    report = "\n".join(lines)
    (out / "verdict.txt").write_text(report + "\n")
    print(report)
    return 0 if all(ok for _, ok, _ in verdicts) and not tries else 1


if __name__ == "__main__":
    raise SystemExit(main())
