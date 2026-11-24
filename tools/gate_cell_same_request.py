#!/usr/bin/env python3
"""gate_cell_same_request.py — the gate cell that serves one model on each branch and sends the SAME requests to both.

Each branch (`compiled`, the PyTorch branch; `triton`, the Triton branch) is served through the engine's entry point
(`python -m neurobrix serve --http`) in a FRESH home of its own (its own engine home, its own owned compilation and
replay caches, a cache holding only the model under test), then asked three one-turn greedy requests: the request
body carries the model and the messages and NOTHING else, so the container's own greedy defaults apply. The answers
of the two branches are compared byte for byte. One line per request and branch, one line per comparison.

THE DATE. A chat template may render today's date into the prompt; two requests a day apart then differ in their
first tokens and the branches part at a near-tie that has nothing to do with them. The date is pinned through the
VENDOR TEMPLATE'S OWN DOOR: each gating request sends its own system message, the template's DEFAULT system message
with the date written out (`--date`, default 2026-11-23). The default message is READ FROM THE TEMPLATE (the
container's `modules/tokenizer/tokenizer_config.json`), never copied here: the template is rendered twice with the
engine's own chat-template environment, once with no system message (the default) and once with a probe in the
system slot, and the text the default puts where the probe goes is the message. A template that writes the date
elsewhere than in that message (a Llama-3 `date_string`) cannot be pinned this way and is REFUSED by name.

THE OPEN RED. A last request per branch, the data file's open red question with NO system message — the template
renders TODAY's date — is reported apart ("open red: the <dd/mm> prompt, the branches part at a near-tie"). It never gates.

    gate_cell_same_request.py --model <cache entry> [--cache <dir holding the entry>] [--date 2026-11-23]
                              [--branches compiled,triton] [--hardware <profile>] [--out <dir>] [--python <engine python>] [--lib-src <library src>]

THE QUESTIONS live in ONE data file beside this tool, `gate_cell_questions.json`: the gating set and the "open,
measured" list, each entry with its text, its measured minimum margin, the position and token of that minimum, the
card class and the date of the measure. The run sends the gating entries in order and NEVER gates an entry whose
status is OPEN, wherever it sits; the open red is the open entry marked `"pinned": false`.

THE CHECK (`--check-question "<text>"`, run once when a question is added). A gate question must be one where the
model CHOOSES at every step: a position where the top two logits nearly tie is decided by the dtype's rounding, and
two correct branches may part there. The check takes the VENDOR's own top-2 margin at EVERY decoded position of its
greedy answer: transformers, the same rendered prompt with the pinned date (the template's default system message,
as the run sends it), the container's dtype, the logits the vendor model returns (after its own scaling, granite's
`logits_scaling`). It runs in a worker process of `--vendor-python`, offline, in a fresh HF home, reading no token
or credential file; `--vendor-extra-site` lends only the packages that python lacks (never torchvision). It prints
the minimum margin, its position and token, and the verdict: GATE when the minimum is at least the floor, else OPEN.

THE FLOOR (`--floor`, default 0.5 vendor logit units). granite's top logits sit near ~24, where one bf16 step is
0.125; the ties measured on 23-24/11 were 0 to 0.125 (0 to 1 bf16 step) and the clear positions about 2.8 to 3.75.
0.5 is 4 bf16 steps: above any rounding of one or two steps, far below a clear choice.

    gate_cell_same_request.py --model <cache entry> --check-question "<text>" [--floor 0.5] [--date 2026-11-23]
                              --hf-snapshot <the vendor snapshot> [--vendor-python <py>] [--vendor-extra-site <site>]

Exit 0 when every gating request is byte-equal across the branches; 1 otherwise (the open red never changes it);
2 when the template's date cannot be pinned. The check exits 0 on GATE, 1 on OPEN. Metal (the Mac) is NOT the gate's
card: a run or a check here is a probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import date as _date, datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import atelier  # noqa: E402 — the workshop's roots
from check_measurement_environment import owned_cache_env  # noqa: E402
from one_name import cli_name  # noqa: E402

REPO_SRC = TOOLS.parent / "src"
DEFAULT_DATE = "2026-11-23"
FLAGS = {"compiled": ["--compiled"], "triton": ["--triton"]}
QUESTIONS_FILE = TOOLS / "gate_cell_questions.json"
DEFAULT_FLOOR = 0.5
# packages the vendor worker may take from --vendor-extra-site when its python lacks them (never torchvision)
VENDOR_PKGS = ("transformers", "tokenizers", "huggingface_hub", "hf_xet", "regex", "tqdm", "yaml", "_yaml",
               "safetensors", "filelock", "fsspec", "packaging", "requests", "urllib3", "idna", "certifi",
               "charset_normalizer", "typing_extensions", "numpy", "jinja2", "markupsafe", "httpx", "httpcore",
               "anyio", "sniffio", "h11", "click", "typer", "shellingham", "rich", "pygments", "markdown_it", "mdurl",
               "annotated_doc")
PROBE = "\x00system-slot-probe\x00"
REFUSAL = "this template's date cannot be pinned through a system message"


class DateCannotBePinned(RuntimeError):
    """The template writes the date outside its default system message: no system message can fix it."""


def environment(pinned: datetime | None):
    """The engine's own chat-template environment, `strftime_now` answering `pinned` when one is given."""
    sys.path.insert(0, str(REPO_SRC))
    from neurobrix.core.module.tokenizer.chat_template import environment as engine_environment
    env = engine_environment()
    if pinned is not None:
        env.globals["strftime_now"] = lambda fmt: pinned.strftime(fmt)
    return env


def render(template: str, messages: list, pinned: datetime | None = None) -> str:
    """The prompt `template` renders for `messages` at the generation prompt, the date pinned to `pinned` (None: now)."""
    return environment(pinned).from_string(template).render(
        messages=messages, bos_token="", eos_token="", add_generation_prompt=True)


def default_system_message(template: str, when: datetime) -> str | None:
    """The system message the template writes for itself when a conversation has none, with its date at `when` — read
    from the template: None when the template renders no date at all, DateCannotBePinned when the date sits outside
    that message. The text is what the default render puts where a probe system message goes in the explicit render."""
    user = [{"role": "user", "content": "q"}]
    other = datetime(when.year - 1, 1 if when.month != 1 else 2, 1)
    explicit = [{"role": "system", "content": PROBE}] + user
    default_now, default_other = render(template, user, when), render(template, user, other)
    if render(template, explicit, when) != render(template, explicit, other):
        raise DateCannotBePinned(REFUSAL)                           # the date is written around the system slot
    if default_now == default_other:
        return None                                                # no date reaches the prompt
    with_probe = render(template, explicit, when)
    head, tail = with_probe.split(PROBE)
    if not (default_now.startswith(head) and default_now.endswith(tail)) or len(default_now) < len(head) + len(tail):
        raise DateCannotBePinned(REFUSAL)                           # the default is not the explicit one with a message
    return default_now[len(head):len(default_now) - len(tail)]


def template_of(container: Path) -> str:
    """The container's chat template text (`modules/tokenizer/tokenizer_config.json`), refused by name when it holds none."""
    p = container / "modules" / "tokenizer" / "tokenizer_config.json"
    template = json.loads(p.read_text()).get("chat_template")
    if not isinstance(template, str) or not template:
        raise SystemExit(f"{p}: no single chat_template string; no date to pin")
    return template


def served_name(cache: Path, entry: str) -> str:
    """The model's one name for the command line: the name the import recorded beside the entry (`name.json`), else the
    catalogue's (`one_name.cli_name`)."""
    try:
        return json.loads((cache / entry / "name.json").read_text())["name"]
    except (OSError, KeyError, ValueError):
        return cli_name(entry)


def read_questions(path: Path = QUESTIONS_FILE) -> dict:
    """The questions data file: its `gating` and `open_measured` lists, every entry with a text and a status (GATE or
    OPEN); refused by name when an entry lacks either."""
    data = json.loads(Path(path).read_text())
    for kind in ("gating", "open_measured"):
        for e in data.get(kind, []):
            if not isinstance(e.get("text"), str) or e.get("status") not in ("GATE", "OPEN"):
                raise SystemExit(f"{path}: an entry of {kind!r} lacks its text or a GATE/OPEN status: {e!r}")
    return data


def gating_questions(data: dict) -> list:
    """The texts the run gates on, in order: the `gating` entries, an OPEN one never among them."""
    return [e["text"] for e in data.get("gating", []) if e["status"] == "GATE"]


def open_red_question(data: dict) -> str | None:
    """The text of the open red: the first open entry sent with no system message (`"pinned": false`), else None."""
    return next((e["text"] for e in data.get("open_measured", []) if e.get("pinned") is False), None)


def bodies(model: str, system: str | None, questions: list) -> list:
    """The gating request bodies, in order: the model and its messages, nothing else (the container's greedy
    defaults apply); each carries its own system message when `system` is given."""
    lead = [{"role": "system", "content": system}] if system is not None else []
    return [{"model": model, "messages": lead + [{"role": "user", "content": q}]} for q in questions]


# ---------------------------------------------------------------- the check: the vendor's top-2 margin, every step

def top2(row) -> tuple:
    """((id, logit), (id, logit)) of the two largest logits of `row`, the lower id first on a tie (the greedy pick)."""
    import heapq
    (i1, v1), (i2, v2) = heapq.nlargest(2, enumerate(row), key=lambda iv: iv[1])
    return (i1, float(v1)), (i2, float(v2))


def min_margin(steps: list) -> tuple:
    """(margin, position) of the smallest top-1 minus top-2 logit over the decoded `steps` (each a top2 pair);
    position 0 is the first decoded token, the first minimum wins."""
    margins = [a[1] - b[1] for a, b in steps]
    k = min(range(len(margins)), key=margins.__getitem__)
    return margins[k], k


def verdict(margin: float, floor: float) -> str:
    """GATE when the question's minimum margin is at least `floor`, OPEN otherwise."""
    return "GATE" if margin >= floor else "OPEN"


def vendor_worker(job_path: str) -> None:
    """The vendor's greedy answer with its top-2 at every step: transformers, offline, the job's dtype and device, the
    logits the model returns (after its own scaling); one forward per token on the model's own KV cache."""
    job = json.loads(Path(job_path).read_text())
    if job.get("shim"):
        sys.path.append(job["shim"])                                  # the python's own packages win
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = job["device"]
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(job["snapshot"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(job["snapshot"], dtype=getattr(torch, job["dtype"]),
                                                 local_files_only=True).to(dev).eval()
    ids = tok(job["text"], add_special_tokens=True)["input_ids"]
    steps, past, inp = [], None, torch.tensor([ids], device=dev)
    with torch.inference_mode():
        for _ in range(job["max_new"]):
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            pair = top2(out.logits[0, -1].float().cpu().tolist())
            steps.append([[pair[0][0], pair[0][1], tok.decode([pair[0][0]])], [pair[1][0], pair[1][1], tok.decode([pair[1][0]])]])
            if pair[0][0] in job["eos"]:
                break
            inp = torch.tensor([[pair[0][0]]], device=dev)
    card = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    Path(job["out"]).write_text(json.dumps({"transformers": transformers.__version__, "torch": torch.__version__,
                                            "device": dev, "card": card, "prompt_tokens": len(ids), "steps": steps}))


def vendor_shim(vendor_python: str, extra_site: str | None, home: Path) -> str | None:
    """A directory of links to only the packages `vendor_python` lacks, taken from `extra_site` (the worker appends it
    at the END of its path, so the python's own packages win; torchvision is never linked)."""
    if not extra_site:
        return None
    src, shim = Path(extra_site), home / "vendor_site"
    shim.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([vendor_python, "-c", "import importlib.util,sys,json;print(json.dumps("
                            "[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]))", *VENDOR_PKGS],
                           capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    missing = json.loads(probe.stdout.strip() or "[]")
    for m in missing:
        dists = [f"{m.replace('_', '?')}-*.dist-info"] + (["pyyaml-*.dist-info"] if m == "yaml" else [])
        for cand in [*src.glob(m), *src.glob(m + ".py"), *src.glob(m + ".*.so"), *(c for d in dists for c in src.glob(d))]:
            if not (shim / cand.name).exists():
                (shim / cand.name).symlink_to(cand)
    return str(shim)


def vendor_steps(a, text: str, dtype: str, eos: list, max_new: int) -> dict:
    """Run the vendor worker on the rendered `text` in a fresh home: offline, a fresh HF home, no token read."""
    home = atelier.tmp(f"gcsr_check_{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}")
    for d in ("tmp", "hf", "xdg"):
        (home / d).mkdir(parents=True, exist_ok=True)
    job = {"snapshot": a.hf_snapshot, "device": a.vendor_device, "dtype": dtype, "text": text, "eos": eos,
           "max_new": max_new, "out": str(home / "vendor.json"), "shim": vendor_shim(a.vendor_python, a.vendor_extra_site, home)}
    (home / "job.json").write_text(json.dumps(job))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "HUGGING", "TRITON_"))}
    env.update(PYTHONNOUSERSITE="1", PYTHONPATH="", HF_HOME=str(home / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TMPDIR=str(home / "tmp"), XDG_CACHE_HOME=str(home / "xdg"),
               TRITON_CACHE_DIR=str(home / "tmp" / "triton"))
    with open(home / "err.txt", "w") as fe:
        rc = subprocess.call([a.vendor_python, str(Path(__file__).resolve()), "--vendor-worker", str(home / "job.json")],
                             env=env, cwd=str(home), stdout=fe, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"the vendor worker failed rc={rc}:\n" + (home / "err.txt").read_text()[-1500:])
    return {**json.loads((home / "vendor.json").read_text()), "home": str(home)}


def check_question(a, cache: Path, system: str | None) -> int:
    """--check-question: the vendor's top-2 margin at every decoded position of its greedy answer to the question, the
    prompt rendered as the run sends it; prints the minimum, its position and token, the verdict and one measure line."""
    if not a.hf_snapshot or not (Path(a.hf_snapshot) / "config.json").exists():
        raise SystemExit(f"--hf-snapshot {a.hf_snapshot!r}: no vendor snapshot (config.json) there")
    defaults = json.loads((cache / a.model / "runtime" / "defaults.json").read_text())
    lead = [{"role": "system", "content": system}] if system is not None else []
    text = render(template_of(cache / a.model), lead + [{"role": "user", "content": a.check_question}])
    c = cache / a.model
    eos = container_value(defaults, "eos_token_id", c)
    eos = list(eos) if isinstance(eos, list) else [eos]
    dtype = container_value(defaults, "dtype", c)
    max_new = a.max_tokens or int(container_value(defaults, "max_tokens", c))
    v = vendor_steps(a, text, dtype, eos, max_new)
    steps = [(tuple(s[0][:2]), tuple(s[1][:2])) for s in v["steps"]]
    m, k = min_margin(steps)
    said = verdict(m, a.floor)
    with open(Path(v["home"]) / "steps.tsv", "w") as f:
        f.write("position\ttop1_id\ttop1_text\ttop1_logit\ttop2_id\ttop2_text\ttop2_logit\tmargin\n")
        for i, (t1, t2) in enumerate(v["steps"]):
            f.write(f"{i}\t{t1[0]}\t{t1[2]!r}\t{t1[1]}\t{t2[0]}\t{t2[2]!r}\t{t2[1]}\t{t1[1] - t2[1]}\n")
    low = sorted(range(len(steps)), key=lambda i: steps[i][0][1] - steps[i][1][1])[:3]
    answer = "".join(s[0][2] for s in v["steps"])
    print(f"check {a.check_question!r}: vendor transformers {v['transformers']} torch {v['torch']} on {v['card']}, "
          f"{dtype}, date {a.date}, {v['prompt_tokens']} prompt tokens, {len(steps)} decoded")
    print(f"  answer {answer[:120]!r}")
    for i in low:
        print(f"  position {i:<4} margin {steps[i][0][1] - steps[i][1][1]:.4f}  top1 {v['steps'][i][0][2]!r} "
              f"{steps[i][0][1]:.4f}  top2 {v['steps'][i][1][2]!r} {steps[i][1][1]:.4f}")
    print(f"  minimum margin {m:.4f} at position {k} token {v['steps'][k][0][2]!r} (vs {v['steps'][k][1][2]!r}): "
          f"{said} (floor {a.floor})  steps in {v['home']}/steps.tsv")
    print("  measure " + json.dumps({"min_margin": round(m, 4), "position": k, "token": v["steps"][k][0][2],
                                     "runner_up": v["steps"][k][1][2], "decoded": len(steps), "verdict": said,
                                     "floor": a.floor, "date_rendered": a.date, "model": a.model, "dtype": dtype,
                                     "vendor": f"transformers {v['transformers']} torch {v['torch']} {v['card']}",
                                     "measured": _date.today().isoformat()}))
    return 0 if said == "GATE" else 1


def sha(text: str) -> str:
    """The first sixteen hex digits of the sha256 of `text`: how an answer is named in a line."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def free_port() -> int:
    """A loopback port nobody holds at this moment."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, body: dict, budget: float) -> dict:
    """One chat completion over HTTP; the answer's text, finish reason, token count and seconds, or its error."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=budget) as resp:
            r = json.loads(resp.read())
        c = r["choices"][0]
        return {"text": (c["message"].get("content") or ""), "finish": c.get("finish_reason"),
                "tokens": (r.get("usage") or {}).get("completion_tokens"), "seconds": round(time.time() - t0, 1)}
    except Exception as exc:                                       # noqa: BLE001 — the line names it
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def serve_and_ask(branch: str, entry: str, cache: Path, a, requests: list, out: Path) -> dict:
    """Serve `entry` on `branch` in a fresh home, send `requests` (name, body) in order, stop the daemon: name -> result."""
    home = atelier.tmp(f"gcsr_{os.getpid()}_{branch}")
    shutil.rmtree(home, ignore_errors=True)
    (home / "cache").mkdir(parents=True)
    (home / "cache" / entry).symlink_to((cache / entry).resolve())
    if (atelier.root("engine_home") / "hardware").is_dir():         # the machine's hardware profiles: a read-only COPY
        read_only_copy(atelier.root("engine_home") / "hardware", home / "hardware")
    sock, ready, log = atelier.tmp(f"gcsr_{os.getpid()}.sock"), home / "ready.json", out / f"{branch}.serve.log"
    env = {**os.environ, **owned_cache_env(home), "NEUROBRIX_HOME": str(home), "NEUROBRIX_CACHE": str(home / "cache"),
           "PYTHONPATH": os.pathsep.join([str(REPO_SRC)] + ([str(Path(a.lib_src).resolve())] if a.lib_src else [])),
           "PYTHONNOUSERSITE": "1", "NBX_SOCKET_PATH": str(sock),
           "NEUROBRIX_PYTHON": a.python}
    port = free_port()
    cmd = [a.python, "-m", "neurobrix", "serve", "--model", served_name(cache, entry), "--foreground", "--timeout", "7200",
           "--http", "--http-host", "127.0.0.1", "--http-port", str(port), "--ready-file", str(ready), *FLAGS[branch]]
    if a.hardware:
        cmd += ["--hardware", a.hardware]
    results, t0 = {}, time.time()
    with open(log, "w") as lf:
        daemon = subprocess.Popen(cmd, env=env, cwd=str(REPO_SRC.parent), stdout=lf, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while not (ready.exists() and ready.stat().st_size) and daemon.poll() is None and time.time() - t0 < a.ready_timeout:
            time.sleep(2)
        if not (ready.exists() and ready.stat().st_size):
            why = f"exited rc={daemon.returncode}" if daemon.poll() is not None else f"not ready in {a.ready_timeout} s"
            return {name: {"error": f"the {branch} daemon {why} (log: {log})"} for name, _ in requests}
        time.sleep(0.5)
        rec = json.loads(ready.read_text().strip().splitlines()[-1])
        if rec.get("state") != "ready":
            return {name: {"error": f"the {branch} daemon is not ready: {str(rec.get('error'))[:160]}"} for name, _ in requests}
        for name, body in requests:
            results[name] = post(rec["http"]["port"], body, 1800)
            (out / f"{branch}.{name}.json").write_text(json.dumps({"request": body, "result": results[name]}, indent=1))
        return results
    finally:
        if daemon.poll() is None:
            os.killpg(daemon.pid, signal.SIGTERM)
            try:
                daemon.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(daemon.pid, signal.SIGKILL)
        for leftover in sock.parent.glob(sock.stem + ".*"):
            leftover.unlink(missing_ok=True)
        shutil.rmtree(home, ignore_errors=True)


def line(name: str, branch: str, r: dict) -> str:
    """The one line of a request on a branch: its answer's sha and its numbers, or its error."""
    if r.get("error"):
        return f"{name:<9}{branch:<9}ERROR {r['error']}"
    return (f"{name:<9}{branch:<9}sha {sha(r['text'])}  finish {r['finish']}  {r['tokens']} tok  {r['seconds']} s  "
            f"{r['text'][:60]!r}")


ANSWERED = ("stop", "length")


def unusable(r: dict) -> str | None:
    """Why one branch's answer to a request cannot be compared, or None when it is a real answer: an error, no text, an
    EMPTY text (a whitespace-only one is empty: two would compare SAME byte for byte), or a finish other than
    stop/length (a fault both branches share would otherwise compare SAME)."""
    if r.get("error"):
        return f"gave no answer: {r['error']}"
    text, finish = r.get("text"), r.get("finish")
    if text is None or not str(text).strip() or finish not in ANSWERED:
        return f"gave no usable answer: text {str(text)[:60]!r}, finish {finish!r}"
    return None


def compare(name: str, by_branch: dict) -> tuple[bool, str]:
    """(equal, the line) of one request across the branches: byte-equal answers; an error, an empty text or a finish
    other than stop/length is NOT COMPARED, named by branch and request, and never equal."""
    bad = {b: unusable(r) for b, r in by_branch.items()}
    bad = {b: why for b, why in bad.items() if why}
    if bad:
        return False, f"{name:<9}compare   NOT COMPARED: " + "; ".join(f"{b} ({name}) {why}" for b, why in bad.items())
    if len(by_branch) < 2:
        return False, (f"{name:<9}compare   NOT COMPARED: one branch only ({', '.join(by_branch)}) answered {name}; "
                       f"nothing to compare it with")
    got = by_branch
    shas = {b: sha(r["text"]) for b, r in got.items()}
    if len(set(shas.values())) == 1:
        return True, f"{name:<9}compare   SAME  ({'='.join(by_branch)}) byte for byte"
    first, second = [r["text"].encode() for r in list(got.values())[:2]]
    at = next((i for i, (x, y) in enumerate(zip(first, second)) if x != y), min(len(first), len(second)))
    return False, f"{name:<9}compare   DIFFER " + " / ".join(f"{b} {s}" for b, s in shas.items()) + f", first byte apart at {at}"


def container_value(defaults: dict, key: str, container: Path):
    """The value `key` of the container's runtime/defaults.json, REFUSED by name when the container states none: a gate
    tool never supplies a default of its own."""
    v = defaults.get(key)
    if v is None or v == [] or v == "":
        raise SystemExit(f"{container}/runtime/defaults.json states no {key!r}; the gate tool does not default it")
    return v


def read_only_copy(src: Path, dst: Path) -> None:
    """Copy the directory `src` to `dst` with every file in read-only mode (0444): the run reads the hardware profiles,
    and neither it nor the original can be written through the copy."""
    shutil.copytree(src, dst)
    for f in dst.rglob("*"):
        if f.is_file() and not f.is_symlink():
            f.chmod(0o444)


def main(argv=None) -> int:
    """The gate cell's run (both branches, the gating questions of the data file) or, with --check-question, the check."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--vendor-worker"]:
        vendor_worker(argv[1])
        return 0
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="the cache entry (the container's directory name)")
    p.add_argument("--cache", default=None, help="the directory holding the entry (default: the workshop's `cache` root)")
    p.add_argument("--date", default=DEFAULT_DATE, help="the date the template renders, YYYY-MM-DD (default %(default)s)")
    p.add_argument("--branches", default="compiled,triton", help="comma list among: " + ", ".join(FLAGS))
    p.add_argument("--hardware", default=None, help="hardware profile id (default: auto-detected)")
    p.add_argument("--out", default=None, help="where logs and answers are kept (default: a folder of the workshop's tmp root)")
    p.add_argument("--python", default=None, help="the engine's interpreter (default: the workshop's `engine`)")
    p.add_argument("--lib-src", default=None, help="the library's src tree, when the interpreter does not import the pinned one")
    p.add_argument("--ready-timeout", type=int, default=3600)
    p.add_argument("--questions", default=str(QUESTIONS_FILE), help="the questions data file (default: the one beside the tool)")
    p.add_argument("--check-question", default=None, help="check this question's vendor top-2 margin at every step; no run")
    p.add_argument("--floor", type=float, default=DEFAULT_FLOOR, help="the check's floor, vendor logit units (default %(default)s)")
    p.add_argument("--hf-snapshot", default=None, help="the check: the vendor's snapshot directory (default: the workshop's "
                   "`hf_snapshots` root / the entry)")
    p.add_argument("--vendor-python", default=None, help="the check: a python with torch (default: --python)")
    p.add_argument("--vendor-extra-site", default=None, help="the check: a site-packages lending ONLY what that python lacks")
    p.add_argument("--vendor-device", default="auto", help="the check: cuda, mps, cpu or auto")
    p.add_argument("--max-tokens", type=int, default=None, help="the check: decoded tokens at most (default: the container's)")
    a = p.parse_args(argv)
    a.python = a.python or atelier.python("engine")
    if a.lib_src:                                                   # the tool imports the engine's template environment
        sys.path.insert(0, str(Path(a.lib_src).resolve()))
    branches = [b for b in a.branches.split(",") if b]
    unknown = [b for b in branches if b not in FLAGS]
    if unknown or not branches:
        raise SystemExit(f"--branches {a.branches!r}: choose among {', '.join(FLAGS)}")
    cache = Path(a.cache) if a.cache else atelier.root("cache")
    pinned = datetime.fromisoformat(a.date)
    try:
        system = default_system_message(template_of(cache / a.model), pinned)
    except DateCannotBePinned as e:
        print(f"{a.model}: REFUSED — {e}")
        return 2
    if a.check_question is not None:
        a.vendor_python = a.vendor_python or a.python
        if a.hf_snapshot is None:
            a.hf_snapshot = str(atelier.root("hf_snapshots") / a.model)
        return check_question(a, cache, system)
    questions = read_questions(Path(a.questions))
    out = atelier.tmp("gate_cell_same_request", f"{a.model}_{time.strftime('%Y%m%d_%H%M%S')}") if not a.out else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    today = _date.today()
    print(f"{a.model}: date pinned to {a.date} through the template's default system message "
          f"{system!r}" if system is not None else f"{a.model}: the template renders no date; no system message sent")
    gating = bodies(served_name(cache, a.model), system, gating_questions(questions))
    red = open_red_question(questions) if system is not None else None
    named = [(f"req{k}", b) for k, b in enumerate(gating, 1)]
    if red is not None:
        named.append(("open_red", bodies(served_name(cache, a.model), None, [red])[0]))
    answers = {b: serve_and_ask(b, a.model, cache, a, named, out) for b in branches}
    ok = bool(gating)
    for name, _ in named[:len(gating)]:
        for b in branches:
            print(line(name, b, answers[b][name]))
        same, text = compare(name, {b: answers[b][name] for b in branches})
        ok &= same
        print(text)
    if red is not None:
        print(f"open red: the {today:%d/%m} prompt, the branches part at a near-tie (never gates)")
        for b in branches:
            print("  " + line("open_red", b, answers[b]["open_red"]))
        print("  " + compare("open_red", {b: answers[b]["open_red"] for b in branches})[1])
    print(f"verdict {'GREEN' if ok else 'RED'}: {len(gating)} gating requests, {len(branches)} branches, date {a.date}  "
          f"(Metal is not the gate's card; logs in {out})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
