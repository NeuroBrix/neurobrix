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

THE OPEN RED. A fourth request per branch, the first question with NO system message — the template renders TODAY's
date — is reported apart ("open red: the <dd/mm> prompt, the branches part at a near-tie"). It never gates.

    gate_cell_same_request.py --model <cache entry> [--cache <dir holding the entry>] [--date 2026-11-23]
                              [--branches compiled,triton] [--hardware <profile>] [--out <dir>] [--python <engine python>] [--lib-src <library src>]

Exit 0 when every gating request is byte-equal across the branches; 1 otherwise (the open red never changes it);
2 when the template's date cannot be pinned. Metal (the Mac) is NOT the gate's card: a run here is a probe.
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
QUESTIONS = ("Name a planet of the solar system.", "How far is it from the Sun, roughly?", "Name one of its moons, if it has any.")
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


def bodies(model: str, system: str | None) -> list:
    """The three gating request bodies, in order: the model and its messages, nothing else (the container's greedy
    defaults apply); each carries its own system message when `system` is given."""
    lead = [{"role": "system", "content": system}] if system is not None else []
    return [{"model": model, "messages": lead + [{"role": "user", "content": q}]} for q in QUESTIONS]


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
    for shared in ("hardware",):                                    # the machine's hardware profiles, read only
        if (atelier.root("engine_home") / shared).is_dir():
            (home / shared).symlink_to(atelier.root("engine_home") / shared)
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


def compare(name: str, by_branch: dict) -> tuple[bool, str]:
    """(equal, the line) of one request across the branches: byte-equal answers, none of them an error."""
    got = {b: r for b, r in by_branch.items() if r.get("text") is not None and not r.get("error")}
    if len(got) != len(by_branch) or len(got) < 2:
        return False, f"{name:<9}compare   NOT COMPARED: " + ", ".join(b for b in by_branch if b not in got) + " gave no answer"
    shas = {b: sha(r["text"]) for b, r in got.items()}
    if len(set(shas.values())) == 1:
        return True, f"{name:<9}compare   SAME  ({'='.join(by_branch)}) byte for byte"
    first, second = [r["text"].encode() for r in list(got.values())[:2]]
    at = next((i for i, (x, y) in enumerate(zip(first, second)) if x != y), min(len(first), len(second)))
    return False, f"{name:<9}compare   DIFFER " + " / ".join(f"{b} {s}" for b, s in shas.items()) + f", first byte apart at {at}"


def main(argv=None) -> int:
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
    out = atelier.tmp("gate_cell_same_request", f"{a.model}_{time.strftime('%Y%m%d_%H%M%S')}") if not a.out else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    today = _date.today()
    print(f"{a.model}: date pinned to {a.date} through the template's default system message "
          f"{system!r}" if system is not None else f"{a.model}: the template renders no date; no system message sent")
    gating = bodies(served_name(cache, a.model), system)
    named = [(f"req{k}", b) for k, b in enumerate(gating, 1)] + [("open_red", bodies(served_name(cache, a.model), None)[0])]
    answers = {b: serve_and_ask(b, a.model, cache, a, named, out) for b in branches}
    ok = True
    for name, _ in named[:-1]:
        for b in branches:
            print(line(name, b, answers[b][name]))
        same, text = compare(name, {b: answers[b][name] for b in branches})
        ok &= same
        print(text)
    if system is not None:
        print(f"open red: the {today:%d/%m} prompt, the branches part at a near-tie (never gates)")
        for b in branches:
            print("  " + line("open_red", b, answers[b]["open_red"]))
        print("  " + compare("open_red", {b: answers[b]["open_red"] for b in branches})[1])
    print(f"verdict {'GREEN' if ok else 'RED'}: {len(gating)} gating requests, {len(branches)} branches, date {a.date}  "
          f"(Metal is not the gate's card; logs in {out})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
