#!/usr/bin/env python3
"""gate_serve_cell.py — the gate's SERVE cell: one daemon, three successive requests, the third at the first's speed.

A run cell (`regression_matrix.py run`) launches one process per request: what a daemon keeps between two requests
never meets it. Two defects of 2026-11-11 lived exactly there — a served mixture-of-experts model answered its first
request at full speed and every later one two hundred times slower, then twice slower — and no cell could see them.

CONTRACT
  * the daemon is started through the engine's entry point (`python -m neurobrix serve`, the family's engine flag,
    the hardware profile given), on an instance name of its own (`NBX_SOCKET_PATH`, in the workshop's tmp root) and a loopback port:
    it can neither answer nor be answered by a daemon serving someone else;
  * three identical streamed chat requests (temperature 0), one after the other, to that one daemon: request 1 FRESH,
    requests 2 and 3 REPEATED (the daemon's prefix cache answers their prompt), then, after the conversation and its
    oracle, a FOURTH identical request, FRESH AFTER A RESET: the daemon offers no request that empties its cache, so
    the reset is a restart of the same daemon, done by this script (`RESET`);
  * per request: the time to the first chunk, the chunks and their pieces, the median time between two chunks, the
    sha256 of the text, and `cached_tokens`, the prompt tokens the daemon says it answered from its cache (asked with
    `stream_options.include_usage`, read from `usage.prompt_tokens_details`);
  * RED when the daemon does not come up, a request fails or returns no text, or the THIRD request's median time
    between chunks is more than `SLOWER_IS_RED` times the FIRST's (the gate's speed rule, the same factor as a cell
    against its base);
  * the CACHE judgment (`judge_cache`): the repeated requests and the fresh-after-reset one must be byte-equal to the
    fresh one. A difference is RED and names the first differing piece (one streamed piece is one decoded token's
    text) with both pieces; the logit margin is not given: the daemon does not return log-probabilities. RED too when
    the repeated request reused no cached token (the cache was not exercised) or the request after the restart reused
    some (the reset did not empty it). NEEDS CARD when the daemon did not report `cached_tokens`: the number is never
    invented. A cell listed in the `--known` file (a JSON object, cell -> its proof file) answers RED-KNOWN for a
    difference and does not fail the run;
  * a five-turn CONVERSATION on the same daemon, each request adding the last answer and a new user turn (a multi-row
    continuation after a decode, an agent's every turn), then the same conversation on a second daemon whose replay is
    off (`serve_without_replay.py`, the oracle): RED when a turn answers nothing or not the oracle's text. The replay
    is built to give the direct path's bytes; on 2026-11-12 a served MoE answered turns 3-5 EMPTY with it, right
    without it, and no three identical requests could see it;
  * one JSON row appended to <out>/serve_rows.jsonl whatever happens (verdict, reason, the four requests, the cache
    judgment, the load time, the cards' peak memory read from outside), the daemon stopped by its pid, its log kept
    beside the row; a table of at most ten lines printed and written to <out>/<cell>.table.txt. Exit 0 on GREEN or
    RED-KNOWN, 1 otherwise.

    gate_serve_cell.py --model <hf name> --gpu <cards> --hardware <profile> --mode triton|native --src <tree>/src
                       --out <dir> [--known <known.json>] [--port 8090] [--max-tokens 60] [--ready-timeout 900]
                       [--python <engine python>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atelier  # noqa: E402 — the workshop's roots

SLOWER_IS_RED = 1.5          # the gate's rule: slower than its base by more than this is a speed regression
MODES = {"native": [], "triton": ["--triton"]}
RESET = "a restart of the same daemon: it offers no request that empties its prefix cache"
USAGE_AT = "src/neurobrix/serving/openai_api/routes.py, `_usage` (usage.prompt_tokens_details.cached_tokens)"
PROMPT = "Explain what a hash table is, how collisions are handled, and give a short Python example."


FOLLOW_UPS = ("Say it in one sentence.", "Now give a Python example.", "What is its lookup cost?", "Name one drawback.")


def judge_conversation(on: list, off: list) -> tuple:
    """(verdict, reason) of the conversation's turns with the replay (`on`) against the oracle without it (`off`) —
    pure, tested without a card."""
    if len(on) != len(FOLLOW_UPS) + 1 or len(off) != len(on):
        return "RED", f"the conversation answered {len(on)} turn(s) with the replay, {len(off)} without, of {len(FOLLOW_UPS) + 1}"
    for i, (a, b) in enumerate(zip(on, off), 1):
        if not a:
            return "RED", f"turn {i} of the conversation answered nothing with the replay"
        if a != b:
            return "RED", f"turn {i} of the conversation differs from the replay-off oracle"
    return "GREEN", f"the {len(on)} turns of the conversation equal the replay-off oracle's"


def spoken(part: dict) -> str:
    """What a chat message or a stream delta says: its reasoning (`reasoning_content`, a thinking model's text between
    its markers) then its answer (`content`). A thinking model's first tokens are all reasoning: counted as `content`
    alone, Qwen3-30B-A3B-Thinking answered "0 chunks, no text to time" (2026-11-12)."""
    return (part.get("reasoning_content") or "") + (part.get("content") or "")


def converse(port: int, model: str, max_tokens: int, budget: float) -> list:
    """The conversation's answers, turn by turn (greedy, not streamed); a failed turn ends it."""
    msgs, texts = [{"role": "user", "content": PROMPT}], []
    for i in range(len(FOLLOW_UPS) + 1):
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": False, "messages": msgs}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=budget) as resp:
                message = json.loads(resp.read())["choices"][0]["message"]
                text = spoken(message)
        except Exception:                                      # noqa: BLE001 — the judgment names the short turn
            return texts
        texts.append(text)
        if i < len(FOLLOW_UPS):
            msgs = msgs + [{"role": "assistant", "content": message.get("content") or ""},
                           {"role": "user", "content": FOLLOW_UPS[i]}]
    return texts


def judge(requests: list) -> tuple:
    """(verdict, reason) of the three requests' records — pure, so that it is tested without a card."""
    if len(requests) != 3:
        return "RED", f"{len(requests)} request(s) answered of 3"
    for i, r in enumerate(requests, 1):
        if r.get("error"):
            return "RED", f"request {i} failed: {r['error']}"
        if not r.get("text_sha") or r.get("chunks", 0) < 3:
            return "RED", f"request {i} returned {r.get('chunks', 0)} chunk(s): no text to time"
    first, third = requests[0]["median_gap_s"], requests[2]["median_gap_s"]
    if first <= 0:
        return "RED", "request 1 has no measurable time between chunks"
    if third > SLOWER_IS_RED * first:
        return "RED", (f"request 3 runs at {third:.3f} s a chunk against {first:.3f} on request 1 "
                       f"({third / first:.2f}x, over {SLOWER_IS_RED}x)")
    return "GREEN", f"request 3 at {third:.3f} s a chunk against {first:.3f} on request 1 ({third / first:.2f}x)"


def first_difference(a: list, b: list):
    """None when two answers' pieces say the same bytes, else (position, piece of a, piece of b), a missing piece None."""
    if "".join(a) == "".join(b):
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    i = min(len(a), len(b))
    return i, (a[i] if i < len(a) else None), (b[i] if i < len(b) else None)


def judge_cache(cell: str, requests: list, after_reset: dict, known: dict) -> dict:
    """The cache judgment — pure, tested without a card: the repeated requests (2, 3) and the fresh-after-reset one (4)
    against the fresh one (1), byte for byte, with the cached tokens each reports."""
    asked = list(requests[:3]) + [after_reset or {}]
    cached = [r.get("cached_tokens") for r in asked]
    out = {"cached_tokens": cached, "diffs": [], "known_proof": known.get(cell)}
    for i, r in enumerate(asked, 1):
        if r.get("error") or not r.get("pieces"):
            return {**out, "verdict": "RED", "reason": f"request {i} gave no answer to compare: "
                                                       f"{r.get('error') or 'no text'}"}
    for i, r in enumerate(asked[1:], 2):
        d = first_difference(asked[0]["pieces"], r["pieces"])
        if d:
            out["diffs"].append({"request": i, "cached_tokens": cached[i - 1], "piece": d[0], "fresh": d[1],
                                 "this": d[2], "logit_margin": None})
    said = "; ".join(f"request {d['request']} ({d['cached_tokens']} cached tokens) differs from the fresh request 1 "
                     f"at piece {d['piece']}: {d['this']!r} against {d['fresh']!r}" for d in out["diffs"])
    if out["diffs"] and cell not in known:
        return {**out, "verdict": "RED", "reason": said}
    if None in cached:
        return {**out, "verdict": "NEEDS CARD", "reason": (
            f"the daemon did not report cached_tokens on request {cached.index(None) + 1}; it should in {USAGE_AT}")}
    if not cached[1]:
        return {**out, "verdict": "RED", "reason": "the repeated request 2 reused no cached token: the cache was not exercised"}
    if cached[3]:
        return {**out, "verdict": "RED", "reason": (f"the request after the reset reused {cached[3]} cached tokens: "
                                                    "the reset did not empty the cache")}
    if out["diffs"]:
        return {**out, "verdict": "RED-KNOWN", "reason": f"{said} (known, proof: {known[cell]})"}
    return {**out, "verdict": "GREEN", "reason": ("the repeated and the fresh-after-reset requests equal the fresh one, "
                                                  f"cached tokens {'/'.join(str(c) for c in cached)}")}


def passes(verdict: str) -> bool:
    """A run passes on GREEN, or on RED-KNOWN: a difference listed known with its proof."""
    return verdict in ("GREEN", "RED-KNOWN")


def read_known(path) -> dict:
    """The known cells, cell -> its proof file: a JSON object; an entry without an existing proof file is refused."""
    if not path:
        return {}
    known = json.loads(Path(path).read_text())
    if not isinstance(known, dict):
        raise SystemExit(f"--known {path}: a JSON object is expected, cell name -> the path of its proof file")
    for cell, proof in known.items():
        where = Path(path).parent / str(proof)
        if not isinstance(proof, str) or not proof or not where.is_file():
            raise SystemExit(f"--known {path}: the cell {cell!r} names no existing proof file ({proof!r}); "
                             "give the path, relative to the known file, of the file that proves the difference known")
    return known


def table(row: dict) -> str:
    """At most ten lines: the cell, the four requests, the three judgments, the verdict."""
    lines = [f"cell {row['cell']}  engine {row.get('engine')}  load {row.get('load_s')} s",
             f"{'request':<22}{'cached':>8}{'pieces':>8}{'gap s':>9}  sha"]
    named = ("1 fresh", "2 repeated", "3 repeated", "4 fresh after reset")
    for name, r in zip(named, list(row.get("requests") or []) + [row.get("after_reset") or {}]):
        lines.append(f"{name:<22}{str(r.get('cached_tokens')):>8}{r.get('chunks', 0):>8}{str(r.get('median_gap_s')):>9}"
                     f"  {r.get('text_sha') or r.get('error') or '-'}")
    lines += [f"speed:        {row.get('speed_verdict', '-')} {row.get('speed_reason', '')}",
              f"cache:        {row.get('cache_verdict', '-')} {row.get('cache_reason', '')}",
              f"conversation: {row.get('conversation_verdict', '-')} {row.get('conversation_reason', '')}",
              f"verdict:      {row.get('verdict')} {row.get('reason', '')}"]
    return "\n".join(lines[:10])


def ask(port: int, model: str, max_tokens: int, budget: float) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, times, pieces, usage = time.time(), [], [], None
    try:
        with urllib.request.urlopen(req, timeout=budget) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:])
                usage = event.get("usage") or usage
                piece = "".join(spoken(c.get("delta") or {}) for c in event.get("choices") or [])
                if piece:
                    times.append(time.time() - t0)
                    pieces.append(piece)
    except Exception as exc:                                   # noqa: BLE001 — the row names it
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "chunks": len(times)}
    gaps, text = [b - a for a, b in zip(times, times[1:])], "".join(pieces)
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"first_s": round(times[0], 3) if times else None, "chunks": len(times), "wall_s": round(time.time() - t0, 2),
            "median_gap_s": round(statistics.median(gaps), 4) if gaps else 0.0,
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16] if text else None, "text_head": text[:80],
            "pieces": pieces, "cached_tokens": cached, "usage": usage}


def cards_memory(gpu: str) -> list:
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit()]


def timed(a, row, peak) -> None:
    """The three timed requests, then the conversation, on the daemon with its replay."""
    for _ in range(3):
        row["requests"].append(ask(a.port, a.model, a.max_tokens, 600.0))
        peak.append(cards_memory(a.gpu))
        if row["requests"][-1].get("error"):
            return
    row["conversation_on"] = converse(a.port, a.model, 64, 600.0)


def serve(argv_serve, env, src, log, port, ready_timeout, sock, work) -> tuple:
    """(load seconds, None or why the daemon did not serve): one daemon up through its entry point, `work()` run against
    it, the daemon stopped by its process group whatever happens and its socket's leftovers removed."""
    t0 = time.time()
    with open(log, "w") as lf:
        daemon = subprocess.Popen(argv_serve, env=env, cwd=str(src.parent), stdout=lf, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        ready = False
        while time.time() - t0 < ready_timeout and daemon.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                    ready = b'"id"' in r.read()
            except Exception:                                  # noqa: BLE001 — not up yet
                ready = False
            if ready:
                break
            time.sleep(4)
        load = round(time.time() - t0, 1)
        if not ready:
            return load, (f"the daemon did not come up in {ready_timeout} s" if daemon.poll() is None
                          else f"the daemon exited rc={daemon.returncode}")
        work()
        return load, None
    finally:
        if daemon.poll() is None:
            os.killpg(daemon.pid, signal.SIGTERM)
            try:
                daemon.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(daemon.pid, signal.SIGKILL)
        for leftover in sock.parent.glob(sock.stem + ".*"):
            leftover.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("--model", "--gpu", "--hardware", "--src", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--known", default=None, help="JSON object: cell -> its proof file; a difference there is RED-KNOWN")
    a = p.parse_args(argv)
    known = read_known(a.known)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = f"{a.model.split('/')[-1]}_{a.mode}"
    log = out / f"{cell}.serve.log"
    src = Path(a.src).resolve()
    sha = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {"cell": cell, "model": a.model, "mode": a.mode, "gpu": a.gpu, "hardware": a.hardware, "engine": sha,
           "at": time.strftime("%Y-%m-%d %H:%M:%S"), "requests": [], "reset": RESET}
    # the instance's name lives in the workshop's tmp root: an AF_UNIX address holds ~100 bytes, a campaign path more
    sock = atelier.tmp(f"serve_cell_{os.getpid()}.sock")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "PYTHONPATH": str(src), "PYTHONNOUSERSITE": "1",
           "NEUROBRIX_PYTHON": a.python, "NBX_SOCKET_PATH": str(sock)}
    argv_serve = [a.python, "-m", "neurobrix", "serve", "--model", a.model, "--hardware", a.hardware, *MODES[a.mode],
                  "--http", "--http-host", "127.0.0.1", "--http-port", str(a.port), "--foreground"]
    row["command"] = " ".join(argv_serve[1:])
    peak = []
    try:
        row["load_s"], err = serve(argv_serve, env, src, log, a.port, a.ready_timeout, sock, lambda: timed(a, row, peak))
        if err:
            row["verdict"], row["reason"] = "RED", err
        else:
            row["verdict"], row["reason"] = row["speed_verdict"], row["speed_reason"] = judge(row["requests"])
            shas = {r.get("text_sha") for r in row["requests"]}
            row["texts_equal"] = len(shas) == 1 and None not in shas
            row["peak_mib"] = [max(c) for c in zip(*peak)] if peak and all(peak) else None
            off_argv = [a.python, str(Path(__file__).resolve().parent / "serve_without_replay.py"), *argv_serve[3:]]
            _s, err = serve(off_argv, env, src, out / f"{cell}.oracle.serve.log", a.port, a.ready_timeout, sock,
                            lambda: row.__setitem__("conversation_off", converse(a.port, a.model, 64, 600.0)))
            conv = (("RED", f"the replay-off oracle's daemon: {err}") if err
                    else judge_conversation(row.get("conversation_on") or [], row.get("conversation_off") or []))
            row["conversation_verdict"], row["conversation_reason"] = conv
            if row["verdict"] == "GREEN" and conv[0] != "GREEN":
                row["verdict"], row["reason"] = conv
            _s, err = serve(argv_serve, env, src, out / f"{cell}.reset.serve.log", a.port, a.ready_timeout, sock,
                            lambda: row.__setitem__("after_reset", ask(a.port, a.model, a.max_tokens, 600.0)))
            row.setdefault("after_reset", {"error": f"the restarted daemon: {err}"} if err else {})
            cache = judge_cache(cell, row["requests"], row["after_reset"], known)
            row["cache_verdict"], row["cache_reason"] = cache.pop("verdict"), cache.pop("reason")
            row["cache"] = cache
            if row["verdict"] == "GREEN" and row["cache_verdict"] != "GREEN":
                row["verdict"], row["reason"] = row["cache_verdict"], row["cache_reason"]
    finally:
        with open(out / "serve_rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    (out / f"{cell}.table.txt").write_text(table(row) + "\n")
    print(table(row))
    return 0 if passes(row["verdict"]) else 1


if __name__ == "__main__":
    sys.exit(main())
