#!/usr/bin/env python3
"""gate_serve_cell.py — the gate's SERVE cell: one daemon, three successive requests, the third at the first's speed.

A run cell (`regression_matrix.py run`) launches one process per request: what a daemon keeps between two requests
never meets it. Two defects of 2026-11-11 lived exactly there — a served mixture-of-experts model answered its first
request at full speed and every later one two hundred times slower, then twice slower — and no cell could see them.

CONTRACT
  * the daemon is started through the engine's entry point (`python -m neurobrix serve`, the family's engine flag,
    the hardware profile given), on an instance name of its own (`NBX_SOCKET_PATH`, in the workshop's tmp root) and a loopback port:
    it can neither answer nor be answered by a daemon serving someone else;
  * three identical streamed chat requests (temperature 0), one after the other, to that one daemon;
  * per request: the time to the first chunk, the chunks, the median time between two chunks, the sha256 of the text;
  * RED when the daemon does not come up, a request fails or returns no text, or the THIRD request's median time
    between chunks is more than `SLOWER_IS_RED` times the FIRST's (the gate's speed rule, the same factor as a cell
    against its base); GREEN otherwise. Whether the three texts are byte-equal is RECORDED, not judged: a request
    that reuses the previous one's cache may round differently, and the text's judgment is the reader's;
  * one JSON row appended to <out>/serve_rows.jsonl whatever happens (verdict, reason, the three requests, the load
    time, the cards' peak memory read from outside), the daemon stopped by its pid, its log kept beside the row.

    gate_serve_cell.py --model <hf name> --gpu <cards> --hardware <profile> --mode triton|native --src <tree>/src
                       --out <dir> [--port 8090] [--max-tokens 60] [--ready-timeout 900] [--python <engine python>]
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
PROMPT = "Explain what a hash table is, how collisions are handled, and give a short Python example."


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


def ask(port: int, model: str, max_tokens: int, budget: float) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, times, text = time.time(), [], ""
    try:
        with urllib.request.urlopen(req, timeout=budget) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                piece = "".join((c.get("delta") or {}).get("content") or ""
                                for c in json.loads(line[5:]).get("choices") or [])
                if piece:
                    times.append(time.time() - t0)
                    text += piece
    except Exception as exc:                                   # noqa: BLE001 — the row names it
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "chunks": len(times)}
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"first_s": round(times[0], 3) if times else None, "chunks": len(times), "wall_s": round(time.time() - t0, 2),
            "median_gap_s": round(statistics.median(gaps), 4) if gaps else 0.0,
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16] if text else None, "text_head": text[:80]}


def cards_memory(gpu: str) -> list:
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("--model", "--gpu", "--hardware", "--src", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = f"{a.model.split('/')[-1]}_{a.mode}"
    log = out / f"{cell}.serve.log"
    src = Path(a.src).resolve()
    sha = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {"cell": cell, "model": a.model, "mode": a.mode, "gpu": a.gpu, "hardware": a.hardware, "engine": sha,
           "at": time.strftime("%Y-%m-%d %H:%M:%S"), "requests": []}
    # the instance's name lives in the workshop's tmp root: an AF_UNIX address holds ~100 bytes, a campaign path more
    sock = atelier.tmp(f"serve_cell_{os.getpid()}.sock")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "PYTHONPATH": str(src), "PYTHONNOUSERSITE": "1",
           "NEUROBRIX_PYTHON": a.python, "NBX_SOCKET_PATH": str(sock)}
    argv_serve = [a.python, "-m", "neurobrix", "serve", "--model", a.model, "--hardware", a.hardware, *MODES[a.mode],
                  "--http", "--http-host", "127.0.0.1", "--http-port", str(a.port), "--foreground"]
    row["command"] = " ".join(argv_serve[1:])
    t0, peak = time.time(), []
    with open(log, "w") as lf:
        daemon = subprocess.Popen(argv_serve, env=env, cwd=str(src.parent), stdout=lf, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        ready = False
        while time.time() - t0 < a.ready_timeout and daemon.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=3) as r:
                    ready = b'"id"' in r.read()
            except Exception:                                  # noqa: BLE001 — not up yet
                ready = False
            if ready:
                break
            time.sleep(4)
        row["load_s"] = round(time.time() - t0, 1)
        if not ready:
            row["verdict"], row["reason"] = "RED", (f"the daemon did not come up in {a.ready_timeout} s"
                                                    if daemon.poll() is None else f"the daemon exited rc={daemon.returncode}")
        else:
            for _ in range(3):
                row["requests"].append(ask(a.port, a.model, a.max_tokens, 600.0))
                peak.append(cards_memory(a.gpu))
                if row["requests"][-1].get("error"):
                    break
            row["verdict"], row["reason"] = judge(row["requests"])
            shas = {r.get("text_sha") for r in row["requests"]}
            row["texts_equal"] = len(shas) == 1 and None not in shas
            row["peak_mib"] = [max(c) for c in zip(*peak)] if peak and all(peak) else None
    finally:
        if daemon.poll() is None:
            os.killpg(daemon.pid, signal.SIGTERM)
            try:
                daemon.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(daemon.pid, signal.SIGKILL)
        for leftover in sock.parent.glob(sock.stem + ".*"):
            leftover.unlink(missing_ok=True)
        with open(out / "serve_rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"{row['verdict']} {cell} ({a.gpu}, {a.hardware}, {sha}): {row['reason']}")
    return 0 if row["verdict"] == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
