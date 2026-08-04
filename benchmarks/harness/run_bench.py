"""Three-column benchmark runner (see ../README.md for the rules).

One cell = one backend serving one row's checkpoint on ONE pinned GPU,
warm (1 warmup request), then N timed requests. Cells run strictly
sequentially. Every cell writes a raw JSON artifact; the run writes an
environment manifest and a summary table. Fairness rules 1-6 of the
methodology are enforced structurally here (GPU guard, fp16, warm
serving, fixed prompts, N repetitions, pinned environment).

Usage:
  python benchmarks/harness/run_bench.py --row llm_dense_tinyllama \
      --columns vllm,ollama,neurobrix_pytorch,neurobrix_triton \
      --gpu 3 --date 2026_08_04
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

VLLM_BIN = Path.home() / "bench_venvs" / "vllm073" / "bin"
OLLAMA_MODELS = Path.home() / "bench_assets" / "ollama_models"
GGUF = Path.home() / "bench_assets" / "gguf"
SNAPSHOTS = Path.home() / "hf_snapshots"
VLLM_PORT = 8077
OLLAMA_PORT = 11435


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def gpu_guard(gpu: int) -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(f"REFUSED: compute apps present on GPUs:\n{out}")


def gpu_mem_sampler(gpu: int, stop: threading.Event, peak: list) -> None:
    while not stop.is_set():
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        try:
            peak[0] = max(peak[0], int(out))
        except ValueError:
            pass
        stop.wait(1.0)


def http_json(url: str, payload: dict | None = None, timeout: float = 300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_stream_lines(url: str, payload: dict, timeout: float = 600):
    """POST and yield (wall_time, line) per streamed line."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            yield time.perf_counter(), raw


def wait_for(probe, timeout_s: float, label: str,
             proc: subprocess.Popen | None = None) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label}: server process died (rc={proc.returncode}) — "
                f"see its cell log")
        try:
            if probe():
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{label} not ready after {timeout_s}s")


# ---------------------------------------------------------------- cells

def cell_vllm(row: dict, gpu: int, n: int) -> dict:
    snap = SNAPSHOTS / row["neurobrix_model"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.Popen(
        [str(VLLM_BIN / "vllm"), "serve", str(snap),
         "--dtype", "float16", "--port", str(VLLM_PORT)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    t_launch = time.perf_counter()
    try:
        wait_for(lambda: http_json(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models"), 600, "vllm")
        cold_start = time.perf_counter() - t_launch

        def one() -> dict:
            t0 = time.perf_counter()
            ttft = None
            n_tok = 0
            for wall, raw in http_stream_lines(
                f"http://127.0.0.1:{VLLM_PORT}/v1/completions",
                {"model": str(snap), "prompt": row["prompt"],
                 "max_tokens": row["max_new_tokens"], "temperature": 0,
                 "ignore_eos": True, "stream": True},
            ):
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                if ttft is None:
                    ttft = wall - t0
                n_tok += 1
                t_last = wall
            return {"wall_s": t_last - t0, "ttft_s": ttft,
                    "tokens": n_tok,
                    "tok_per_s": (n_tok - 1) / (t_last - t0 - ttft)
                    if n_tok > 1 else None}
        one()  # warmup
        return {"cold_start_s": cold_start,
                "requests": [one() for _ in range(n)]}
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)


def cell_ollama(row: dict, gpu: int, n: int) -> dict:
    tag = f"{row['id']}-f16"
    gguf = next(GGUF.glob("*.f16.gguf"))
    OLLAMA_MODELS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
           "OLLAMA_MODELS": str(OLLAMA_MODELS)}
    proc = subprocess.Popen(["ollama", "serve"], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    t_launch = time.perf_counter()
    try:
        wait_for(lambda: http_json(
            f"http://127.0.0.1:{OLLAMA_PORT}/api/version"), 120, "ollama")
        mf = GGUF / f"Modelfile.{tag}"
        mf.write_text(f"FROM {gguf}\n")
        subprocess.run(["ollama", "create", tag, "-f", str(mf)],
                       env=env, check=True, capture_output=True)

        def one() -> dict:
            t0 = time.perf_counter()
            ttft = None
            final = None
            for wall, raw in http_stream_lines(
                f"http://127.0.0.1:{OLLAMA_PORT}/api/generate",
                {"model": tag, "prompt": row["prompt"], "stream": True,
                 "options": {"temperature": 0,
                             "num_predict": row["max_new_tokens"]}},
            ):
                chunk = json.loads(raw)
                if ttft is None and chunk.get("response"):
                    ttft = wall - t0
                if chunk.get("done"):
                    final = chunk
                    t_last = wall
            n_tok = final.get("eval_count", 0)
            eval_ns = final.get("eval_duration", 0)
            return {"wall_s": t_last - t0, "ttft_s": ttft,
                    "tokens": n_tok,
                    "tok_per_s": n_tok / (eval_ns / 1e9)
                    if eval_ns else None}
        cold = one()  # warmup (includes model load)
        return {"cold_start_s": (time.perf_counter() - t_launch),
                "warmup": cold,
                "requests": [one() for _ in range(n)]}
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)


def cell_neurobrix(row: dict, gpu: int, n: int, triton: bool,
                   hardware: str, log_dir: Path) -> dict:
    from neurobrix.serving.client import DaemonClient
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    # Pinned run = matching single-GPU hardware profile (the battery
    # closure-config pattern): the placement plan must be solved for
    # the ONE visible device, never for the physical rig.
    cmd = [sys.executable, "-m", "neurobrix", "serve",
           "--model", row["neurobrix_model"], "--hardware", hardware,
           "--foreground"]
    if triton:
        cmd.append("--triton")
    log = (log_dir / f"server_neurobrix_{'triton' if triton else 'pytorch'}"
           ".log").open("w")
    proc = subprocess.Popen(cmd, env=env, cwd=str(REPO),
                            stdout=log, stderr=subprocess.STDOUT)
    t_launch = time.perf_counter()
    try:
        wait_for(DaemonClient.is_running, 900, "neurobrix daemon",
                 proc=proc)

        client = DaemonClient()

        def warm_probe() -> bool:
            try:
                client.connect()
            except Exception:
                return False
            st = client.status()
            st = st.get("result", st) or {}
            return st.get("model") == row["neurobrix_model"]

        wait_for(warm_probe, 900, "neurobrix warm", proc=proc)
        cold_start = time.perf_counter() - t_launch

        def one() -> dict:
            t0 = time.perf_counter()
            res = client.generate(prompt=row["prompt"],
                                  max_tokens=row["max_new_tokens"],
                                  temperature=0)
            wall = time.perf_counter() - t0
            r = res.get("result", res) or {}
            n_tok = r.get("tokens_generated") or r.get("num_tokens")
            return {"wall_s": wall, "ttft_s": None,  # daemon RPC: no wire streaming yet
                    "tokens": n_tok,
                    "tok_per_s": (n_tok / wall) if n_tok else None}
        one()  # warmup
        out = {"cold_start_s": cold_start,
               "requests": [one() for _ in range(n)]}
        client.close()
        return out
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()


CELLS = {
    "vllm": lambda row, gpu, n, hw, ld: cell_vllm(row, gpu, n),
    "ollama": lambda row, gpu, n, hw, ld: cell_ollama(row, gpu, n),
    "neurobrix_pytorch": lambda row, gpu, n, hw, ld: cell_neurobrix(
        row, gpu, n, triton=False, hardware=hw, log_dir=ld),
    "neurobrix_triton": lambda row, gpu, n, hw, ld: cell_neurobrix(
        row, gpu, n, triton=True, hardware=hw, log_dir=ld),
}


def env_manifest(gpu: int) -> dict:
    q = subprocess.run(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    ollama_v = subprocess.run(["ollama", "--version"],
                              capture_output=True, text=True).stdout.strip()
    return {
        "gpu": q.stdout.strip(),
        "gpu_index": gpu,
        "engine_commit": git,
        "vllm": "0.7.3 (venv bench_venvs/vllm073, torch 2.5.1+cu121, "
                "xformers 0.0.28.post3, V0 engine, XFormers backend)",
        "ollama": ollama_v,
        "ollama_flash_attention": os.environ.get(
            "OLLAMA_FLASH_ATTENTION", "unset (default)"),
        "python": sys.version.split()[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--hardware", default="v100-32g",
                    help="single-GPU profile matching the pinned GPU")
    args = ap.parse_args()

    rows = {r["id"]: r for r in
            load_yaml(REPO / "benchmarks" / "config" / "rows.yml")["rows"]}
    row = rows[args.row]
    out_dir = REPO / "benchmarks" / "results" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_manifest.json").write_text(
        json.dumps(env_manifest(args.gpu), indent=1) + "\n")

    for col in args.columns.split(","):
        gpu_guard(args.gpu)
        stop = threading.Event()
        peak = [0]
        th = threading.Thread(
            target=gpu_mem_sampler, args=(args.gpu, stop, peak))
        th.start()
        t0 = time.time()
        try:
            result = CELLS[col](row, args.gpu, args.repetitions,
                                args.hardware, out_dir)
            status = "ok"
        except Exception as e:
            result = {"error": repr(e)}
            status = "error"
        stop.set()
        th.join()
        artifact = {
            "row": args.row, "column": col, "status": status,
            "peak_gpu_mem_mib": peak[0],
            "started_unix": t0, **result,
        }
        path = out_dir / f"{args.row}_{col}.json"
        path.write_text(json.dumps(artifact, indent=1) + "\n")
        print(f"[bench] {args.row}/{col}: {status} "
              f"peak={peak[0]}MiB -> {path.name}", flush=True)
        time.sleep(5)  # let the GPU drain between cells
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
