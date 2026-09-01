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


class DNR(RuntimeError):
    """Does-Not-Run capability verdict: this backend cannot serve this
    row on the V100 rig (arch missing at the era pin, sm_80+-only
    dependency, bf16-only stack, no runtime serves the family). A DNR
    cell is a first-class recorded result — the capability axis of the
    public matrix — never a silent skip."""


def _gpu_env(gpu: int | None) -> dict:
    """CUDA visibility for a cell. gpu=None = the MACHINE config: the
    whole rig stays visible and the backend places freely (its best
    weapon). An int = the pinned single-GPU closure config."""
    return {} if gpu is None else {"CUDA_VISIBLE_DEVICES": str(gpu)}


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def gpu_guard(gpu: int, wait_s: float = 0.0) -> None:
    """Refuse to run while ANY compute app holds a GPU (exclusive-
    machine fairness). wait_s>0 grants a grace window first: the
    previous cell's daemon can take tens of seconds to free 60+GB
    (memory-manager device syncs) — between OUR OWN cells we wait for
    the drain instead of aborting the row (three refusal-races on
    2026-08-08 motivated this)."""
    t0 = time.perf_counter()
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        if not out:
            return
        if time.perf_counter() - t0 >= wait_s:
            raise SystemExit(
                f"REFUSED: compute apps present on GPUs:\n{out}")
        time.sleep(5.0)


def gpu_mem_sampler(gpu: int | None, stop: threading.Event,
                    peak: list) -> None:
    """Track peak GPU memory. Pinned config samples the ONE GPU; the
    machine config (gpu=None) samples every GPU and tracks the peak of
    the rig-wide SUM (what the backend actually consumed across its
    free placement)."""
    id_args = [] if gpu is None else [f"--id={gpu}"]
    while not stop.is_set():
        out = subprocess.run(
            ["nvidia-smi", *id_args,
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        try:
            peak[0] = max(peak[0], sum(int(v) for v in out.splitlines()))
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

def cell_vllm(row: dict, gpu: int | None, n: int) -> dict:
    snap = SNAPSHOTS / row["neurobrix_model"]
    env = {**os.environ, **_gpu_env(gpu)}
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


def cell_ollama(row: dict, gpu: int | None, n: int) -> dict:
    # Two supply routes, per row:
    #   gguf:        local f16 GGUF -> `ollama create` (LLM rows)
    #   ollama_pull: official library tag -> `ollama pull` (rows whose
    #                GGUF-import path is documented-broken at the pin,
    #                e.g. qwen3vlmoe — their packaging is their best
    #                weapon; the tag's precision is recorded).
    pull_tag = row.get("ollama_pull")
    tag = pull_tag or f"{row['id']}-f16"
    gguf = None
    if not pull_tag:
        # The GGUF is named per-row in rows.yml — a bare glob would
        # silently serve the wrong checkpoint once a second row's
        # GGUF lands.
        gguf = GGUF / row["gguf"]
        if not gguf.exists():
            raise FileNotFoundError(f"row GGUF missing: {gguf}")
    OLLAMA_MODELS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_gpu_env(gpu),
           "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
           "OLLAMA_MODELS": str(OLLAMA_MODELS)}
    proc = subprocess.Popen(["ollama", "serve"], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    t_launch = time.perf_counter()
    try:
        wait_for(lambda: http_json(
            f"http://127.0.0.1:{OLLAMA_PORT}/api/version"), 120, "ollama")
        if pull_tag:
            try:
                subprocess.run(["ollama", "pull", pull_tag], env=env,
                               check=True, capture_output=True,
                               text=True, timeout=7200)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or e.stdout or "").strip()[-400:]
                # Capability failures only — a network hiccup is an
                # error to retry, never a DNR verdict.
                if ("unsupported" in err.lower()
                        or "requires ollama" in err.lower()
                        or "not found" in err.lower()):
                    raise DNR(f"ollama pull {pull_tag} at the era "
                              f"pin: {err}") from e
                raise
        else:
            mf = GGUF / f"Modelfile.{tag}"
            mf.write_text(f"FROM {gguf}\n")
            try:
                subprocess.run(["ollama", "create", tag, "-f", str(mf)],
                               env=env, check=True, capture_output=True,
                               text=True)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or e.stdout or "").strip()[-400:]
                if ("unsupported" in err.lower()
                        or "unknown architecture" in err.lower()
                        or "unknown model architecture" in err.lower()):
                    raise DNR(f"ollama create at the era pin: {err}")
                raise

        payload_extra: dict = {}
        if row.get("input_image"):
            # VLM rows: same committed image every column reads.
            import base64
            payload_extra["images"] = [base64.b64encode(
                (REPO / row["input_image"]).read_bytes()).decode()]

        def one() -> dict:
            t0 = time.perf_counter()
            ttft = None
            final = None
            for wall, raw in http_stream_lines(
                f"http://127.0.0.1:{OLLAMA_PORT}/api/generate",
                {"model": tag, "prompt": row["prompt"], "stream": True,
                 **payload_extra,
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


def _neurobrix_daemon(row: dict, gpu: int | None, triton: bool,
                      hardware: str, log_dir: Path):
    """Start the warm daemon for a row; return (proc, client,
    cold_start_s). Row-level env pins (e.g. NBX_FORCE_RAND_SEED for
    voice rows) ride the daemon environment — closure-config doctrine.

    gpu=int → PINNED config: one visible GPU + the matching single-GPU
    hardware profile (battery closure-config pattern — the plan is
    solved for the ONE visible device, never the physical rig).
    gpu=None → MACHINE config: the whole rig visible, NO --hardware
    (autodetect fingerprints the real machine) — Prism places freely.
    """
    from neurobrix.serving.client import DaemonClient
    env = {**os.environ, **_gpu_env(gpu),
           **{k: str(v) for k, v in (row.get("env") or {}).items()}}
    cmd = [sys.executable, "-m", "neurobrix", "serve",
           "--model", row["neurobrix_model"], "--foreground"]
    if gpu is not None:
        cmd += ["--hardware", hardware]
    if triton:
        cmd.append("--triton")
    cfg_tag = "_machine" if gpu is None else ""
    log = (log_dir / f"server_neurobrix_{'triton' if triton else 'pytorch'}"
           f"_{row['id']}{cfg_tag}.log").open("w")
    proc = subprocess.Popen(cmd, env=env, cwd=str(REPO),
                            stdout=log, stderr=subprocess.STDOUT)
    t_launch = time.perf_counter()
    wait_for(DaemonClient.is_running, 900, "neurobrix daemon", proc=proc)
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
    return proc, client, time.perf_counter() - t_launch


def _stop_daemon(proc) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()


def cell_neurobrix_media(row: dict, gpu: int, n: int, triton: bool,
                         hardware: str, log_dir: Path) -> dict:
    """image / video / stt / omni rows: warm daemon, N timed requests.

    Metrics per class (never tok/s): image → s/image + s/step; video →
    s/video + s/step + s/frame; stt → wall + RTF vs the pinned input
    duration; omni → wall prompt→wav. Output artifacts land under the
    R29 tree (media dir), their sha256 in the committed JSON.
    """
    import hashlib
    mclass = row["metric_class"]
    media_dir = (REPO / "validation_outputs" /
                 f"bench_reference_{log_dir.name}" / row["id"])
    media_dir.mkdir(parents=True, exist_ok=True)
    ext = {"image": "png", "video": "mp4", "omni": "wav"}.get(mclass, "txt")

    kwargs: dict = {"prompt": row.get("prompt") or ""}
    if row.get("steps") is not None:
        kwargs["steps"] = row["steps"]
    if row.get("seed") is not None:
        kwargs["seed"] = row["seed"]
    if row.get("mode"):
        kwargs["mode"] = row["mode"]
    if row.get("max_new_tokens") is not None:
        kwargs["max_tokens"] = row["max_new_tokens"]
    if row.get("audio"):
        kwargs["audio_path"] = str(REPO / row["audio"])
    if row.get("input_image"):
        # Upscaler rows (and any image-input row): the pinned input
        # image rides the serving warm path's image_path kwarg.
        kwargs["image_path"] = str(REPO / row["input_image"])
    if row.get("height") is not None:
        kwargs["height"] = row["height"]
    if row.get("width") is not None:
        kwargs["width"] = row["width"]
    if row.get("num_frames") is not None:
        kwargs["num_frames"] = row["num_frames"]

    proc, client, cold_start = _neurobrix_daemon(
        row, gpu, triton, hardware, log_dir)
    try:
        eng = ("triton" if triton else "pytorch") + (
            "_machine" if gpu is None else "")

        def one(idx: int) -> dict:
            out_path = media_dir / f"neurobrix_{eng}_r{idx}.{ext}"
            t0 = time.perf_counter()
            res = client.generate(output_path=str(out_path), **kwargs)
            wall = time.perf_counter() - t0
            r = res.get("result", res) or res or {}
            rec = {"wall_s": wall}
            if out_path.exists():
                rec["sha256"] = hashlib.sha256(
                    out_path.read_bytes()).hexdigest()
            if mclass in ("image", "video") and row.get("steps"):
                rec["s_per_step"] = wall / row["steps"]
            if mclass == "video" and row.get("num_frames"):
                rec["s_per_frame"] = wall / row["num_frames"]
            if mclass == "stt":
                rec["text"] = (r.get("text") or "")[:200]
                # Pinned input duration is row data, never a literal
                # (gardien 2026-08-08: the constant was duplicated).
                rec["rtf"] = wall / row["audio_duration_s"]
            return rec

        one(-1)  # warmup (weights load into VRAM on first request)
        out = {"cold_start_s": cold_start,
               "requests": [one(i) for i in range(n)]}
        client.close()
        return out
    finally:
        _stop_daemon(proc)


def cell_neurobrix(row: dict, gpu: int, n: int, triton: bool,
                   hardware: str, log_dir: Path) -> dict:
    if row.get("metric_class", "tokens") != "tokens":
        return cell_neurobrix_media(row, gpu, n, triton, hardware, log_dir)
    from neurobrix.serving.client import DaemonClient  # noqa: F401
    import hashlib
    proc, client, cold_start = _neurobrix_daemon(
        row, gpu, triton, hardware, log_dir)
    try:

        stream_kwargs: dict = {"prompt": row["prompt"],
                               "max_tokens": row["max_new_tokens"],
                               "temperature": 0}
        if row.get("input_image"):
            # VLM rows: the committed image asset, same input every
            # column reads (routed through the shared CLI/daemon brick).
            stream_kwargs["image_path"] = str(REPO / row["input_image"])

        # R29 answer export (evidence-chain fix 2026-08-10): the cell
        # JSONs carried metrics only, leaving the LLM/VLM answers with
        # no on-disk artifact to inspect. Every request's answer text
        # is hashed into the JSON; request r0's full text (2000-char
        # R29 bound) lands in the media tree.
        media_dir = (REPO / "validation_outputs" /
                     f"bench_reference_{log_dir.name}" / row["id"])
        media_dir.mkdir(parents=True, exist_ok=True)
        eng = ("triton" if triton else "pytorch") + (
            "_machine" if gpu is None else "")
        answers: list = []

        def one() -> dict:
            # Streamed RPC: per-token events from the daemon's decode loop.
            # TTFT = client wall at the first token event; decode rate =
            # (n-1)/(t_last-t0-ttft) — the exact formula of the vLLM cell,
            # so the columns stay methodologically symmetric.
            t0 = time.perf_counter()
            ttft = None
            t_last = None
            n_events = 0
            final = {}
            for kind, payload in client.generate_stream(**stream_kwargs):
                wall = time.perf_counter()
                if kind == "token":
                    if ttft is None:
                        ttft = wall - t0
                    t_last = wall
                    n_events += 1
                else:
                    final = payload or {}
            wall_s = time.perf_counter() - t0
            # Exact daemon count when the final payload carries it (llm
            # flow); otherwise the per-token stream events ARE the exact
            # count — one event per sampled token at the generator site
            # (vlm flow's final payload has no token count).
            n_tok = final.get("tokens") or n_events or None
            text = final.get("text") or ""
            answers.append(text)
            rec = {"wall_s": wall_s, "ttft_s": ttft,
                   "tokens": n_tok, "tokens_streamed": n_events,
                   "tok_per_s": (n_tok - 1) / (t_last - t0 - ttft)
                   if (n_tok and n_tok > 1 and ttft is not None)
                   else None}
            if text:
                rec["answer_sha256"] = hashlib.sha256(
                    text.encode("utf-8")).hexdigest()
            return rec
        one()  # warmup
        answers.clear()  # warmup answer is not evidence
        out = {"cold_start_s": cold_start,
               "requests": [one() for _ in range(n)]}
        if answers and answers[0]:
            answer_path = (media_dir /
                           f"neurobrix_{eng}_answer_r0.txt")
            answer_path.write_text(answers[0][:2000] + "\n",
                                   encoding="utf-8")
            out["answers_identical"] = all(
                a == answers[0] for a in answers)
        client.close()
        return out
    finally:
        _stop_daemon(proc)


DIFFUSERS_BIN = Path.home() / "bench_venvs" / "diffusers" / "bin"


def cell_subprocess_venv(row: dict, gpu: int | None, n: int, script: str,
                         python_bin: Path, log_dir: Path,
                         extra_env: dict | None = None) -> dict:
    """Run a competitor cell in its pinned venv as a subprocess.

    The cell script receives the row as JSON on argv and MUST print a
    single JSON object on its last stdout line:
    {cold_start_s, requests: [...]} — same shape as in-process cells.
    Media artifacts go under the R29 tree; shas inside the JSON.
    """
    if not python_bin.exists():
        raise FileNotFoundError(
            f"pinned venv python missing: {python_bin} — install per "
            f"benchmarks/config/backends.yml before running this column")
    media_dir = (REPO / "validation_outputs" /
                 f"bench_reference_{log_dir.name}" / row["id"])
    media_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_gpu_env(gpu), **(extra_env or {})}
    log = (log_dir / f"cell_{script.rsplit('/', 1)[-1].split('.')[0]}"
           f"_{row['id']}.log").open("w")
    res = subprocess.run(
        [str(python_bin), str(REPO / "benchmarks" / "harness" / script),
         "--row", json.dumps(row), "--n", str(n),
         "--media-dir", str(media_dir), "--repo", str(REPO)],
        env=env, capture_output=True, text=True, timeout=7200)
    log.write(res.stdout + "\n--- stderr ---\n" + res.stderr)
    log.close()
    if res.returncode != 0:
        raise RuntimeError(
            f"{script} rc={res.returncode}: {res.stderr[-500:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def cell_diffusers(row: dict, gpu: int | None, n: int,
                   log_dir: Path) -> dict:
    # Per-row venv override (`diffusers_venv`), same mechanism as the
    # upscaler stacks: a pipeline class absent from the era pin runs in
    # its own dated venv, recorded by the cell's pins block
    # (sana_video / diffusers036 is the first user — R16 2026-09-01).
    venv = row.get("diffusers_venv")
    python_bin = (Path.home() / "bench_venvs" / venv / "bin" / "python"
                  if venv else DIFFUSERS_BIN / "python")
    return cell_subprocess_venv(
        row, gpu, n, "diffusers_cell.py", python_bin, log_dir)


MINICPMO_BIN = Path.home() / "bench_venvs" / "minicpmo_vendor" / "bin"


def cell_vendor(row: dict, gpu: int | None, n: int,
                log_dir: Path) -> dict:
    # Vendor competitor: the model's official HF code, pinned. STT
    # (whisper) runs on the vllm073 stack (torch 2.5.1+cu121,
    # transformers 4.49.0); omni voice runs in the dedicated
    # minicpmo_vendor venv (transformers 4.51.0 + minicpmo-utils —
    # pins + sources in backends.yml vendor_transformers block).
    python_bin = (MINICPMO_BIN / "python"
                  if row["metric_class"] == "omni"
                  else VLLM_BIN / "python")
    return cell_subprocess_venv(
        row, gpu, n, "vendor_cell.py", python_bin, log_dir)


def cell_vendor_upscaler(row: dict, gpu: int | None, n: int,
                         log_dir: Path) -> dict:
    # Upscaler competitor: the model's vendor/reference stack in the
    # row's pinned venv (`upscaler_venv` — basicsr version conflicts
    # force per-stack venvs, S3 R16 synthesis). HF snapshots ride the
    # NAS per the S3 disk policy.
    python_bin = (Path.home() / "bench_venvs" / row["upscaler_venv"]
                  / "bin" / "python")
    nas = Path.home() / "hf_snapshots"
    extra_env = {"HF_HOME": str(nas / "hf_home"),
                 "HUGGINGFACE_HUB_CACHE": str(nas / "hf_home" / "hub")}
    return cell_subprocess_venv(row, gpu, n, "upscaler_cell.py",
                                python_bin, log_dir, extra_env)


CELLS = {
    "vllm": lambda row, gpu, n, hw, ld: cell_vllm(row, gpu, n),
    "ollama": lambda row, gpu, n, hw, ld: cell_ollama(row, gpu, n),
    "diffusers": lambda row, gpu, n, hw, ld: cell_diffusers(
        row, gpu, n, ld),
    "vendor_transformers": lambda row, gpu, n, hw, ld: cell_vendor(
        row, gpu, n, ld),
    "vendor_upscaler": lambda row, gpu, n, hw, ld: cell_vendor_upscaler(
        row, gpu, n, ld),
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
    ap.add_argument("--config", default="pinned",
                    choices=["pinned", "machine", "both"],
                    help="pinned = one visible GPU (closure-config); "
                         "machine = whole rig visible, Prism free and "
                         "competitors' best weapons; both = the two, "
                         "intersected with the row's serving_configs")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells whose result JSON is already ok")
    args = ap.parse_args()

    rows = {r["id"]: r for r in
            load_yaml(REPO / "benchmarks" / "config" / "rows.yml")["rows"]}
    row = rows[args.row]
    out_dir = REPO / "benchmarks" / "results" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_manifest.json").write_text(
        json.dumps(env_manifest(args.gpu), indent=1) + "\n")

    row_configs = row.get("serving_configs", ["pinned"])
    wanted = ["pinned", "machine"] if args.config == "both" \
        else [args.config]
    configs = [c for c in wanted if c in row_configs]
    if not configs:
        raise SystemExit(f"row {args.row} declares serving_configs="
                         f"{row_configs}; nothing matches --config "
                         f"{args.config}")

    # Declarative DNR columns: sourced era-pin impossibilities recorded
    # as first-class cells without launching anything (the runtime DNR
    # class — the cell tried and failed — is preferred when an attempt
    # is meaningful; declarative is for arch-absent-at-pin facts).
    for col, evidence in (row.get("dnr_columns") or {}).items():
        for cfg in configs:
            tag = "__machine" if cfg == "machine" else ""
            p = out_dir / f"{args.row}_{col}{tag}.json"
            if not p.exists():
                p.write_text(json.dumps({
                    "row": args.row, "column": col, "config": cfg,
                    "status": "dnr", "evidence": evidence,
                }, indent=1) + "\n")
                print(f"[bench] {args.row}/{col}[{cfg}]: dnr (declared) "
                      f"-> {p.name}", flush=True)

    for cfg in configs:
        cell_gpu = args.gpu if cfg == "pinned" else None
        tag = "__machine" if cfg == "machine" else ""
        for col in args.columns.split(","):
            path = out_dir / f"{args.row}_{col}{tag}.json"
            # Cell-level idempotence: a power-loss resume re-runs the
            # recorded command verbatim, so completed cells must not be
            # overwritten (dnr is a completed verdict too).
            if path.exists() and not args.force:
                try:
                    prior = json.loads(path.read_text())
                except ValueError:
                    prior = {}
                if prior.get("status") in ("ok", "dnr"):
                    print(f"[bench] {args.row}/{col}[{cfg}]: skip, "
                          f"existing {prior['status']} result "
                          f"({path.name}); use --force to re-run",
                          flush=True)
                    continue
            gpu_guard(args.gpu, wait_s=180.0)
            stop = threading.Event()
            peak = [0]
            th = threading.Thread(
                target=gpu_mem_sampler, args=(cell_gpu, stop, peak))
            th.start()
            t0 = time.time()
            try:
                result = CELLS[col](row, cell_gpu, args.repetitions,
                                    args.hardware, out_dir)
                status = "ok"
            except DNR as e:
                result = {"evidence": str(e)}
                status = "dnr"
            except Exception as e:
                result = {"error": repr(e)}
                status = "error"
            stop.set()
            th.join()
            artifact = {
                "row": args.row, "column": col, "config": cfg,
                "status": status,
                "peak_gpu_mem_mib": peak[0],
                "started_unix": t0, **result,
            }
            path.write_text(json.dumps(artifact, indent=1) + "\n")
            print(f"[bench] {args.row}/{col}[{cfg}]: {status} "
                  f"peak={peak[0]}MiB -> {path.name}", flush=True)
            time.sleep(5)  # let the GPU drain between cells
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
