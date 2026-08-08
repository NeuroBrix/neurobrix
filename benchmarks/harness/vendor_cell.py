"""Vendor competitor cell — the model's official HF code, pinned.

Supervisor rule: when no third-party runtime serves a family, the
competitor column is the vendor's own code on the same machine. This
cell runs inside a pinned venv (invoked by run_bench.cell_vendor) and
prints ONE JSON object on its last stdout line:
  {"cold_start_s": float, "requests": [...], "pins": {...}}

Families implemented:
  stt   — transformers WhisperForConditionalGeneration pipeline path
          (fp16, greedy), metric wall + RTF vs the pinned input.
  omni  — vendor chat API with speech-out (MiniCPM-o class:
          trust_remote_code, sdpa, fp16 — the card-official non-FA2
          path; runs in the minicpmo_vendor venv, pins in
          backends.yml). Metric: wall prompt→wav.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def run_stt(row: dict, n: int, repo: Path, media_dir: Path) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    snap = Path.home() / "hf_snapshots" / row["checkpoint"].split("/")[-1]
    src = str(snap if snap.exists() else row["checkpoint"])
    audio_path = repo / row["audio"]

    t0 = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        src, torch_dtype=torch.float16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(src)
    # Measurement boundary: the WHOLE request (audio file → text),
    # identical to the NeuroBrix cell (its daemon request also pays
    # file read + mel + decode). Model load stays in cold_start.
    import soundfile as sf
    cold_start = time.perf_counter() - t0

    def one() -> dict:
        t1 = time.perf_counter()
        audio, sr = sf.read(str(audio_path), dtype="float32")
        feats = processor(audio, sampling_rate=sr,
                          return_tensors="pt").input_features
        feats = feats.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            ids = model.generate(feats, do_sample=False)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        wall = time.perf_counter() - t1
        return {"wall_s": wall, "rtf": wall / row["audio_duration_s"],
                "text": text.strip()[:200]}

    one()  # warmup (cudnn autotune, graph init)
    return {
        "cold_start_s": cold_start,
        "requests": [one() for _ in range(n)],
        "pins": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float16",
            "decoding": "greedy (do_sample=False)",
            "source": src,
        },
    }


def run_omni_voice(row: dict, n: int, repo: Path,
                   media_dir: Path) -> dict:
    """MiniCPM-o-class speech-out through the vendor's own chat API.

    Card-official V100 path (R16 verdict 2026-08-08, sources in
    backends.yml minicpmo_venv): attn_implementation="sdpa" (the
    card's stated alternative to FA2), torch.float16 (vendor lineage
    guidance for non-bf16 GPUs), init_tts(), generate_audio=True.
    Greedy (sampling=False) to match the pinned-decoding rule.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    snap = Path.home() / "hf_snapshots" / row["checkpoint"].split("/")[-1]
    src = str(snap if snap.exists() else row["checkpoint"])

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        src, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.float16).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model.init_tts()
    cold_start = time.perf_counter() - t0

    def one(idx: int) -> dict:
        out_wav = media_dir / f"vendor_r{idx}.wav"
        t1 = time.perf_counter()
        model.chat(
            msgs=[{"role": "user", "content": [row["prompt"]]}],
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=int(row.get("max_new_tokens") or 32),
            use_tts_template=True,
            generate_audio=True,
            output_audio_path=str(out_wav),
        )
        wall = time.perf_counter() - t1
        rec = {"wall_s": wall}
        if out_wav.exists():
            rec["sha256"] = hashlib.sha256(
                out_wav.read_bytes()).hexdigest()
        return rec

    one(-1)  # warmup
    return {
        "cold_start_s": cold_start,
        "requests": [one(i) for i in range(n)],
        "pins": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float16",
            "attn_implementation": "sdpa",
            "decoding": "greedy (sampling=False)",
            "source": src,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--media-dir", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    row = json.loads(args.row)
    repo = Path(args.repo)
    media_dir = Path(args.media_dir)

    if row["metric_class"] == "stt":
        out = run_stt(row, args.n, repo, media_dir)
    elif row["metric_class"] == "omni":
        out = run_omni_voice(row, args.n, repo, media_dir)
    else:
        raise SystemExit(
            f"vendor_cell: metric_class '{row['metric_class']}' not "
            f"implemented yet — its venv+branch lands with its row")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
