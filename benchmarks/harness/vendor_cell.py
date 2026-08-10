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

    # Reference voice: on this checkpoint lineage the vendor speech leg
    # REQUIRES a reference wav — token2wav crashes on prompt_wav=None
    # even on the "default voice" warning path (2026-08-08 bench cell
    # log: TypeError in stepaudio2 token2wav._prepare_prompt on every
    # request; model.chat swallowed it and returned text-only, so the
    # recorded walls measured NO speech synthesis — the invalidated
    # vendor number). The canonical snapshot asset is the vendor's
    # best-weapon config, recorded in pins.
    ref_wav = Path(src) / "assets" / "HT_ref_audio.wav"
    if not ref_wav.is_file():
        raise SystemExit(
            f"vendor_cell: reference wav missing at {ref_wav} — the "
            f"vendor speech leg cannot synthesize without it")
    sys_msg = model.get_sys_prompt(
        ref_audio=str(ref_wav), mode="audio_assistant", language="en")

    def one(idx: int) -> dict:
        out_wav = media_dir / f"vendor_r{idx}.wav"
        t1 = time.perf_counter()
        model.chat(
            msgs=[sys_msg,
                  {"role": "user", "content": [row["prompt"]]}],
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=int(row.get("max_new_tokens") or 32),
            use_tts_template=True,
            generate_audio=True,
            output_audio_path=str(out_wav),
        )
        wall = time.perf_counter() - t1
        # HARD GATE — a generate_audio=True request that produced no wav
        # is a FAILED request; recording its wall would bank a
        # no-speech timing as a speech number (the 2026-08-08 silent
        # failure). Fail the whole cell loudly instead.
        if not out_wav.exists():
            raise SystemExit(
                f"vendor_cell: request r{idx} produced no wav at "
                f"{out_wav} — refusing to record a speech wall for a "
                f"run that synthesized no speech")
        return {"wall_s": wall,
                "sha256": hashlib.sha256(
                    out_wav.read_bytes()).hexdigest()}

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
            "ref_audio": str(ref_wav),
            "sys_prompt_mode": "audio_assistant",
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
