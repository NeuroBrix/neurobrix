#!/usr/bin/env python3
"""Long-form STT accuracy gate (D-STT-LONGFORM-CHUNKING, 2026-09-02).

A long transcript is judged against an independent full-work reference
(faster-whisper / NeMo on the same 600 s input) by PHRASE CONTAINMENT:
N evenly spaced 8-word phrases are sampled from the reference and each
must appear, normalized (lower-case, punctuation stripped, whitespace
collapsed), in the candidate. The gate passes at >= min_hits / N. Two
ASR engines disagree on rare words, so the bar is 8/10 by default —
strict enough to refuse a truncated or scrambled transcript outright
(a 105-byte truncation hits 0-1/10), loose enough to survive honest
per-engine word choices. Also prints the byte ratio (candidate /
reference) as a coverage sanity line.

    python3 benchmarks/harness/stt_longform_gate.py --candidate out.txt \\
        --reference validation_outputs/.../out_fw_1.txt [--n 10] [--min-hits 8]
"""
from __future__ import annotations

import argparse
import re
import sys


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--phrase-words", type=int, default=8)
    ap.add_argument("--min-hits", type=int, default=8)
    args = ap.parse_args()

    cand_raw = open(args.candidate).read()
    ref_raw = open(args.reference).read()
    cand, ref = _norm(cand_raw), _norm(ref_raw)
    ref_words = ref.split()
    if len(ref_words) < args.phrase_words * args.n:
        print(f"GATE ERROR: reference too short ({len(ref_words)} words)")
        return 2

    span = len(ref_words) - args.phrase_words
    hits = 0
    rows = []
    for i in range(args.n):
        start = (span * i) // max(1, args.n - 1)
        phrase = " ".join(ref_words[start:start + args.phrase_words])
        hit = phrase in cand
        hits += int(hit)
        rows.append((i, hit, phrase))
    ratio = len(cand_raw) / max(1, len(ref_raw))
    for i, hit, phrase in rows:
        print(f"  [{i:2d}] {'HIT ' if hit else 'MISS'}  {phrase}")
    verdict = "PASS" if hits >= args.min_hits else "FAIL"
    print(f"phrase containment {hits}/{args.n} (min {args.min_hits}) | "
          f"bytes {len(cand_raw)}/{len(ref_raw)} = {ratio:.2f} | {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
