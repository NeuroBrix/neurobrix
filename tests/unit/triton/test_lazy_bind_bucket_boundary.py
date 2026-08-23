"""Integration test — the lazy weight bind across a KV bucket boundary.

No row in the full-zoo gate crosses a bucket boundary (the LLM rows
generate 8-48 tokens from short prompts, staying inside the first
256-token bucket), so the lazy-bind skip (`18d8380`) had never crossed
one under proof. This test crosses at least one by construction — a
~200-token prompt plus 120 generated tokens passes 256 — and compares
the output BYTE FOR BYTE against the same generation with the skip
disabled (NBX_LAZY_BIND=0).

WHAT THE BOUNDARY EXERCISES — established by this test's own first
failure. The initial version asserted the stale-yes -> late-bind marker
and failed; reading `signature()` explained why: the signature is fully
determined by PRIOR-step state (the KV owner's bucket contribution
advances during the previous step's execution, before `would_replay`
runs), so in a single-process run `would_replay` and `maybe_run` always
agree and the late-bind guard cannot fire. It stays as defence in depth
(warm/serving paths, future signature changes) with its diagnostic
marker. What the boundary DOES exercise is the full cycle
skip -> eager re-bind -> re-record -> skip, which this test proves via
the double "plan frozen" recording plus the byte equality.

TinyLlama keeps it fast enough for the suite (~1.1B, greedy, bucket 256
like the canonical row). Runs a subprocess per arm (~2-3 min on a free
V100); skipped without a GPU.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:  # pragma: no cover - shim
        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[3]

# ~200 tokens of prompt: 40 repetitions of a ~5-token sentence.
_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 40
           + "\nContinue the story:")


def _has_gpu() -> bool:
    r = subprocess.run(["nvidia-smi", "--query-gpu=count",
                        "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def _run(arm_env: dict, tag: str, outdir: Path) -> tuple[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": env.get("NBX_TEST_GPU", "2"),
        "NBX_TRITON_REPLAY": "1",
        "NBX_REPLAY_KV_DECODE": "1",
        "NBX_REPLAY_GRAPH": "1",
        "NBX_FORCE_RAND_SEED": "1234",
        "NBX_LAZY_BIND_DIAG": "1",
    })
    env.update(arm_env)
    out = outdir / f"{tag}.txt"
    r = subprocess.run(
        ["python3", "-u", "-m", "neurobrix", "run",
         "--hardware", "v100-32g",
         "--model", "TinyLlama-1.1B-Chat-v1.0",
         "--prompt", _PROMPT, "--max-tokens", "120",
         "--temperature", "0", "--triton", "--output", str(out)],
        env=env, cwd=str(REPO), capture_output=True, text=True,
        timeout=1200)
    assert r.returncode == 0, (
        f"{tag} failed rc={r.returncode}:\n{r.stdout[-800:]}\n{r.stderr[-400:]}")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return sha, r.stdout


def test_lazy_bind_is_byte_exact_across_a_bucket_boundary() -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        sha_off, _ = _run({"NBX_LAZY_BIND": "0"}, "off", outdir)
        sha_on, log_on = _run({"NBX_LAZY_BIND": "1"}, "on", outdir)

        assert sha_on == sha_off, (
            f"outputs differ across a bucket boundary: lazy={sha_on[:12]} "
            f"vs eager={sha_off[:12]} — the parked-bind path corrupted "
            f"the op-by-op fallback")

        # ACTIVATION PROOF — what the boundary ACTUALLY exercises.
        #
        # First version asserted the late-bind marker here and FAILED,
        # and the failure was the finding: reading `signature()` shows it
        # is fully determined by PRIOR-step state (the KV owner's bucket
        # contribution advances during the previous step's execution,
        # before would_replay runs), so the stale-yes -> late-bind branch
        # cannot fire in a single-process run. It remains as defence in
        # depth for warm/serving paths and future signature changes, with
        # its diagnostic marker for the day it does.
        #
        # What a boundary crossing DOES exercise, and what this asserts:
        # skip (bucket-1 replays) -> would_replay False at the boundary
        # (new signature, no plan) -> EAGER re-bind -> re-record -> skip
        # again in bucket 2. The proof is at least two "plan frozen"
        # recordings in one run, which cannot happen without the
        # boundary, plus the byte equality above which proves the
        # skip/eager transitions corrupted nothing.
        frozen = log_on.count("plan frozen")
        assert frozen >= 2, (
            f"only {frozen} 'plan frozen' recording(s) in the run — the "
            f"generation did not cross a bucket boundary, so the "
            f"skip->eager->re-record->skip cycle was never exercised; "
            f"lengthen the prompt or the generation")


if __name__ == "__main__":
    if not _has_gpu():
        raise SystemExit("no GPU")
    test_lazy_bind_is_byte_exact_across_a_bucket_boundary()
    print("PASS: byte-exact across the boundary; skip->eager->re-record->skip cycle proven")
