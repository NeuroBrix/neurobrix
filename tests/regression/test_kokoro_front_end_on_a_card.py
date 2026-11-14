"""Kokoro's text front end on a card, end to end through `neurobrix run`, both engines: the tagger component the engine
runs gives the vendor's tags, and the phonemes are the front end's reading of those tags.

The card-less half (tokens, rows, the lexicon pass, the word list) is `tests/unit/audio/
test_kokoro_reads_english_as_its_makers_front_end.py`; this file proves the one part it cannot: the `text_tagger`
component run by the engine's own executor, on the request's card, compiled and triton. Per judge text and voice
variant (us af_heart, gb bf_emma): the run's developer line `[Text front end] ... tags [...] phonemes "..."` holds the
tags spaCy gave (`data/kokoro_front_end_vendor.json.gz`) and the phonemes the front end gives for them card-less.

    CUDA_VISIBLE_DEVICES=<card> NEUROBRIX_PYTHON=<engine python> NEUROBRIX_CACHE=<cache holding the rebuilt Kokoro-82M> \\
        PYTHONPATH=src:nbxtensor/src pytest tests/regression/test_kokoro_front_end_on_a_card.py -p no:cacheprovider

Skipped by name without a card, or when the Kokoro-82M of the configured cache carries no tagged front end.
`NEUROBRIX_HARDWARE`, when set, is the profile the runs are given (`--hardware`)."""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "unit" / "audio" / "data" / "kokoro_front_end_vendor.json.gz"
VOICES = {"us": "af_heart", "gb": "bf_emma"}
LINE = re.compile(r"\[Text front end\] (\S+) \((\w+)\): tags (\[.*?\]) phonemes (\".*\")\s*$", re.M)


def _container() -> Path:
    from neurobrix.core.paths import cache_dir
    root = cache_dir() / "Kokoro-82M"
    if not (root / "modules" / "g2p" / "front_end.json").is_file():
        pytest.skip(f"the Kokoro-82M installed in {cache_dir()} does not carry the tagged text front end")
    return root


def _has_card() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "" or shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def _cells():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        rows = json.load(f)["texts"]
    return [pytest.param(r, mode, id=f"{r['variant']}-{i}-{mode}")
            for i, r in enumerate(r for r in rows if r["corpus"] == "judge") for mode in ("compiled", "triton")]


@pytest.mark.parametrize("row,mode", _cells())
def test_the_engine_runs_the_tagger_and_reads_the_vendors_tags(row, mode, tmp_path):
    if not _has_card():
        pytest.skip("no card visible: the tagger component runs on a card here")
    root = _container()
    from neurobrix.core.module.audio import front_end as FE
    from neurobrix.core.module.audio.front_end.lexicon_pass import phonemize
    defaults = json.loads((root / "runtime" / "defaults.json").read_text())
    fe = FE.for_container(str(root), defaults["phoneme_lang"])
    lp = fe.lexicon_pass(row["variant"])
    text2, words, feats = lp.preprocess(row["text"])
    want, _ = phonemize([{"text": t, "whitespace": w, "tag": g} for t, w, g in row["tokens"]], lp, words, feats)
    out = tmp_path / "speech.wav"
    cmd = [os.environ["NEUROBRIX_PYTHON"], "-u", "-m", "neurobrix", "run", "--model", "hexgrad/Kokoro-82M", f"--{mode}",
           "--prompt", row["text"], "--speaker", VOICES[row["variant"]], "--output", str(out)]
    if os.environ.get("NEUROBRIX_HARDWARE"):
        cmd += ["--hardware", os.environ["NEUROBRIX_HARDWARE"]]
    env = {**os.environ, "NBX_LOG": "debug", "PYTHONUNBUFFERED": "1",
           "NBX_SOCKET_PATH": str(tmp_path / "s.sock")}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, cwd=str(REPO))
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"rc={proc.returncode}\n{log[-1500:]}"
    assert out.exists(), f"no speech written:\n{log[-800:]}"
    seen = LINE.findall(log)
    assert seen, f"the run printed no text front end line:\n{log[-1500:]}"
    comp, variant, tags, phonemes = seen[-1]
    assert (comp, variant) == (fe.component, row["variant"])
    assert json.loads(tags) == [g for _t, _w, g in row["tokens"]], "the card's tags are not spaCy's"
    assert json.loads(phonemes) == want
