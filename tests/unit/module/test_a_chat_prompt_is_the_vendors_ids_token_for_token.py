"""A chat request is prefilled with the ids the vendor's own tokenizer gives it, token for token
(transformers `apply_chat_template(messages, add_generation_prompt=True)`, tokenize=True).

The vendor encodes a rendered chat with no special token added (`add_special_tokens=False`): the template has
already written every special token it wants. And a SentencePiece tokenizer whose `tokenizer_config.json` says
`"legacy": false` puts the word-start prefix only at the start of the input (`_get_prepend_scheme` -> "first"),
never again after an added token, whatever the `tokenizer.json` normalizer says.

Measured red on the Mac, 2026-11-19, engine main 79b020076b35, TinyLlama-1.1B-Chat-v1.0 served for 5 turns:
the engine prefilled 25, 94, 165, 232, 300 tokens where the vendor's are 23, 90, 159, 224, 290: a `<s>` the
template never wrote (the post-processor's), and one extra `▁` after every `</s>` (the `Prepend` normalizer
run again on each segment after an added token). The vendor's greedy answers were measured on those wrong prompts.
granite-3.1-1b-a400m-instruct (byte-level BPE, no prefix, no BOS added) was equal and must stay equal.

The vendor's ids are recorded in data/chat_prompt_vendor_ids.json (transformers 5.23.0, the 5 requests of the Mac
baseline). The tokenizer files are the models' own, read from the snapshot directory NBX_VENDOR_SNAPSHOTS
(default ~/Mounts/Super-HF-Snapshots); a host without them skips by name.

Run: python -m pytest tests/unit/module/test_a_chat_prompt_is_the_vendors_ids_token_for_token.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from neurobrix.core.module.tokenizer.sp_tokenizer import load_tokenizer_from_path  # noqa: E402

RECORD = json.loads((Path(__file__).parent / "data" / "chat_prompt_vendor_ids.json").read_text(encoding="utf-8"))
SNAPSHOTS = Path(os.environ.get("NBX_VENDOR_SNAPSHOTS", str(Path.home() / "Mounts" / "Super-HF-Snapshots")))


@pytest.mark.parametrize("model", sorted(RECORD["models"]))
def test_every_turn_is_prefilled_with_the_vendors_ids(model):
    folder = SNAPSHOTS / model
    if not (folder / "tokenizer.json").exists():
        pytest.skip(f"{model}: no tokenizer files under {SNAPSHOTS} (set NBX_VENDOR_SNAPSHOTS)")
    tok = load_tokenizer_from_path(folder)
    wrong = []
    for k, turn in enumerate(RECORD["models"][model], 1):
        ours = [int(t) for t in tok.apply_chat_template(turn["messages"], tokenize=True, add_generation_prompt=True)]
        if ours != turn["ids"]:
            at = next((i for i, (a, b) in enumerate(zip(ours, turn["ids"])) if a != b), min(len(ours), len(turn["ids"])))
            wrong.append(f"turn {k}: {len(ours)} ids against the vendor's {len(turn['ids'])}, first difference at "
                         f"{at}: ours {ours[at:at + 3]} vendor {turn['ids'][at:at + 3]}")
    assert not wrong, f"{model}: " + "; ".join(wrong)


def test_two_threads_encoding_at_once_each_get_the_vendors_ids():
    """Serving encodes on more than one thread (the HTTP thread checks the context, the worker prefills), with ONE
    tokenizer. Whether a segment is prefixed is a per-call fact: kept on the shared normalizer or pre-tokenizer, one
    thread's call changes another's (qa, 2026-11-19: 5 wrong prompts of 12000 with two threads). The thread switch is
    forced as often as the interpreter allows, so a shared per-call state shows within a few thousand encodes."""
    import sys
    import threading
    model = "TinyLlama-1.1B-Chat-v1.0"
    folder = SNAPSHOTS / model
    if not (folder / "tokenizer.json").exists():
        pytest.skip(f"{model}: no tokenizer files under {SNAPSHOTS} (set NBX_VENDOR_SNAPSHOTS)")
    tok = load_tokenizer_from_path(folder)
    turns = RECORD["models"][model]
    wrong = []

    def work(k):
        turn = turns[k]
        for _ in range(3000):
            ours = [int(t) for t in tok.apply_chat_template(turn["messages"], tokenize=True, add_generation_prompt=True)]
            if ours != turn["ids"]:
                wrong.append(k)

    kept = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=work, args=(k,)) for k in (0, 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(kept)
    assert not wrong, f"{len(wrong)} of 6000 encodes from two threads gave other ids than the vendor's"
