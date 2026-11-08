"""The child process of test_a_turn_prefills_only_its_new_tokens.py: one engine, two arms of the same conversation.

    python session_kv_driver.py --model M [--hardware H] --mode triton|compiled --chunk C|0 --conv FILE --out FILE

The conversation file holds {"turn1": [messages], "turn2": [messages]}. A is the model's chat template over turn 1
(no generation prompt), the full ids the template over both turns with the generation prompt; A is their common
prefix. Each request decodes ONE token (its logits are the turn's), so a cache holds exactly what was prefilled.

- arm "split": one session — turn 1 = A, then turn 2 = the full ids, served from the session's cache.
- arm "whole": a fresh session — the full ids in one request.

Recorded per arm, as the engine computed them (InferenceEngine, the daemon's engine; its own session record): the sha256
of every layer's K and V bytes over the full ids' positions, read back from the cache; the sha256 of the turn's
first-token logits (NBX_DUMP_LOGITS, the flows' own dump); the rows the language model's executor RAN during the turn
(counted at its `run`, outside the session code that decides them) and the session's ledger. With --chunk C the plan
carries KVCachePlan.prefill_chunk = C, the field Prism will fill (docs/design/prism_kv.md §5bis).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile


def _rows(run_inputs) -> int:
    t = run_inputs.get("input_ids")
    if t is None:
        t = run_inputs.get("inputs_embeds")
    return int(t.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", default=None,
                    help="profile id; default: the machine's own profile for the visible cards (first launch's)")
    ap.add_argument("--mode", required=True, choices=["triton", "compiled"])
    ap.add_argument("--chunk", default="0", help="rows of the prefill chunk; 0: one chunk; 'plan': the plan's own")
    ap.add_argument("--conv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    from neurobrix.serving.engine import InferenceEngine
    from neurobrix.serving.session import session_ceiling
    from nbxtensor.memory.kv_session import KVSession
    from neurobrix.core.runtime.prompt_length import _flatten_tokenizer_output
    from neurobrix.core.runtime.lm_facts import session_lm_name

    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    engine = InferenceEngine(a.model, a.hardware or get_or_create_default_profile(), mode=a.mode)
    engine.load()
    kv_plan = engine._plan.kv_cache_plan
    if a.chunk != "plan":
        kv_plan.prefill_chunk = int(a.chunk) or None
    conv = json.load(open(a.conv))
    tok = engine.tokenizer
    full = [int(t) for t in _flatten_tokenizer_output(
        tok.apply_chat_template(conv["turn1"] + conv["turn2"], tokenize=True, add_generation_prompt=True))]
    first = [int(t) for t in _flatten_tokenizer_output(
        tok.apply_chat_template(conv["turn1"], tokenize=True, add_generation_prompt=False))]
    la = 0
    while la < min(len(first), len(full)) and first[la] == full[la]:
        la += 1
    gen_info = engine._pkg.topology.get("flow", {}).get("generation", {})
    lm = engine._executor.executors[session_lm_name(gen_info, engine._executor.executors)]
    counted = {"rows": 0}
    orig_run = lm.run

    def counting_run(run_inputs, *args, **kwargs):
        counted["rows"] += _rows(run_inputs)
        return orig_run(run_inputs, *args, **kwargs)
    lm.run = counting_run

    decode = {"global.max_tokens": 1, "global.temperature": 0.0}
    dump = os.path.join(tempfile.mkdtemp(dir=os.environ.get("NBX_SESSION_KV_TMP")), "logits.json")

    def turn(ids, kv):
        counted["rows"] = 0
        os.environ["NBX_DUMP_LOGITS"] = dump
        try:
            r = engine._generate_from_token_ids(ids, decode, kv_session=kv)
        finally:
            os.environ.pop("NBX_DUMP_LOGITS", None)
        logits = np.asarray(json.load(open(dump))["logits"], dtype=np.float32)
        return {"rows_run": counted["rows"], "ledger": dict(kv.last_turn), "text": r.get("text", ""),
                "logits_sha": hashlib.sha256(logits.tobytes()).hexdigest(), "argmax": int(logits.argmax())}

    def kv_shas(kv, n):
        handle = kv.held(lm)
        if handle is None:
            raise RuntimeError("the session holds no cache after its turn")
        out = []
        if a.mode == "triton":
            import ctypes
            from nbxtensor.memory.allocator import DeviceAllocator
            cache = handle
            positions = -(-n // cache.page_tokens) * cache.page_tokens
            for layer in range(cache.num_layers):
                row = []
                for which in ("k", "v"):
                    g = cache.gather_pages(layer, which, positions, cache.num_kv_heads, cache.dtype)
                    g = g.narrow(2, 0, n).contiguous()
                    buf = (ctypes.c_char * g._nbytes)()
                    DeviceAllocator.set_device(g._device_idx)
                    DeviceAllocator.sync_device()
                    DeviceAllocator.memcpy(ctypes.addressof(buf), g.data_ptr(), g._nbytes, 2)
                    row.append(hashlib.sha256(bytes(buf)).hexdigest())
                out.append(row)
        else:
            cache = handle.cache
            for layer in range(cache.config.num_layers):
                k, v = cache.get_cache(layer)
                if int(k.shape[2]) != n:
                    raise RuntimeError(f"layer {layer} holds {int(k.shape[2])} positions, expected {n}")
                out.append([hashlib.sha256(k.contiguous().cpu().numpy().tobytes()).hexdigest(),
                            hashlib.sha256(v.contiguous().cpu().numpy().tobytes()).hexdigest()])
        return out

    ceiling, source = session_ceiling(kv_plan)
    res = {"model": a.model, "mode": a.mode, "chunk": kv_plan.prefill_chunk,
           "chunk_reason": getattr(kv_plan, "prefill_chunk_reason", ""), "len_A": la, "len_full": len(full),
           "page_tokens": None}
    split = KVSession(ceiling, source)
    res["split_turn1"] = turn(full[:la], split)
    res["split_turn2"] = turn(full, split)
    res["split_kv"] = kv_shas(split, len(full))
    whole = KVSession(ceiling, source)
    res["whole"] = turn(full, whole)
    res["whole_kv"] = kv_shas(whole, len(full))
    # R33: a Triton session's process never loads torch, at import or at execution
    res["torch_loaded"] = "torch" in sys.modules
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
