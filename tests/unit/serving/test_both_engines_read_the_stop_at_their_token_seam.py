"""Both engines' per-token seam reads the stop flag (Studio request 3, R30).

The compiled generator is driven for real on CPU (torch logits, its own
sampler). The triton generator's `step` is driven for real too, on CPU, with
its device-touching collaborators (the sampler, the context upload) replaced by
stand-ins: the seam under test is pure Python, and the triton module must reach
it without torch (R33 — `neurobrix.decode_stop` is stdlib-only, asserted below
in a clean interpreter).

Injection: with the `decode_stop.requested` call removed from either
generator's `step`, its test ran to max_tokens and went RED; restored, green.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from neurobrix import decode_stop

_CFG = {"max_tokens": 10, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "repetition_penalty": 1.0, "vocab_size": 8}


def _drive(gen, logits, stop_after: int):
    """Run the generator's own loop; set the stop flag once `stop_after`
    tokens exist. Returns (steps taken, honoured_at)."""
    event = threading.Event()
    decode_stop.bind(event)
    try:
        steps = 0
        for step_idx in gen:
            _, done = gen.step(logits, step_idx)
            steps += 1
            if steps == stop_after:
                event.set()   # what the daemon's control thread does
            if done:
                break
        return steps, decode_stop.honoured_at()
    finally:
        decode_stop.unbind()


def test_the_compiled_generator_stops_at_the_next_token():
    torch = pytest.importorskip("torch")
    from neurobrix.core.module.autoregressive.generator import AutoregressiveGenerator
    gen = AutoregressiveGenerator(dict(_CFG))
    logits = torch.zeros(1, _CFG["vocab_size"])
    steps, honoured = _drive(gen, logits, stop_after=3)
    # the flag is read at the token AFTER it was set: one more token, then done
    assert steps == 4 and honoured == 4 and gen.num_generated == 4 and gen.is_done


def test_the_compiled_generator_without_a_bound_flag_runs_its_length():
    torch = pytest.importorskip("torch")
    from neurobrix.core.module.autoregressive.generator import AutoregressiveGenerator
    gen = AutoregressiveGenerator(dict(_CFG))
    for step_idx in gen:
        _, done = gen.step(torch.zeros(1, _CFG["vocab_size"]), step_idx)
        if done:
            break
    assert gen.num_generated == _CFG["max_tokens"] and decode_stop.honoured_at() is None


def test_the_triton_generator_stops_at_the_next_token(monkeypatch):
    from neurobrix.triton import generator as tg
    # device-touching collaborators replaced; `step`'s own logic runs as shipped
    monkeypatch.setattr(tg, "NBXTensor", SimpleNamespace(from_numpy=lambda a: a))
    monkeypatch.setattr(tg, "DeviceAllocator", SimpleNamespace(set_device=lambda i: None))
    gen = tg.TritonGenerator(dict(_CFG))
    gen._sampler = lambda logits, input_ids=None: SimpleNamespace(item=lambda: 1)
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    # the logits as the Triton head hands them: fp32 (the generator widens any other dtype first, generator.py)
    logits = SimpleNamespace(ndim=2, shape=(1, _CFG["vocab_size"]), _dtype=NBXDtype.float32)
    steps, honoured = _drive(gen, logits, stop_after=2)
    assert steps == 3 and honoured == 3 and len(gen.get_generated_tokens()) == 3 and gen.is_done


def test_the_stop_seam_is_stdlib_only_r33():
    code = ("import sys; import neurobrix.decode_stop as d; "
            "bad = [m for m in ('torch', 'numpy', 'triton') if m in sys.modules]; "
            "print(','.join(bad))")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "", f"decode_stop pulled in {out}"


def test_a_flag_bound_on_one_thread_is_invisible_to_another():
    event = threading.Event()
    event.set()
    decode_stop.bind(event)
    seen = []
    try:
        t = threading.Thread(target=lambda: seen.append(decode_stop.requested(1)))
        t.start(); t.join()
        assert seen == [False] and decode_stop.requested(1) is True
    finally:
        decode_stop.unbind()
