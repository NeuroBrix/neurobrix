"""Prism prices a session's prefill chunk asking the running code generator ONCE (2026-11-09).

`prefill_chunk.certified_times` reads the certified time of every key of every candidate (thousands per plan on a MoE
model); `autotune_certified.served_certification` asked `running_generator()` for each, and on a process with no card
visible (a cardless test, a census, `--explain-plan` on a host) each ask probed the CUDA driver again (cuInit raised,
nothing cached): the prism unit tests that solve a streamed LLM took over 15 minutes, measured with py-spy at
launcher.py:503 under prefill_chunk.served.

What this test does if the code is wrong: the generator is asked once per key priced — red (seen red on lot-a
3508bf42d: 5 asks for 5 keys)."""
from nbxtensor.ops import autotune_certified as AC
from neurobrix.core.prism import prefill_chunk as P


def test_the_generator_is_asked_once_for_every_key_priced(monkeypatch):
    asks = []
    monkeypatch.setattr(AC, "running_generator", lambda: asks.append(1) or "cuda-70-triton_3.8.0")
    monkeypatch.setattr(AC, "_load_class", lambda *a, **k: {f"k{i}": {"proof": {"best_ms": 0.5}} for i in range(5)})
    monkeypatch.setattr(AC, "entry_for_generator", lambda cert, gen: cert)
    served = P.certified_times("nvidia", "volta", 16)
    times = [served("nbxtensor.ops.matmul.matmul_kernel", "fp16", f"k{i}") for i in range(5)]
    assert times == [0.5] * 5
    assert len(asks) == 1, f"the generator was asked {len(asks)} times for 5 keys"
