# Replay recording in a one-shot request — the decision, and the measurement

**Decision: recording does not run in a one-shot request, and by default it does
not.** `NBX_TRITON_REPLAY` is opt-in and appears nowhere in `cli/` or `core/`,
so a plain `neurobrix run` never records. That default is correct. Turning it on
is a deliberate choice and the engine now says once, on stderr, what it costs.

## The two figures, measured
Apple M4 Pro, TinyLlama-1.1B-Chat-v1.0, 120 generated tokens, greedy,
`NBX_FORCE_RAND_SEED=1234`, each arm its own process under a 4096 MB floor.
**Byte-identical output in every arm** — recording changes cost, not answers.

    short prompt (~6 tokens)     plain  19.71 s   recording  21.20 s   **+7.6%**
    long prompt (~480 tokens)    plain 282.05 s   recording 2257.70 s  **8.0x**

**The cost is context-dependent and a single number misstates it.** An earlier
note in this campaign reported "8.0x" as the cost of recording; that is the cost
at ~480 tokens of context. At a short prompt it is within noise of nothing.

## Why a one-shot request cannot recover it
**The recorded plan does not outlive its process.** Plans live on the sequence
object (`seq.__dict__["_replay_plans"]`). The only thing `replay.py` writes to
disk is the slab SIZE cache (`_store_slab_size` ->
`~/.neurobrix/replay_cache/<component>.json`, signature -> byte count). There is
no plan serialiser and no loader — checked in the code, not inferred from a
timing.

So a single `neurobrix run` records, replays within its own generation, and
discards the recording at exit. The next process starts again.

## It is not a failure to replay
Within one process the machinery works: **87.5% of sequence runs took the fast
path** in a 120-token generation (14 replayed, 2 declined). The cost above is
what recording and replaying cost at that context, paid again by the next
process.

## The honest tension, worth stating for whoever picks this up
Replay exists to remove the per-launch Python band — the module docstring
measures that band at ~0.57 ms/launch on the Ming denoiser. And per-launch host
time IS the dominant cost on this machine: a decode's kernels run at 2.7 GB/s
against 30-50 GB/s for the identical launch in a tight loop, because the GPU's
OS-managed clock falls during host gaps (measured: a 20 ms gap makes the same
kernel 4x slower). Replay aims at exactly the right target and does not hit it
here.

That is not a contradiction to explain away; it is the next question. Nothing in
this document claims why.

## Not measured
Whether `serve` and `chat` — processes that outlive a request — recover the
cost. That is where the design intends the payback and it has not been measured
on this machine.
