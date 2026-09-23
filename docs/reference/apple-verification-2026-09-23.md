# Judged verification — Apple M4 Pro, 2026-09-23

**Re-run after the switchover and unchanged: 29 of 30 cells clean, ZERO autotune misses.**
The prune removed 956 entries and serving did not move, which is what the pre-flight
measurement predicted — all 956 carried no generator record, so register 56 already served
them to no card. Directory gate after the switchover: `rc=0`, 16 files, 0 refused.

Three modes, ten catalogue cells each, replaying the CENSUS's own recorded command.
Run on a QUIET host with the replay cache CLEARED before the arms, so a miss count is a
property of the certified directory and not of a warm cache (register 502).

| arm | cell | rc | autotune misses |
|---|---|---|---|
| `--triton` | swin2SR-classical-sr-x2-64 | 0 | 0 |
| `--triton` | swin2SR-realworld-sr-x4-64-bsrgan-psnr | 0 | 0 |
| `--triton` | swinir-classical-x2 | 0 | 0 |
| `--triton` | swinir-classical-x4 | 0 | 0 |
| `--triton` | real-esrgan-x2 | 0 | 0 |
| `--triton` | Real-ESRGAN-x4 | 0 | 0 |
| `--triton` | whisper-large-v3-turbo | 0 | 0 |
| `--triton` | Kokoro-82M | 0 | 0 |
| `--triton` | TinyLlama-1.1B-Chat-v1.0 | 0 | 0 |
| `--triton` | chatterbox | 137 | 0 |
| `--triton-sequential` | swin2SR-classical-sr-x2-64 | 0 | 0 |
| `--triton-sequential` | swin2SR-realworld-sr-x4-64-bsrgan-psnr | 0 | 0 |
| `--triton-sequential` | swinir-classical-x2 | 0 | 0 |
| `--triton-sequential` | swinir-classical-x4 | 0 | 0 |
| `--triton-sequential` | real-esrgan-x2 | 0 | 0 |
| `--triton-sequential` | Real-ESRGAN-x4 | 0 | 0 |
| `--triton-sequential` | whisper-large-v3-turbo | 0 | 0 |
| `--triton-sequential` | Kokoro-82M | 0 | 0 |
| `--triton-sequential` | TinyLlama-1.1B-Chat-v1.0 | 0 | 0 |
| `--triton-sequential` | chatterbox | 0 | 0 |
| `--compiled` | swin2SR-classical-sr-x2-64 | 0 | 0 |
| `--compiled` | swin2SR-realworld-sr-x4-64-bsrgan-psnr | 0 | 0 |
| `--compiled` | swinir-classical-x2 | 0 | 0 |
| `--compiled` | swinir-classical-x4 | 0 | 0 |
| `--compiled` | real-esrgan-x2 | 0 | 0 |
| `--compiled` | Real-ESRGAN-x4 | 0 | 0 |
| `--compiled` | whisper-large-v3-turbo | 0 | 0 |
| `--compiled` | Kokoro-82M | 0 | 0 |
| `--compiled` | TinyLlama-1.1B-Chat-v1.0 | 0 | 0 |
| `--compiled` | chatterbox | 0 | 0 |

**29 of 30 cells clean. ZERO autotune misses across all 30 cells.**

## The one exception, and why it is not a miss

`chatterbox --triton` exits **rc=137** — SIGKILL from macOS jetsam while loading `cond_enc`
under `TritonCFGEngine` with CFG enabled, which runs two conditioning passes and so carries
roughly double the resident weights. It is a MEMORY limit of an 18 GB card, not a coverage
gap: its autotune miss count is 0, and the same model runs clean in `--triton-sequential`
and `--compiled`. Same class as the 11 models already named APPLE (memory / rung) in
`docs/reference/catalogue-state.md`.

## What had to be fixed before this measurement meant anything

Three instruments were answering a different question than the one asked, and each
produced a CLEAN-looking answer:

1. **The census's recorded command could not be replayed** (`" ".join(argv)`), so
   `--prompt The quick brown fox ...` split at every space. 45 of 59 models were affected,
   and three verification cells — Kokoro-82M, TinyLlama, chatterbox — had NEVER been
   verified: they failed rc=2 and it read as a harness quirk. Fixed with `shlex.join`.
2. **A warm replay cache made misses vanish.** A key that is uncertified but cached is
   served silently and prints no miss line. Kokoro reported 8 misses cold and 0 warm with
   nothing certified in between. Register 502; the arms now clear the cache first.
3. **Reading an in-flight log as a result** — twice. The reader now refuses to report an
   arm without its closing marker.

## The census gap this exposed

Once the cells could actually run, they formed **73 keys the 3 106-key census never named**
— 65 from chatterbox in `--triton-sequential`, 8 from Kokoro in `--triton`. The census
shadow reads graphs without executing, so it under-predicts what a real run forms. All 73
were certified (`e279922a`), and this run confirms them on a cold cache.
