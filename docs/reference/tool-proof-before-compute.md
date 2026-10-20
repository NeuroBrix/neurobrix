# A tool is proven before any large computation or queue

Owner's permanent rule, 2026-10-20 16:25. Our hours were lost in our own tools, not in the engine. Between
2026-10-19 and 2026-10-20 alone:

- the gate's scope probe hung on a walk that never ended (two gates lost);
- two gate schedulers placed cells on the same card (holds read, never taken) and three cells died out of memory;
- run timeouts taken from carried times, not from the real request (Wan cells stopped at step 19 of 50);
- a "32 GB" census run on a profile of two 16 GB cards, which overwrote rows of the 16 GB table;
- a census forked one worker per core (three panics on the Mac);
- the certifier synthesised GEMM keys at the bucket's top (86 keys failed);
- the softmax oracle took minutes per candidate (a 4 h certification timed out);
- a carried verdict hid the Qwen3-VL decode regression from 2026-10-16 to 2026-10-20;
- a filler stopped on "queue empty".

## The rule

Before launching a computation longer than 30 minutes, or any queue (gate, census, certification, replay, filler):

1. **The tool runs first on the smallest real case** — one model, one key, one cell.
2. **Read what it WROTE yourself** — the result line, the log, the report, the hold, the resume point — and check
   the profile, the card and its memory class, the timeout derived from the real request (never a carried time),
   and the branch or tree.
3. **Estimate the total duration from that case and write it down before launching.**
4. **A watcher checks during the run that the log grows**, and wakes on a silent log.

A modified tool goes through steps 1 and 2 again. A computation launched without this proof and then lost is a
method fault, and is written down as one.

Related: `proving-by-doors.md` (a door says "never"), `vacuous-gates-register.md` (a gate is trusted only once seen
failing).
