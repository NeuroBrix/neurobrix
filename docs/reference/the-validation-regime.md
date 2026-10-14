# The validation regime — a gate proves, a judgment judges, once each

The owner's rules (2026-10-13 14:20, via the supervisor), measured on the mq25 landing gate: 157 runs,
16.5 card-hours (2.2 h loading, 15.9 h executing); 12 runs — Allegro-TI2V x4 at 1.2-1.6 h, Allegro x4,
Wan2.1-I2V-14B x4 — took 9.7 of the 16.5; the two memory classes doubled everything; the cards were
55/93/65/61 % busy and 36/4/24/34 % idle. The candidate gate before it: 35.8 card-hours. That is why a
batch took a day. From the next batch:

1. **One class per gate: the card's real VRAM (32 GB here).** The 16 GB class is not replayed per batch.
   Prism LISTS, without executing anything, the models whose 16 GB plan differs from the 32 GB plan
   (strategy or streaming); only those get a 16 GB run, and only when the batch touches Prism, the
   streaming strategies or the tiling. One full 16 GB pass before the stable release, once.
   **Precision (2026-10-13 18:05):** a model validates ONCE per distinct Prism plan, not per card class; two cards that
   give the same plan make one run. The listing of plans is Prism's, computed without executing (the explained plan under
   each saved profile), and the scheduler places a plan-identical cell on any card, the row naming the card it ran on.
2. **A gate run lasts at most 10 minutes.** A gate proves rc 0, zero certified miss and the agreement of
   the arms — not quality. A video or image model's gate request is the smallest that exercises the
   path: the steps, the frames, the resolution at the model's floor. Quality is judged at the real
   request ONCE per model and per branch (`judgments.jsonl`), replayed only when the computed scope
   reaches that model. Allegro-TI2V at 1.5 h per run, four runs per gate, is forbidden: its gate request
   is reduced, and the kernel's slowness (hi + lo) stays an optimisation workstream after the stable.
3. **Computed scope, without exception.** A batch touching two lines does not replay 98 models. When the
   tool answers "everything", the report names the touched units and why each reaches the whole
   catalogue; otherwise the batch is split — one fix per batch, landed at once — instead of stacking
   fourteen shared units.
4. **GPU: a card at 0 % outside a cell's loading phase is a fault, noted.** The scheduler places the next
   cell without waiting for a tally.
5. **The gate is ONE script** — scheduler, rows, tally — that runs alone and wakes the agent ONCE. No
   agent turn while it runs, except on a new red. Reports keep sixty lines; intermediate diagnostics are
   not narrated.

What exists today and what this changes: `tools/regression_matrix.py` (rows, carries, known-cause
records), `tools/gate_scope.py` (the computed scope), the campaign scheduler and tally. Done: the Prism listing of "16 GB plan differs from 32 GB plan" is `tools/plan_listing.py` (the cell's request
explained under each profile, no execution; the Triton explanation needs a visible card). Owed by this regime: the per-model gate requests at the model's floor (data-driven, in the family and
model registries, never in a tool), `judgments.jsonl` as the one record of quality per model and branch,
and the gate as one script with one wake.
