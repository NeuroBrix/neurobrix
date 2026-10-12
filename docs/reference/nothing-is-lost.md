# Nothing is lost

Owner's rule, 2026-10-12. One night, four delegated tasks ran until morning and were
never collected; their work appeared nowhere. A task that ran a night without a report
is a fault of method, not an incident.

## The rule

1. **Every piece of work launched** — a delegated agent, a tmux chain, a script, a GPU
   computation — has, **before** it starts, one line in the report: what it does, where
   it writes, how its end will be known; and a watcher on that end. Nothing runs "in the
   air".
2. **A delegated task writes its own deliverable to disk**, inside the campaign, at a path
   fixed when it is launched — not only in its answer. Its report exists even if nobody
   collects it. When it ends, it is collected and closed in the same turn.
3. **At every wake, every compaction and every end of turn**: the real list of live
   delegated tasks and chains is read from the system and compared with the report; every
   gap is corrected and noted.
4. **Recovery is owed, not optional**: work that was left uncollected is found again from
   its traces (campaign files, session journals), written into the report under
   "RECOVERED", and integrated; what is truly lost is named with its cause.

## Why a rule and not a habit

A delegated task's answer is one message; if the turn that should read it never comes,
the answer is gone with it. A file in the campaign survives the turn, the session and the
compaction. The watcher turns "I will check later" into an event. The comparison at every
wake makes an uncollected task visible before it has cost a night.

Related: [proving by doors](proving-by-doors.md) — the comparison is a census; the fixed
path and the watcher are the door.
