# What NeuroBrix Studio asks of the engine — a list, not a work order

Written 2026-09-13 from the channel that exists: the Studio repository's
tracker issue #9 ("Tracker: engine capabilities Studio needs — accepted,
being fixed at the source"), its integration plan
(`docs/studio-engine-integration-plan.md` in `NeuroBrix/neurobrix-studio`),
and issues #5 and #8. Each request below was re-checked against this engine's
source at `46479ae` on the day it was written; the line says what the engine
does today, what the request needs from it, and which architecture rule it
touches. **Nothing here is scheduled by this document** — it exists so the
requests are visible when someone decides.

The tracker's own instruction to Studio stands and is repeated here because it
binds the engine side too: no client-side detokeniser, no version guessing, no
killing the daemon in place of cancellation, no polling workaround for the
blocked socket, no temporary shim. Where a Studio feature depends on one of
these, it reports the capability unavailable with a reason.

| # | request | what the engine does today (read at 46479ae) | what it needs | rule it touches |
|---|---|---|---|---|
| 1 | **Machine-readable discovery without loading a model** — engine version, protocol version, endpoint, supported operations, where things live on disk | The daemon answers `status` (`serving/server.py:404`) once it is up with a model; there is no version or protocol field in the envelope (`serving/protocol.py`: `method` + `params`, no ids, no negotiation); the CLI's `registry` output is human-facing (`cli/commands/registry.py`), `--json` exists on `validate` only (`cli/__init__.py:512`) | One command and one daemon method that report the same record, read from the engine's own sources of truth (`pyproject` version, the protocol module's constant, the cache root the registry uses) — no constant duplicated in Studio | Data-driven: the record is read, never typed twice. R34 unaffected. **Landed 2026-09-16 (CLI half):** `neurobrix info --json` — version, paths, models, hardware profiles, compute stack (`docs/reference/json-output.md`). **Daemon half landed the same day:** `status` returns the same identity record (`serving/engine.py::daemon_identity`: engine, protocol, endpoint, operations) and every envelope carries `protocol` + `engine`. |
| 2 | **Decoded text deltas in the live stream** | The streaming callback emits `{"step", "n", "token": token_id, "done"}` (`serving/server.py:338-342`); the full text arrives with the terminal response (`:386`) | The stream carries the text the engine's own tokenizer decodes for each delta; both engines (compiled, triton) — the tokenizer runner is engine-internal already (Tokenizer Engine) | R30 (mode universality): one seam in each generator, same event shape. R34: decoding stays inside the engine; Studio never ships a tokenizer. |
| 3 | **Acknowledged cancellation** — stop a running generation, confirmed only when native work has stopped | No `cancel` method in the dispatcher (`serving/server.py:319-409`); a send failure on a closed socket may interrupt streaming, which is not an acknowledgement | An operation identity per request, a cancel method that flips a flag the decode loop reads at each step, and an acknowledgement sent after the device has synced — never on disconnect or timeout | R30: the flag is read in both engines' loops. The triton loop must remain torch-free (R33). |
| 4 | **A daemon that stays responsive during inference** — control requests answered while a generation runs | Connections are accepted and served one at a time in the accept loop (`serving/server.py:282` → `_dispatch`); a running generation blocks `status` and would block `cancel` | A control path that does not wait on the inference path: one inference at a time is fine, but status and cancel must be answered while it runs. The tracker names this the governing item: cancellation on the current sequential dispatcher would never arrive during the generation it means to stop | Serving architecture (not a compute rule). Existing debts on the same surface: D-SERVE-WARM-REFREEZE, D-SERVE-WARM-KV-GROWTH-ASYMMETRY, P-SERVE-UNLOAD-LIVE-SET — a redesign of the dispatcher should read them first. |
| 5 | **JSON for hub queries, inventory, removal; NDJSON for long-running CLI progress; diagnostics to stderr** | `registry` prints tables and progress bars for a person; the import path already has licence and gated-access requirements the plan asks Studio to preserve | `--json` on the registry commands, one line per progress event on stdout, everything else on stderr; the licence prompt becomes an explicit parameter, never a silently passed flag | Data-driven output (Output Dispatch Engine's discipline applied to the CLI's own records). **Landed 2026-09-16 (read half):** `--json` on every read command, one record on stdout, human lines on stderr, schema versioned per command and gated by a parse test. NDJSON progress for the long-running commands and the explicit licence parameter are still owed. |
| 6 | **Download / extraction / installation lifecycle stays in the engine** — real byte counts, phases, cleanup of incomplete work, a model visible only after success | The importer resumes an interrupted download (D-IMPORT-RESUMABLE-DOWNLOAD, closed 2026-09-13) and installs in phases; the phases are logged, not emitted as records | The same lifecycle emitting one record per phase (item 5's NDJSON), the "visible only after success" rule stated by the installer rather than assumed | Hub doctrine (five layers): the installed set is the engine's, Studio reads it through the engine. |
| 7 | **A compatibility handshake** — Studio refuses an incompatible engine before work starts | Nothing negotiates; a mismatch surfaces as a failed method | A protocol version in item 1's record and in every envelope; the daemon refuses a client outside the range it declares | Release Alignment Doctrine: the protocol version is versioned with the engine, one number everywhere. **Landed 2026-09-16 (first half):** `PROTOCOL_VERSION` in `serving/protocol.py`, in every envelope and in the identity record; the daemon does not yet refuse a client outside a declared range (no request carries a client version) — owed with the dispatcher work of items 3-4. |
| 8 | **Windows instance isolation** — the Windows loopback port is fixed; `NBX_SOCKET_PATH` isolates only the Unix socket | `serving/protocol.py`: Unix socket on macOS/Linux, a fixed loopback TCP port on Windows | A per-instance port (or named pipe) chosen and reported through item 1, so two Studios or a Studio and a CLI daemon do not collide | R23 (hardware universality) applied to the host OS: no platform is the exception. |

## What this list is not

* Not a promise of order or date. The tracker says "the next version that
  comes out of the engine" and this document does not overrule it.
* Not a design. Items 3 and 4 in particular touch the serving dispatcher's
  shape, which three open debts already describe; the design belongs to the
  chantier that takes them together.
* Not verified against a running Studio: the Studio repository is a SvelteKit
  scaffold with a Rust `greet` command at the time of writing (its own plan says
  so); every claim here about the engine was read in the engine's source, every
  claim about Studio in its repository.

## Where to update this

When one of the eight lands, the row gets the commit and the date and stays
(a request that was met is part of the record); a new request from the Studio
channel is appended as row 9, never inserted.
